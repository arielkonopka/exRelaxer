# kernels

`core/kernels.hpp`, `core/kernels.cpp`, `core/random.hpp`, `core/random.cpp`

The numeric core shared by every layer type: weighted sums, the sign
learning rule, a weight matrix laid out for SIMD, and thread sizing. Layers
own their weights in `kernels::weight_matrix` and call these kernels; the
[neuron](neuron.md) only receives the resulting sum.

## weight_matrix

Weights of `rows` neurons over `cols` inputs, stored in **blocks of 8 rows**;
inside a block the 8 rows' weights for one input are adjacent:

```
block b, input c, lane l  ->  data[(b * cols + c) * 8 + l]
```

One pass over the inputs then computes 8 weighted sums with SIMD: each input
value is broadcast and multiplied into 8 lanes at once. The last block is
padded with zero rows. Storage is 64-byte aligned.

| Member | Purpose |
|--------|---------|
| `weight_matrix(rows, cols, rowMajor)` | build from row-major values |
| `at(r, c)`, `copyRow(r, out)`, `setRow(r, values)`, `setColumn(c, values)` | element, row and column access |
| `appendColumns(count, rowMajor)` | add inputs (growth); existing weights keep their values |
| `reshape(rows, cols)` | a zero matrix reusing the storage (scratch tiles) |
| `multiply(x, sums, firstBlock, lastBlock)` | `sums[r] = Σ x[c] × w[r][c]` for the rows of those blocks |
| `learn(signs, delta, active, limit, firstBlock, lastBlock)` | `w[r][c] = clamp(w[r][c] + signs[c] × delta[r], ±limit)` for rows with `active[r]`; other rows untouched |

Every argument is a `std::span` (bounds-checked in debug builds); no kernel
takes a raw pointer. The same layout serves every layer type: in `dense` and
LocallyConnected2D the lanes are 8 neurons reading one input pool; in
Conv2D's forward pass they are 8 positions of a tile, multiplied by one
kernel; in its learning pass, 8 window inputs, multiplied by one channel's
deltas over the tile's positions (see [spatial](spatial.md#conv2d)).

## Determinism

Every weighted sum is accumulated **in input order starting from 0**, like a
plain scalar loop. The SIMD kernel keeps that order: lane `l` of a block adds
input 0, then input 1, ... of row `l`, so parallelism is across rows, never
within a sum. Results are therefore bit-identical to the scalar code
(`kernels::dot`), and independent of SIMD width, thread count and block
split. The learning update is elementwise, so it is exact too; inactive rows
are selected bit for bit, not recomputed.

The build compiles with `-ffp-contract=off`: otherwise a compiler may fuse a
multiply and an add into one FMA instruction, whose single rounding changes
the last bits. `KernelsTest` checks all of this against scalar references,
including partial blocks and the multi-threaded path.

## SIMD implementation

With GCC or Clang the kernels use vector extensions: one 8-float vector per
block with AVX, two 4-float halves without it (plain x86-64 / SSE2). The
multiply runs four blocks at a time, giving four independent addition chains
that hide the latency of each addition. The clamp uses vector compares and
selects that compile to `max`/`min`/`blend`, exactly matching `std::max` /
`std::min`. Other compilers use a portable loop with the same arithmetic.

`EXRELAXER_NATIVE=ON` builds for the build machine's CPU (e.g. AVX2). The
default build already reaches about the same speed on this machine.

Single thread, 1000 × 1000, per step (i7-12700H):

| Kernel | Scalar (old layout) | Blocked SIMD |
|--------|---------------------|--------------|
| weighted sums | 910 µs | 101 µs (SSE2) / 106 µs (AVX2) |
| learning | 430 µs | about the same: bound by reading and writing every weight |

## Parallelism

```cpp
inline constexpr size_t work_per_thread = 32768;
int threadsFor(size_t work);   // work / work_per_thread, capped; < 2 means serial
```

A thread must get enough multiply-adds to pay for waking it; on a hybrid CPU
idle pool threads spinning on sibling hyperthreads slow small teams further.
Large layers are memory-bound and keep scaling with threads. Each thread gets
one contiguous run of blocks, the same run in the forward pass and in
learning, so the weights it learns on are still in its cache: dynamic
scheduling balanced fast and slow cores better, but made step + reward on
1000 × 1000 1.6× slower.

## Performance

Best of 15 runs, i7-12700H (20 threads), default build, before and after
the move to layer-owned blocked weights (identical results):

| Workload | Before | After | Speed-up |
|----------|--------|-------|----------|
| dense 64 × 64, step | 4.6 µs | 1.7 µs | 2.7× |
| dense 256 × 256, step | 44 µs | 12 µs | 3.6× |
| dense 256 × 256, step + reward | 80 µs | 29 µs | 2.8× |
| dense 1000 × 1000, step | 74 µs | 29 µs | 2.5× |
| dense 1000 × 1000, step + reward | 966 µs | 78 µs | 12× |
| dense 4096 × 4096, step | 1023 µs | 897 µs | 1.1× (memory bandwidth) |
| dense 4096 × 4096, step + reward | 12.5 ms | 3.4 ms | 3.7× |
| reservoir 200 + 300, step | 28 µs | 12.5 µs | 2.2× |
| 8 layers × 16, step + reward | 3.3 µs | 1.5 µs | 2.2× |

Learning gained the most: it used to run serially, neuron by neuron. Very
small layers are dominated by the neuron dynamics (E-R's `log`), which is
per neuron and cannot be approximated without changing results.

## Random streams

`core/random.hpp`: `exr::reseed(seed)` and one `mt19937` per purpose
(`rng::WeightStream::Initial`, `Growth`, `Sensor`, plus jitter and the seeds
of neurons' spontaneous-firing generators). Weights are drawn row after row,
so each neuron's new weights come from its stream in neuron order, and
adding draws for one purpose never shifts another.
