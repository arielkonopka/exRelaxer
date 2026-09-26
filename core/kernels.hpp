// Numeric kernels shared by every layer type: weighted sums, the sign
// learning rule, and a weight matrix laid out for SIMD.
//
// Determinism: every weighted sum is accumulated in input order starting from
// 0, exactly like a plain scalar loop, whatever the SIMD width or thread count.
// The build disables floating-point contraction (-ffp-contract=off) so no
// compiler turns a multiply-add into an FMA and changes the last bits.
#pragma once
#include <cstddef>
#include <cstdint>
#include <new>
#include <span>
#include <vector>

namespace exr::kernels {

// Sequential dot product over x, in index order (w has at least x.size() entries).
inline float dot(std::span<const float> x, std::span<const float> w)
{
    float sum = 0.0f;
    for (size_t i = 0; i < x.size(); ++i)
        sum += x[i] * w[i];
    return sum;
}

inline float sign(float x) { return static_cast<float>((x > 0.0f) - (x < 0.0f)); }

// The learning rule for one neuron: w[i] += sign(x[i]) * delta, clamped to
// [-limit, limit], for every weight (x has at least w.size() entries). Only
// the sign of each input counts (see neuron.hpp).
void signRule(std::span<float> w, std::span<const float> x, float delta, float limit);

// out[i] = sign(x[i]), the part of the inputs the sign rule uses.
void signs(std::span<const float> x, std::span<float> out);

// Threads for a job of `work` multiply-adds: work / work_per_thread, capped at
// the OpenMP maximum; below 2 the caller runs serially. 1 without OpenMP.
//
// Measured on a 20-thread hybrid CPU (i7-12700H) with the SIMD kernels: a
// thread must get enough work to pay for waking it, and idle pool threads
// spinning on sibling hyperthreads slow small teams further. 256 x 256
// (65k): serial 11 us, 4 threads 25 us. Reservoir 200 + 300 x 500: serial
// 21 us, 4 threads 11 us. 1000 x 1000 is memory-bound and keeps scaling to
// all threads (146 us serial, 29 us on 20).
inline constexpr size_t work_per_thread = 32768;
int threadsFor(size_t work);

// Allocator aligning to a cache line, so no SIMD load or store of a weight
// block is split across two lines. (Its T* signatures are what the standard
// allocator interface requires; nothing else in the library holds one.)
template <typename T>
struct cache_aligned_allocator
{
    using value_type = T;
    static constexpr std::align_val_t alignment{64};

    cache_aligned_allocator() = default;
    template <typename U>
    cache_aligned_allocator(const cache_aligned_allocator<U>&) {}

    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), alignment)); }
    void deallocate(T* p, size_t) { ::operator delete(p, alignment); }
    template <typename U>
    bool operator==(const cache_aligned_allocator<U>&) const { return true; }
};

template <typename T>
using aligned_vector = std::vector<T, cache_aligned_allocator<T>>;

// Weights of `rows` neurons over `cols` inputs. Rows are stored in blocks of
// `lanes`; inside a block the lanes' weights for one input are adjacent:
//
//   block b, input c, lane l  ->  data[(b * cols + c) * lanes + l]
//
// so one pass over the inputs computes `lanes` weighted sums with SIMD, each
// lane still summing in input order. The last block is padded with zero rows.
class weight_matrix
{
public:
    static constexpr size_t lanes = 8;

    weight_matrix() = default;
    // `rowMajor` holds rows * cols values, row after row.
    weight_matrix(size_t rows, size_t cols, std::span<const float> rowMajor);

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t blocks() const { return (rows_ + lanes - 1) / lanes; }
    size_t paddedRows() const { return blocks() * lanes; }

    float at(size_t row, size_t col) const { return data_[index(row, col)]; }
    void copyRow(size_t row, std::span<float> out) const;
    void setRow(size_t row, std::span<const float> values);
    // values[r] into column `col` of every row (values has rows() entries).
    void setColumn(size_t col, std::span<const float> values);

    // Appends `count` inputs; `rowMajor` holds rows * count new weights, row
    // after row. Existing weights keep their values.
    void appendColumns(size_t count, std::span<const float> rowMajor);

    // Makes it a zero rows x cols matrix, reusing the storage (for matrices
    // used as scratch tiles, e.g. Conv2D's input patches).
    void reshape(size_t rows, size_t cols);

    // sums[r] = weighted sum of row r over x (cols() entries), for the rows
    // of blocks [firstBlock, lastBlock). `sums` has paddedRows() entries.
    void multiply(std::span<const float> x, std::span<float> sums, size_t firstBlock, size_t lastBlock) const;

    // The sign rule for every row r with active[r] != 0, using delta[r], for
    // the rows of blocks [firstBlock, lastBlock). `signs` holds sign(x) for
    // every input (see signs()); `delta` and `active` have paddedRows()
    // entries. Inactive rows are untouched.
    void learn(std::span<const float> signs, std::span<const float> delta, std::span<const std::uint8_t> active,
               float limit, size_t firstBlock, size_t lastBlock);



private:
    // The weights of block b: cols() inputs x `lanes` rows.
    std::span<const float> block(size_t b) const { return std::span(data_).subspan(b * cols_ * lanes, cols_ * lanes); }
    std::span<float> block(size_t b) { return std::span(data_).subspan(b * cols_ * lanes, cols_ * lanes); }

    size_t index(size_t row, size_t col) const
    {
        return ((row / lanes) * cols_ + col) * lanes + row % lanes;
    }

    size_t rows_ = 0;
    size_t cols_ = 0;
    aligned_vector<float> data_;  // blocks() * cols_ * lanes
};

} // namespace exr::kernels
