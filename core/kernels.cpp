#include "kernels.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace exr::kernels {

void signRule(std::span<float> w, std::span<const float> x, float delta, float limit)
{
    assert(x.size() >= w.size());
    // min/max instead of std::clamp: same result, but lets the compiler
    // vectorize the loop.
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = std::min(std::max(w[i] + sign(x[i]) * delta, -limit), limit);
}

void signs(std::span<const float> x, std::span<float> out)
{
    assert(out.size() >= x.size());
    for (size_t i = 0; i < x.size(); ++i)
        out[i] = sign(x[i]);
}

int threadsFor(size_t work)
{
#ifdef _OPENMP
    const size_t by_work = work / work_per_thread;
    return static_cast<int>(std::min<size_t>(by_work, static_cast<size_t>(omp_get_max_threads())));
#else
    (void)work;
    return 1;
#endif
}

weight_matrix::weight_matrix(size_t rows, size_t cols, std::span<const float> rowMajor)
    : rows_(rows), cols_(cols), data_(blocks() * cols * lanes, 0.0f)
{
    assert(rowMajor.size() == rows * cols);
    for (size_t r = 0; r < rows; ++r)
        setRow(r, rowMajor.subspan(r * cols, cols));
}

void weight_matrix::copyRow(size_t row, std::span<float> out) const
{
    assert(row < rows_ && out.size() == cols_);
    for (size_t c = 0; c < cols_; ++c)
        out[c] = data_[index(row, c)];
}

void weight_matrix::setRow(size_t row, std::span<const float> values)
{
    assert(row < rows_ && values.size() == cols_);
    for (size_t c = 0; c < cols_; ++c)
        data_[index(row, c)] = values[c];
}

void weight_matrix::setColumn(size_t col, std::span<const float> values)
{
    assert(col < cols_ && values.size() == rows_);
    // A column's lanes are contiguous within each block: copy them in runs.
    for (size_t b = 0; b < blocks(); ++b) {
        const size_t first = b * lanes, count = std::min(lanes, rows_ - first);
        std::ranges::copy(values.subspan(first, count), block(b).subspan(col * lanes).begin());
    }
}

void weight_matrix::appendColumns(size_t count, std::span<const float> rowMajor)
{
    assert(rowMajor.size() == rows_ * count);
    if (count == 0)
        return;
    const size_t cols = cols_ + count;
    aligned_vector<float> data(blocks() * cols * lanes, 0.0f);
    for (size_t b = 0; b < blocks(); ++b) {
        const std::span<float> grown = std::span(data).subspan(b * cols * lanes, cols * lanes);
        // The old inputs of a block are one contiguous run in both layouts.
        std::ranges::copy(block(b), grown.begin());
        for (size_t l = 0; l < lanes; ++l) {
            const size_t r = b * lanes + l;
            if (r >= rows_)
                break;
            for (size_t c = 0; c < count; ++c)
                grown[(cols_ + c) * lanes + l] = rowMajor[r * count + c];
        }
    }
    data_ = std::move(data);
    cols_ = cols;
}

void weight_matrix::reshape(size_t rows, size_t cols)
{
    rows_ = rows;
    cols_ = cols;
    data_.assign(blocks() * cols * lanes, 0.0f);
}

void weight_matrix::copyColumn(size_t col, std::span<float> out) const
{
    assert(col < cols_ && out.size() == rows_);
    for (size_t r = 0; r < rows_; ++r)
        out[r] = data_[index(r, col)];
}

void weight_matrix::zero()
{
    std::fill(data_.begin(), data_.end(), 0.0f);
}

void weight_matrix::removeRows(std::span<const size_t> rows)
{
    if (rows.empty())
        return;
    std::vector<float> kept;
    kept.reserve((rows_ - rows.size()) * cols_);
    std::vector<float> row(cols_);
    size_t next = 0;
    for (size_t r = 0; r < rows_; ++r) {
        if (next < rows.size() && rows[next] == r) {
            ++next;
            continue;
        }
        copyRow(r, row);
        kept.insert(kept.end(), row.begin(), row.end());
    }
    *this = weight_matrix(rows_ - rows.size(), cols_, kept);
}

void weight_matrix::removeColumns(std::span<const size_t> cols)
{
    if (cols.empty())
        return;
    const size_t kept_cols = cols_ - cols.size();
    std::vector<float> kept;
    kept.reserve(rows_ * kept_cols);
    for (size_t r = 0; r < rows_; ++r) {
        size_t next = 0;
        for (size_t c = 0; c < cols_; ++c) {
            if (next < cols.size() && cols[next] == c) {
                ++next;
                continue;
            }
            kept.push_back(data_[index(r, c)]);
        }
    }
    *this = weight_matrix(rows_, kept_cols, kept);
}

// The per-synapse kernels below loop over a block's lanes innermost: one
// input's lanes are contiguous, so the compiler vectorises them.
void weight_matrix::trace(std::span<const float> x, std::span<const float> post, float decay, size_t firstBlock,
                          size_t lastBlock)
{
    assert(x.size() >= cols_ && post.size() >= lastBlock * lanes);
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        const std::span<float> m = block(b);
        const std::span<const float> p = post.subspan(b * lanes, lanes);
        for (size_t c = 0; c < cols_; ++c) {
            const float xc = x[c];
            float* row = m.data() + c * lanes;
            for (size_t l = 0; l < lanes; ++l)
                row[l] = decay * row[l] + p[l] * xc;
        }
    }
}

void weight_matrix::eprop(weight_matrix& adapt, std::span<const float> x, std::span<const float> dyds,
                          std::span<const float> dydthr, std::span<const float> dthrdthr,
                          std::span<const float> dthrds, float decay, size_t firstBlock, size_t lastBlock)
{
    assert(adapt.rows_ == rows_ && adapt.cols_ == cols_ && x.size() >= cols_);
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        const std::span<float> m = block(b), a = adapt.block(b);
        const float* ys = dyds.data() + b * lanes;
        const float* yt = dydthr.data() + b * lanes;
        const float* tt = dthrdthr.data() + b * lanes;
        const float* ts = dthrds.data() + b * lanes;
        for (size_t c = 0; c < cols_; ++c) {
            const float xc = x[c];
            float* e = m.data() + c * lanes;
            float* ad = a.data() + c * lanes;
            for (size_t l = 0; l < lanes; ++l) {
                const float now = ys[l] * xc + yt[l] * ad[l];
                ad[l] = tt[l] * ad[l] + ts[l] * xc;
                e[l] = decay * e[l] + now;
            }
        }
    }
}

void weight_matrix::learnFrom(const weight_matrix& e, std::span<const float> delta, std::span<const float> keep,
                              std::span<const std::uint8_t> active, float limit, size_t firstBlock, size_t lastBlock)
{
    assert(e.rows_ == rows_ && e.cols_ == cols_);
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        float d[lanes], k[lanes];
        bool any = false;
        for (size_t l = 0; l < lanes; ++l) {
            const size_t r = b * lanes + l;
            const bool on = r < rows_ && active[r] != 0;
            d[l] = on ? delta[r] : 0.0f;
            k[l] = on ? keep[r] : 1.0f;
            any = any || on;
        }
        if (!any)
            continue;
        const std::span<float> w = block(b);
        const std::span<const float> m = e.block(b);
        for (size_t c = 0; c < cols_; ++c) {
            float* wr = w.data() + c * lanes;
            const float* mr = m.data() + c * lanes;
            for (size_t l = 0; l < lanes; ++l)
                wr[l] = std::min(std::max(wr[l] * k[l] + d[l] * mr[l], -limit), limit);
        }
    }
}

void weight_matrix::multiplyTransposed(std::span<const float> g, std::span<float> out) const
{
    assert(g.size() >= paddedRows() && out.size() >= cols_);
    for (size_t b = 0; b < blocks(); ++b) {
        const std::span<const float> w = block(b);
        const float* gb = g.data() + b * lanes;
        for (size_t c = 0; c < cols_; ++c) {
            const float* wr = w.data() + c * lanes;
            float s = 0.0f;
            for (size_t l = 0; l < lanes; ++l)
                s += wr[l] * gb[l];
            out[c] += s;
        }
    }
}

#if defined(__GNUC__)
// GCC / Clang vector extensions. A block's `lanes` rows are `per_block`
// native vectors: one 8-float vector with AVX, two 4-float vectors without.
namespace {
#if defined(__AVX__)
constexpr size_t width = 8;
#else
constexpr size_t width = 4;
#endif
constexpr size_t per_block = weight_matrix::lanes / width;
static_assert(weight_matrix::lanes % width == 0);
using vfloat = float __attribute__((vector_size(width * sizeof(float))));

// One vector from / to exactly `width` floats: the fixed-extent span is
// bounds-checked where it is made (subspan / first in debug builds).
inline vfloat load(std::span<const float, width> s)
{
    vfloat v;
    std::memcpy(&v, s.data(), sizeof(v));
    return v;
}
inline void store(std::span<float, width> s, vfloat v) { std::memcpy(s.data(), &v, sizeof(v)); }
inline vfloat splat(float x) { return vfloat{} + x; }

// The `width` floats of vector k of input c in a block.
template <typename Span>
auto lanesOf(Span block, size_t c, size_t k)
{
    return block.subspan(c * weight_matrix::lanes + k * width).template first<width>();
}
} // namespace

void weight_matrix::multiply(std::span<const float> x, std::span<float> sums, size_t firstBlock,
                             size_t lastBlock) const
{
    assert(x.size() >= cols_ && sums.size() >= lastBlock * lanes);
    // One accumulator per lane, each summing its row in input order: the same
    // additions, in the same order, as a scalar dot product. Four blocks at a
    // time give independent addition chains, hiding their latency.
    constexpr size_t group = 4;
    size_t b = firstBlock;
    for (; b + group <= lastBlock; b += group) {
        const std::span<const float> w[group] = {block(b), block(b + 1), block(b + 2), block(b + 3)};
        vfloat acc[group][per_block] = {};
        for (size_t c = 0; c < cols_; ++c) {
            const vfloat xc = splat(x[c]);
            for (size_t g = 0; g < group; ++g)
                for (size_t k = 0; k < per_block; ++k)
                    acc[g][k] += xc * load(lanesOf(w[g], c, k));
        }
        for (size_t g = 0; g < group; ++g)
            for (size_t k = 0; k < per_block; ++k)
                store(sums.subspan((b + g) * lanes + k * width).first<width>(), acc[g][k]);
    }
    for (; b < lastBlock; ++b) {
        const std::span<const float> w = block(b);
        vfloat acc[per_block] = {};
        for (size_t c = 0; c < cols_; ++c) {
            const vfloat xc = splat(x[c]);
            for (size_t k = 0; k < per_block; ++k)
                acc[k] += xc * load(lanesOf(w, c, k));
        }
        for (size_t k = 0; k < per_block; ++k)
            store(sums.subspan(b * lanes + k * width).first<width>(), acc[k]);
    }
}

void weight_matrix::learn(std::span<const float> signs, std::span<const float> delta,
                          std::span<const std::uint8_t> active, float limit, size_t firstBlock, size_t lastBlock)
{
    assert(signs.size() >= cols_ && delta.size() >= lastBlock * lanes && active.size() >= lastBlock * lanes);
    const vfloat hi = splat(limit), lo = splat(-limit);
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        vfloat on[per_block] = {}, d[per_block] = {};
        bool any = false;
        for (size_t k = 0; k < per_block; ++k)
            for (size_t l = 0; l < width; ++l) {
                const size_t r = b * lanes + k * width + l;
                const bool lane_on = active[r] != 0;
                on[k][l] = lane_on ? 1.0f : 0.0f;
                d[k][l] = lane_on ? delta[r] : 0.0f;
                any = any || lane_on;
            }
        if (!any)
            continue;
        const std::span<float> w = block(b);
        for (size_t c = 0; c < cols_; ++c) {
            const vfloat s = splat(signs[c]);
            for (size_t k = 0; k < per_block; ++k) {
                const auto p = lanesOf(w, c, k);
                const vfloat old = load(p);
                vfloat v = old + s * d[k];
                v = v < lo ? lo : v;  // std::max(v, lo), bit for bit (NaN stays NaN)
                v = hi < v ? hi : v;  // std::min(v, hi)
                store(p, on[k] != 0.0f ? v : old);
            }
        }
    }
}

void weight_matrix::learnScaled(std::span<const float> pre, std::span<const float> delta, std::span<const float> keep,
                                std::span<const std::uint8_t> active, float limit, size_t firstBlock, size_t lastBlock)
{
    assert(pre.size() >= cols_ && delta.size() >= lastBlock * lanes && keep.size() >= lastBlock * lanes &&
           active.size() >= lastBlock * lanes);
    const vfloat hi = splat(limit), lo = splat(-limit);
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        vfloat on[per_block] = {}, d[per_block] = {}, kp[per_block] = {};
        bool any = false;
        for (size_t k = 0; k < per_block; ++k)
            for (size_t l = 0; l < width; ++l) {
                const size_t r = b * lanes + k * width + l;
                const bool lane_on = active[r] != 0;
                on[k][l] = lane_on ? 1.0f : 0.0f;
                d[k][l] = lane_on ? delta[r] : 0.0f;
                kp[k][l] = lane_on ? keep[r] : 1.0f;
                any = any || lane_on;
            }
        if (!any)
            continue;
        const std::span<float> w = block(b);
        for (size_t c = 0; c < cols_; ++c) {
            const vfloat s = splat(pre[c]);
            for (size_t k = 0; k < per_block; ++k) {
                const auto p = lanesOf(w, c, k);
                const vfloat old = load(p);
                vfloat v = old * kp[k] + s * d[k];
                v = v < lo ? lo : v;
                v = hi < v ? hi : v;
                store(p, on[k] != 0.0f ? v : old);
            }
        }
    }
}

#else
// Portable fallback: the same arithmetic, one lane at a time.
void weight_matrix::multiply(std::span<const float> x, std::span<float> sums, size_t firstBlock,
                             size_t lastBlock) const
{
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        const std::span<const float> w = block(b);
        float acc[lanes] = {};
        for (size_t c = 0; c < cols_; ++c)
            for (size_t l = 0; l < lanes; ++l)
                acc[l] += x[c] * w[c * lanes + l];
        for (size_t l = 0; l < lanes; ++l)
            sums[b * lanes + l] = acc[l];
    }
}

void weight_matrix::learn(std::span<const float> signs, std::span<const float> delta,
                          std::span<const std::uint8_t> active, float limit, size_t firstBlock, size_t lastBlock)
{
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        const std::span<float> w = block(b);
        for (size_t l = 0; l < lanes; ++l) {
            if (active[b * lanes + l] == 0)
                continue;
            const float d = delta[b * lanes + l];
            for (size_t c = 0; c < cols_; ++c)
                w[c * lanes + l] = std::min(std::max(w[c * lanes + l] + signs[c] * d, -limit), limit);
        }
    }
}
void weight_matrix::learnScaled(std::span<const float> pre, std::span<const float> delta, std::span<const float> keep,
                                std::span<const std::uint8_t> active, float limit, size_t firstBlock, size_t lastBlock)
{
    for (size_t b = firstBlock; b < lastBlock; ++b) {
        const std::span<float> w = block(b);
        for (size_t l = 0; l < lanes; ++l) {
            if (active[b * lanes + l] == 0)
                continue;
            const float d = delta[b * lanes + l], k = keep[b * lanes + l];
            for (size_t c = 0; c < cols_; ++c)
                w[c * lanes + l] = std::min(std::max(w[c * lanes + l] * k + pre[c] * d, -limit), limit);
        }
    }
}
#endif

} // namespace exr::kernels
