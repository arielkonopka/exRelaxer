// Splitting independent work items between OpenMP threads.
#pragma once
#include <algorithm>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace exr {

// Runs job(first, last) over [0, count), in parallel when `threads` >= 2:
// each thread gets one contiguous run of items, the same run for the same
// count and thread count. Callers use it only for items whose results do not
// depend on how they are split, so results never depend on the thread count.
// Contiguous runs also keep a thread on the same data from one pass to the
// next (e.g. forward then learning), which is still in its cache.
template <typename Job>
void parallelChunks(size_t count, int threads, Job job)
{
    if (threads < 2 || count < 2) {
        job(size_t{0}, count);
        return;
    }
#ifdef _OPENMP
    // A plain branch rather than OpenMP's if() clause: that clause still sets
    // up a parallel region on every call, which made small layers several
    // times slower.
    threads = static_cast<int>(std::min<size_t>(static_cast<size_t>(threads), count));
#pragma omp parallel num_threads(threads)
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        const size_t n = static_cast<size_t>(omp_get_num_threads());
        job(count * t / n, count * (t + 1) / n);
    }
#else
    job(size_t{0}, count);
#endif
}

} // namespace exr
