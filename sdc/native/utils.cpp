// *****************************************************************************
// Copyright (c) 2020, Intel Corporation All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     Redistributions of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//     Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
// OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// *****************************************************************************

#include "utils.hpp"

namespace utils
{

tbb::task_arena& get_arena()
{
    static tbb::task_arena arena;

    return arena;
}

void set_threads_num(uint64_t threads)
{
    auto& arena = get_arena();
    arena.terminate();
    arena.initialize(threads);
}

void parallel_copy(void* src, void* dst, uint64_t len, uint64_t size)
{
    using range_t = tbb::blocked_range<uint64_t>;
    tbb::parallel_for(range_t(0,len), [src, dst, size](const range_t& range)
    {
        auto r_src = reinterpret_cast<quant*>(src) + range.begin()*size;
        auto r_dst = reinterpret_cast<quant*>(dst) + range.begin()*size;
        std::copy_n(r_src, range.size()*size, r_dst);
    });
}

template<>
bool nanless<float>(const float& left, const float& right)
{
    return std::less<float>()(left, right) || isnan(right);
}

template<>
bool nanless<double>(const double& left, const double& right)
{
    return std::less<double>()(left, right) || isnan(right);
}

template<>
bool stable_nanless<float>(const float& left, const float& right)
{
    return std::less<float>()(left, right) || (isnan(right) && !isnan(left));
}

template<>
bool stable_nanless<double>(const double& left, const double& right)
{
    return std::less<double>()(left, right) || (isnan(right) && !isnan(left));
}

}
