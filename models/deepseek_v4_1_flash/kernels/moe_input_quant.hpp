// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Group-32 MXFP8 activation quantization and MX_A_ZZ scale packing.
#include <cstdint>
#include <pto/pto-inst.hpp>

#pragma once
namespace moe_input_quant {
using namespace pto;
constexpr int Rows = 16;
constexpr int Columns = 256;
template <typename T, int R, int C>
using Vec = Tile<TileType::Vec, T, R, C>;
using Input = Vec<bfloat16_t, Rows, Columns>;
using Payload = Vec<uint8_t, Rows, Columns>;
using Scales = Vec<uint8_t, 1, Rows * Columns / 32>;

#if defined(__DAV_VEC__) && defined(PTO_NPU_ARCH_A5)
using DirectScales = Vec<uint8_t, 1, Rows * 1024 / 32>;
template <bool Direct, typename ScaleTile>
__tf__ __aicore__ inline void quantize(
    Payload::TileDType __out__ payload, typename ScaleTile::TileDType __out__ scales,
    Input::TileDType __in__ input, uint32_t part) {
    auto src = (__ubuf__ bfloat16_t*)__cce_get_tile_ptr(input);
    auto dst = (__ubuf__ uint8_t*)__cce_get_tile_ptr(payload);
    auto packed_scales = (__ubuf__ uint16_t*)__cce_get_tile_ptr(scales);
    __VEC_SCOPE__ {
        MaskReg all = pset_b32(PAT_ALL), all8 = pset_b8(PAT_ALL);
        uint32_t count64 = 64, count1 = 1;
        MaskReg bytes64 = CreatePredicate<uint8_t>(count64);
        MaskReg half1 = CreatePredicate<uint16_t>(count1);
        RegTensor<uint32_t> lane, bits, ix1, ix2, ix4, ix8, ix16, lo, hi;
        vci((RegTensor<int32_t>&)lane, 0, INC_ORDER);
        vdup(bits, 1u, all, MODE_ZEROING); vxor(ix1, lane, bits, all);
        vdup(bits, 2u, all, MODE_ZEROING); vxor(ix2, lane, bits, all);
        vdup(bits, 4u, all, MODE_ZEROING); vxor(ix4, lane, bits, all);
        vdup(bits, 8u, all, MODE_ZEROING); vxor(ix8, lane, bits, all);
        vdup(bits, 16u, all, MODE_ZEROING); vxor(ix16, lane, bits, all);
        vdup(lo, 0u, all, MODE_ZEROING); vdup(hi, 32u, all, MODE_ZEROING);
        RegTensor<uint8_t> byte_index;
        vci((RegTensor<int8_t>&)byte_index, static_cast<int8_t>(0), INC_ORDER);
        vmuls((RegTensor<int16_t>&)byte_index, (RegTensor<int16_t>&)byte_index,
              static_cast<int16_t>(4), all8);
        RegTensor<float> floor, lower, upper;
        vdup(floor, 1e-4f, all, MODE_ZEROING);
        vdup(lower, -448.0f, all, MODE_ZEROING);
        vdup(upper, 448.0f, all, MODE_ZEROING);
        for (uint16_t row = 0; row < Rows; ++row) {
            for (uint16_t segment = 0; segment < Columns / 64; ++segment) {
                RegTensor<float> x, maximum, shuffled, divisor;
                RegTensor<int32_t> exponent;
                RegTensor<bfloat16_t> raw;
                vlds(raw, src, row * Columns + segment * 64, UNPK_B16);
                vcvt(x, raw, all, PART_EVEN);
                vabs(maximum, x, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix16);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix8);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix4);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix2);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vselr((RegTensor<uint32_t>&)shuffled, (RegTensor<uint32_t>&)maximum, ix1);
                vmax(maximum, maximum, shuffled, all, MODE_ZEROING);
                vmax(maximum, maximum, floor, all, MODE_ZEROING);
                vmuls(maximum, maximum, 1.0f / 448.0f, all, MODE_ZEROING);
                vadds(exponent, (RegTensor<int32_t>&)maximum, 8388607, all, MODE_ZEROING);
                vshrs(exponent, exponent, 23, all, MODE_ZEROING);
                vshls((RegTensor<int32_t>&)divisor, exponent, 23, all, MODE_ZEROING);
                vdiv(x, x, divisor, all, MODE_ZEROING);
                vmax(x, x, lower, all, MODE_ZEROING); vmin(x, x, upper, all, MODE_ZEROING);
                vector_f8e4m3 converted;
                RegTensor<uint8_t> packed;
                vcvt(converted, x, all, ROUND_R, RS_DISABLE, PART_P0);
                vselr(packed, (RegTensor<uint8_t>&)converted, byte_index);
                vsts(packed, dst, row * Columns + segment * 64, NORM_B8, bytes64);
                RegTensor<uint32_t> e0, e1;
                vselr(e0, (RegTensor<uint32_t>&)exponent, lo);
                vselr(e1, (RegTensor<uint32_t>&)exponent, hi);
                vshls(e1, e1, 8u, all, MODE_ZEROING); vor(e0, e0, e1, all);
                vsts((RegTensor<uint16_t>&)e0, packed_scales, Direct ? row * 16 + part * 4 + segment : segment * Rows + row, ONEPT_B16, half1);
            }
        }
    }
}

#endif
}
