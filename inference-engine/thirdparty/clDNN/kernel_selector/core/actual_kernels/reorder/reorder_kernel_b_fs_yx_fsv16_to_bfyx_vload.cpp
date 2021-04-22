// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "reorder_kernel_b_fs_yx_fsv16_to_bfyx_vload.h"
#include "kernel_selector_utils.h"
#include "common_tools.h"
#include <string>
#include <functional>
#include <cmath>

// Tile size : 4x4 or 8x8
#define MIN_TILE_SIZE 4
#define DEFAULT_TILE_SIZE 8

namespace kernel_selector {
ParamsKey ReorderKernel_b_fs_yx_fsv16_to_bfyx_vload::GetSupportedKey() const {
    ParamsKey k;
    k.EnableInputDataType(Datatype::F16);
    k.EnableInputDataType(Datatype::F32);
    k.EnableInputDataType(Datatype::UINT8);
    k.EnableInputDataType(Datatype::UINT16);
    k.EnableInputDataType(Datatype::UINT32);
    k.EnableInputDataType(Datatype::INT8);
    k.EnableInputDataType(Datatype::INT16);
    k.EnableInputDataType(Datatype::INT32);
    k.EnableInputDataType(Datatype::INT64);

    k.EnableOutputDataType(Datatype::F16);
    k.EnableOutputDataType(Datatype::F32);
    k.EnableOutputDataType(Datatype::UINT8);
    k.EnableOutputDataType(Datatype::UINT16);
    k.EnableOutputDataType(Datatype::UINT32);
    k.EnableOutputDataType(Datatype::INT8);
    k.EnableOutputDataType(Datatype::INT16);
    k.EnableOutputDataType(Datatype::INT32);
    k.EnableOutputDataType(Datatype::INT64);

    k.EnableInputLayout(DataLayout::b_fs_yx_fsv16);
    k.EnableInputLayout(DataLayout::b_fs_zyx_fsv16);

    k.EnableOutputLayout(DataLayout::bfyx);
    k.EnableOutputLayout(DataLayout::bfzyx);
    k.EnableOutputLayout(DataLayout::bfwzyx);

    k.EnableBatching();
    k.EnableTensorOffset();
    k.EnableTensorPitches();

    return k;
}

static inline std::string GetTiledOutputOrder(size_t size) {
    std::string order_str = "";
    switch (size) {
    case 4:
        order_str = "b, f + lh, y, x";
        break;
    case 5:
        order_str = "b, f + lh, z, y, x";
        break;
    default: throw std::runtime_error("Unsupported combination\n");
    }
    return order_str;
}

static inline size_t GetFsvAlignment(const reorder_params& params) {
    const auto& in = params.inputs[0];
    int fsv_alignment = -1;
    switch (in.GetLayout()) {
    case DataLayout::b_fs_yx_fsv16:
    case DataLayout::b_fs_zyx_fsv16:
        fsv_alignment = 16;
        break;
    default:
        throw std::runtime_error("Unsupported combination\n");
    }
    return fsv_alignment;
}

static inline size_t GetTileSize(const reorder_params& /*params*/) {
    //const Datatype input_type = params.inputs[0].GetDType();
    //const Datatype output_type = params.output.GetDType();

    // i64 supports tile size 4
    //if ((input_type == Datatype::INT64) || (output_type == Datatype::INT64)) {
    //    return MIN_TILE_SIZE;
    //}

    return DEFAULT_TILE_SIZE;
}

static inline std::vector<size_t> GetGWS(const reorder_params& params) {
    const auto& in = params.inputs[0];
    const size_t tile_size = GetTileSize(params);
    const size_t fsv_alignment = GetFsvAlignment(params);
    std::vector<size_t> gws;

    switch (in.GetLayout()) {
    case DataLayout::b_fs_yx_fsv16:
        gws = { CeilDiv(fsv_alignment, tile_size),
            CeilDiv(in.X().v, tile_size) * in.Y().v,
            in.Batch().v * CeilDiv(in.Feature().v, fsv_alignment) };
        break;
    case DataLayout::b_fs_zyx_fsv16:
        gws = { CeilDiv(fsv_alignment, tile_size),
            CeilDiv(in.X().v, tile_size) * in.Y().v * in.Z().v,
            in.Batch().v * CeilDiv(in.Feature().v, fsv_alignment) };
        break;
    default:
        throw std::runtime_error("Unsupported combination\n");
    }
    return gws;
}

static std::vector<size_t> GetBestLwsFromGws(const reorder_params& params, const std::vector<size_t>& gws, const size_t tile_width, const size_t tile_size) {
    std::vector<size_t> lws{ 1, 1, 1 };
    std::vector<size_t> dims{ 0, 1, 2 };

    // SLM size: elemsize * tile_width * tile_width * work_items <= 64K
    const size_t elem_size = params.inputs[0].ElementSize();
    const size_t max_local_mem_size = params.engineInfo.maxLocalMemSize;
    const size_t max_work_group_size = params.engineInfo.maxWorkGroupSize;
    size_t max_num_work_items = std::min(max_work_group_size, max_local_mem_size / (elem_size * tile_width * tile_size));

    for (size_t i = 0; i < dims.size(); ++i) {
        size_t dim = dims[i];
        size_t max_divider = static_cast<size_t>(std::sqrt(gws[dim]) + 1);
        for (size_t divider = 1; divider <= max_divider; ++divider) {
            if (gws[dim] % divider == 0) {
                const size_t lws0 = gws[dim] / divider;
                if (lws0 <= max_num_work_items) {
                    lws[dim] = std::max(lws[dim], lws0);
                }
                if (divider <= max_num_work_items) {
                    lws[dim] = std::max(lws[dim], divider);
                }
            }
        }
        max_num_work_items /= lws[dim];
    }
    return lws;
}

CommonDispatchData ReorderKernel_b_fs_yx_fsv16_to_bfyx_vload::SetDefault(const reorder_params& params) const {
    CommonDispatchData dispatchData;
    const size_t tile_size = GetTileSize(params);
    dispatchData.gws = GetGWS(params);
    dispatchData.lws = GetBestLwsFromGws(params, dispatchData.gws, tile_size, tile_size);
    return dispatchData;
}

JitConstants ReorderKernel_b_fs_yx_fsv16_to_bfyx_vload::GetJitConstants(const reorder_params& params) const {
    auto jit = ReorderKernelBase::GetJitConstants(params);

    //const size_t b = params.inputs[0].Batch().v;
    const size_t f = params.inputs[0].Feature().v;
    const size_t x = params.inputs[0].X().v;
    const size_t tile_size = GetTileSize(params);
    const size_t output_ndims = params.output.GetDims().size();
    const size_t fsv_alignment = GetFsvAlignment(params);

    const auto gws = GetGWS(params);
    const auto lws = GetBestLwsFromGws(params, gws, tile_size, tile_size);
    const uint64_t total_lws = lws[0] * lws[1] * lws[2];



    jit.AddConstant(MakeJitConstant("OUTPUT_TILED_ORDER", GetTiledOutputOrder(output_ndims)));
    jit.AddConstant(MakeJitConstant("INPUT0_FEATURE_SLICE_NUM", CeilDiv(f, fsv_alignment)));
    jit.AddConstant(MakeJitConstant("TILE_SIZE", tile_size));
    jit.AddConstant(MakeJitConstant("FSV_ALIGNMENT", fsv_alignment));
    jit.AddConstant(MakeJitConstant("TRANS_BUF_SIZE", tile_size * total_lws));

    // whether x is tile_size-aligned
    if (x % tile_size != 0) {
        jit.AddConstant(MakeJitConstant("X_REMAINDER_SIZE", x % tile_size));
        jit.AddConstant(MakeJitConstant("X_REMAINDER_CONDITION", "(x >= (INPUT0_SIZE_X - X_REMAINDER_SIZE)) && (x < INPUT0_SIZE_X)"));
        //jit.AddConstant(MakeJitConstant("X_NO_REMAINDER_CONDITION", "(x < (INPUT0_SIZE_X - X_REMAINDER_SIZE))"));
    }

    return jit;
}

KernelsData ReorderKernel_b_fs_yx_fsv16_to_bfyx_vload::GetKernelsData(const Params& params, const optional_params& options) const {
    assert(params.GetType() == KernelType::REORDER);

    const reorder_params& orgParams = static_cast<const reorder_params&>(params);

    return GetCommonKernelsData(orgParams, options);
}

bool ReorderKernel_b_fs_yx_fsv16_to_bfyx_vload::Validate(const Params& p, const optional_params& o) const {
    if (!ReorderKernelBase::Validate(p, o)) {
        return false;
    }

    // kelvin to do
    // f should be multiple of 16

    return true;
}

KernelsPriority ReorderKernel_b_fs_yx_fsv16_to_bfyx_vload::GetKernelsPriority(const Params& /*params*/, const optional_params& /*options*/) const {
    return FORCE_PRIORITY_5;
}
}  // namespace kernel_selector
