// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//#include "include/fetch.cl"
//#include "include/common.cl"
//#include "include/data_types.cl"
#include "include/include_all.cl"

#define unroll_for __attribute__((opencl_unroll_hint)) for
#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define INPUT0_GET_TILED_INDEX(ORDER) INPUT0_GET_INDEX(ORDER)
#define OUTPUT_GET_TILED_INDEX(ORDER) OUTPUT_GET_INDEX(ORDER)

#define INPUTVTYPE CAT(INPUT0_TYPE, TILE_SIZE)
#define OUTPUTVTYPE CAT(OUTPUT_TYPE, TILE_SIZE)
#define VLOAD CAT(vload, TILE_SIZE)
#define VSTORE CAT(vstore, TILE_SIZE)
#define AS_INPUTVTYPE CAT(as_, INPUTVTYPE)

#define GET_GLOBAL_ID(IDX) ((uint)get_global_id(IDX))
#define GET_LOCAL_ID(IDX) ((uint)get_local_id(IDX))
#define GET_LOCAL_SIZE(IDX) ((uint)get_local_size(IDX))

KERNEL (reorder_data_b_fs_yx_fsv16_to_bfyx_vload)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
    )
{
#if INPUT0_DIMS == 4
    const uint y = GET_GLOBAL_ID(1) % INPUT0_SIZE_Y;
    const uint x = (GET_GLOBAL_ID(1) / INPUT0_SIZE_Y) * TILE_SIZE;
#elif INPUT0_DIMS == 5
    const uint z = GET_GLOBAL_ID(1) % INPUT0_SIZE_Z;
    const uint yx = GET_GLOBAL_ID(1) / INPUT0_SIZE_Z;
    const uint y = yx % INPUT0_SIZE_Y;
    const uint x = (yx / INPUT0_SIZE_Y) * TILE_SIZE;
#else
#error reorder_data_b_fs_yx_fsv16_to_bfyx_vload.cl: input format - not supported
#endif

    const uint fsv = GET_GLOBAL_ID(0) * TILE_SIZE;
    const uint fs = GET_GLOBAL_ID(2) % INPUT0_FEATURE_SLICE_NUM;
    const uint b = GET_GLOBAL_ID(2) / INPUT0_FEATURE_SLICE_NUM;
    const uint f = fsv + fs * FSV_ALIGNMENT;

    const uint x_pitch = FSV_ALIGNMENT;
    const uint y_pitch = x_pitch * (OUTPUT_SIZE_X);

#if INPUT0_DIMS == 4
    const uint fs_pitch = y_pitch * (OUTPUT_SIZE_Y);
    const uint b_pitch = fs_pitch * (INPUT0_FEATURE_SLICE_NUM);
    const uint input_idx_tile = (b * b_pitch) + (fs * fs_pitch) + (y * y_pitch) + (x * x_pitch) + (fsv);
#elif INPUT0_DIMS == 5
    const uint z_pitch = y_pitch * (OUTPUT_SIZE_Y);
    const uint fs_pitch = z_pitch * (OUTPUT_SIZE_Z);
    const uint b_pitch = fs_pitch * (INPUT0_FEATURE_SLICE_NUM);
    const uint input_idx_tile = (b * b_pitch) + (fs * fs_pitch) + (z * z_pitch) + (y * y_pitch) + (x * x_pitch) + (fsv);
#endif

    // get local buf offset
    __local OUTPUTVTYPE transpose_buf[TRANS_BUF_SIZE];
    const uint local_id = GET_LOCAL_ID(0) * GET_LOCAL_SIZE(2) * GET_LOCAL_SIZE(1)
                    + GET_LOCAL_ID(1) * GET_LOCAL_SIZE(2)
                    + GET_LOCAL_ID(2);
    const uint local_buf_offset = local_id * TILE_SIZE;

    // read and transpose
    #ifdef X_REMAINDER_SIZE
        if (X_REMAINDER_CONDITION) {
            unroll_for (uint lw = 0; lw < X_REMAINDER_SIZE; ++lw) {
                const uint input_idx = input_idx_tile + (lw * x_pitch); //INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
                INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
                unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
                    const uint dst = local_buf_offset + lh;
                    transpose_buf[dst][lw] = ACTIVATION(read_data[lh], ACTIVATION_PARAMS);
                }
            }
        } else {
            unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
                const uint input_idx = input_idx_tile + (lw * x_pitch); //INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
                INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
                unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
                    const uint dst = local_buf_offset + lh;
                    transpose_buf[dst][lw] = ACTIVATION(read_data[lh], ACTIVATION_PARAMS);
                }
            }
        }
    #else
        unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
            const uint input_idx = input_idx_tile + (lw * x_pitch); //INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
                const uint dst = local_buf_offset + lh;
                transpose_buf[dst][lw] = ACTIVATION(read_data[lh], ACTIVATION_PARAMS);
            }
        }
    #endif

    // write to ddr
    #ifdef X_REMAINDER_SIZE
        if (X_REMAINDER_CONDITION) {
            unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
                unroll_for (uint i = 0; i < X_REMAINDER_SIZE; ++i) {
                    output[output_idx + i] = transpose_buf[local_buf_offset + lh][i];
                }
            }
        } else {
            unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
                const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);    //output_idx_tile + (lw * x_pitch);
                VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
            }
        }
    #else
        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);    //output_idx_tile + (lw * x_pitch);
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }
    #endif


}

#undef FUNC_WRITE
#undef FUNC_VSTORE
#undef FUNC_VLOAD

#undef GET_LOCAL_SIZE
#undef GET_LOCAL_ID
#undef GET_GLOBAL_ID

#undef AS_INPUTVTYPE
#undef VSTORE
#undef VLOAD
#undef OUTPUTVTYPE
#undef INPUTVTYPE

#undef OUTPUT_GET_TILED_INDEX
#undef INPUT0_GET_TILED_INDEX
#undef CEIL_DIV
#undef unroll_for
