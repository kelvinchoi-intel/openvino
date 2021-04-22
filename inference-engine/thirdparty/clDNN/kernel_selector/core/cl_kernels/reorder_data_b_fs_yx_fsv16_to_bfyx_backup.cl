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

KERNEL (reorder_data_b_fs_yx_fsv16_to_bfyx)(
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
#error reorder_data_b_fs_yx_fsv16_to_bfyx.cl: input format - not supported
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

    #else
        /* read
        unroll_for (uint lw = 0; lw < TILE_SIZE; ++lw) {
            const uint input_idx = input_idx_tile + (lw * x_pitch); //INPUT0_GET_TILED_INDEX(INPUT0_TILED_ORDER);
            INPUTVTYPE read_data = AS_INPUTVTYPE(VLOAD(0, input + input_idx));
            unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
                const uint dst = local_buf_offset + lh;
                transpose_buf[dst][lw] = ACTIVATION(read_data[lh], ACTIVATION_PARAMS);
            }
        }*/

        /* write
        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);    //output_idx_tile + (lw * x_pitch);
            VSTORE(transpose_buf[local_buf_offset + lh], 0, output + output_idx);
        }*/

        uint sub_group_id = get_sub_group_id();
        uint sub_group_local_id = get_sub_group_local_id();

        if(input_idx_tile == 128) {
            INPUTVTYPE read_data = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)input + 126));

            printf("[%d] %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f\n",
                126,
                read_data[0],
                read_data[1],
                read_data[2],
                read_data[3],
                read_data[4],
                read_data[5],
                read_data[6],
                read_data[7]);

            read_data = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)input + 128));

            printf("[%d] sub_group_id:%d, sub_group_local_id:%d, local_id:%d %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f %0.1f\n",
                128,
                sub_group_id, sub_group_local_id,
                local_id,
                read_data[0],
                read_data[1],
                read_data[2],
                read_data[3],
                read_data[4],
                read_data[5],
                read_data[6],
                read_data[7]);
        }

        unroll_for (uint lh = 0; lh < TILE_SIZE; ++lh) {
            // read
            const uint input_idx = input_idx_tile + lh;
            //INPUTVTYPE read_data = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)(input) + input_idx));
            INPUTVTYPE read_data = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)input + (input_idx - sub_group_local_id)));
            //INPUTVTYPE read_data = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)input + (126)));

            INPUTVTYPE read_vload = AS_INPUTVTYPE(VLOAD(0, input + input_idx));

            // write
            const uint output_idx = OUTPUT_GET_TILED_INDEX(OUTPUT_TILED_ORDER);
            VSTORE((read_data), 0, output + output_idx);

            if(input_idx_tile == 128 && output_idx == 8) {
                //const int sub_group_local_id_ = sub_group_local_id; //2;
                const int ddd = (128 - get_sub_group_local_id());

                printf("ddd:%d \n", ddd);

                //INPUTVTYPE read_data1 = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)(input) + (ddd)));
                //INPUTVTYPE read_data2 = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)(input) + (126)));

                INPUTVTYPE read_data1 = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)(&input[ddd])));
                INPUTVTYPE read_data2 = AS_INPUTVTYPE(intel_sub_group_block_read8((const __global uint*)(&input[126])));

                printf("input[ddd]:%d, input[126]:%d \n", input[ddd], input[126]);

                printf("input_idx=%d, output_idx=%d, offset=%d, read_data1[0]=%f, read_data2[0]=%f, input_val:%f, output_val:%f, lh:%d, b:%d f:%d y:%d x:%d \n",
                        input_idx, output_idx,
                        (ddd),
                        read_data1[0], read_data2[0],
                        input[input_idx], output[output_idx],
                        lh, b, f, y, x);
            }
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
