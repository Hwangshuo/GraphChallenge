
// kernel1 for performing Y * W and removing the empty rows
// during the fisrt several layers
// where nonzero elements are in a blocky pattern
//
// different mem channels for different Y rows
#include "gc.h"
#include <stdio.h>
#include <hls_vector.h>
#include <hls_stream.h>
#include <ap_int.h>

using namespace std;

void MM(
    float Y_buf[YT_row_tile][YT_col_tile],
    float input_W[WT_row_tile][YT_row_tile],
    float R_buf[WT_row_tile][YT_col_tile])
{
#pragma HLS INLINE off
SA_LOOP_1:
	for (int k = 0; k < YT_row_tile; k++)
	{
	SA_LOOP_2:
		for (int i = 0; i < WT_row_tile; i++)
		{
#pragma HLS UNROLL
		SA_LOOP_3:
			for (int j = 0; j < YT_col_tile; j++)
			{
#pragma HLS UNROLL
				float last = (k == 0) ? 0 : R_buf[i][j];

				float a_val = (i < WT_row_tile && k < YT_row_tile) ? input_W[i][k] : 0;
				float b_val = (k < YT_row_tile && j < YT_col_tile) ? Y_buf[k][j] : 0;
				float result = last + a_val * b_val;

				R_buf[i][j] = result;
			}
		}
	}


ReLU_LOOP_1:
    for (int i = 0; i < WT_row_tile; i++)
    {
    ReLU_LOOP_2:
        for (int j = 0; j < YT_col_tile; j++)
        {
            if(R_buf[i][j]>32)
            {
                R_buf[i][j]=32;

            }
            else if(R_buf[i][j]<0)
            {
                R_buf[i][j]=0;
            }
        }
    }
}

void WriteBack(
    const float R_buf[WT_row_tile][YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global[Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32> WT_row_start, ap_uint<32> YT_col_part)
{
#pragma HLS INLINE off
    ap_uint<32 * YT_col_tile> write_buf;

WriteBack_LOOP_1:
    for (int i = 0; i < WT_row_tile; i++)
    {
    WriteBack_LOOP_2:
        for (int j = 0; j < YT_col_tile; j++)
        {
            uint2float tmp = {.f = (float)R_buf[i][j]};
            write_buf.range(31 + j * 32, j * 32) = tmp.u;
            if (j == YT_col_tile - 1)
                YT_global[WT_row_start + i][YT_col_part] = write_buf;
        }
    }
}

void PrepareNextY(
    ap_uint<32 * Index_per_group> W_row_index[Neuron / 16 * Index_groups],
    ap_uint<32 * YT_col_tile> YT_global[Neuron][YT_col_channel / YT_col_tile],
    float Y_buf[YT_row_tile][YT_col_tile],
    ap_uint<32 > &W_row_index_start,
    ap_uint<32> YT_col_part)
{
#pragma HLS INLINE off
ap_uint<32 * YT_col_tile> read_buf;

int W_index_buf_index=W_row_index_start;
Y_Prepare_LOOP_1:
    for (int i = 0; i < Index_groups; i++)
    {
    Y_Prepare_LOOP_2:
        for (int j = 0; j < Index_per_group; j++)
        {
        Y_Prepare_LOOP_3:
            for (int p = 0; p < YT_col_tile; p++)
            {
                int k = i * Index_per_group + j;
                if (j == 0 && p == 0)
                    W_index_buf_index++;
                if (p == 0)
                    read_buf = YT_global[W_row_index[W_index_buf_index].range(31 + j * 32, j * 32)][YT_col_part];
                uint2float tmp = {.u = (unsigned int)read_buf.range(31 + p * 32, p * 32)};
                Y_buf[k][p] = tmp.f;
            }
        }
    }
}

void PE(ap_uint<32 * Index_per_group> W_row_index[Neuron / 16 * Index_groups],
        ap_uint<32 * YT_col_tile> Y_global[2][Neuron][YT_col_channel / YT_col_tile],int layer)
{
#pragma HLS DATAFLOW

    float W[WT_row_tile][YT_row_tile] = {
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w},
        {w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w, w}};
    float Y_buf[2][YT_row_tile][YT_col_tile];
    float R_buf[2][WT_row_tile][YT_col_tile];
#pragma HLS ARRAY_PARTITION variable=R_buf type=complete dim=1
#pragma HLS ARRAY_PARTITION variable=Y_buf type=complete dim=1
#pragma HLS ALLOCATION function instances=PrepareNextY limit=1
#pragma HLS ALLOCATION function instances=MM limit=1
#pragma HLS ALLOCATION function instances=WriteBack limit=1
    ap_uint<1> pp = 0;
    ap_uint<1> index = layer % 2;
    ap_uint<32> W_row_index_start = 0;
    ap_uint<32> R_write_start=0;
    for (int YT_col_part = 0; YT_col_part < Neuron/YT_col_tile; YT_col_part++)//! \u6bcf\u4e00\u6b21\u5faa\u73af\u4f1a\u8bfb\u53d6Y_T\u7684YT_col_part*16\u5217
    {

        for(int step=0;step<Neuron / 16 ;step++)
        {

            if (pp == 0)
            {
                PrepareNextY(W_row_index, Y_global[index], Y_buf[0], W_row_index_start, YT_col_part);//! \u9996\u5148\u8bfb\u53d6 Y  \u5757\u5927\u5c0f\u662f32 * 16 W_index\u7684\u8d77\u70b9\u662fW_index_buf_index
                MM(Y_buf[1], W, R_buf[1]);
                WriteBack(R_buf[0], Y_global[1 - index], R_write_start, YT_col_part);//! \u5c06R_buff\u5199\u56de\u5230Y\u4e2d \u56e0\u4e3a\u6bcf\u6b21W\u7684\u884c\u589e\u52a0
            }
            else
            {
                PrepareNextY(W_row_index, Y_global[index], Y_buf[1], W_row_index_start, YT_col_part);
                MM(Y_buf[0], W, R_buf[0]);
                WriteBack(R_buf[1], Y_global[1 - index],R_write_start, YT_col_part);
            }

            pp = 1 - pp;
            W_row_index_start=(step<1)?W_row_index_start:ap_uint<32>(W_row_index_start+Index_groups);
            R_write_start=(step<2)?R_write_start:ap_uint<32>(R_write_start+16);
                        
        }
    }
}

void SpdnnKernel(
    ap_uint<32 * Index_per_group> W_row_index[Layer][Neuron / 16 * Index_groups],
    ap_uint<32 * YT_col_tile> YT_global_0[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_1[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_2[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_3[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_4[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_5[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_6[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_7[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_8[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_9[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_10[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_11[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_12[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_13[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_14[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_15[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_16[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_17[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_18[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_19[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_20[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_21[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_22[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_23[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_24[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_25[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_26[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_27[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_28[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_29[2][Neuron][YT_col_channel / YT_col_tile],
    ap_uint<32 * YT_col_tile> YT_global_30[2][Neuron][YT_col_channel / YT_col_tile], ap_uint<32 * YT_col_tile> YT_global_31[2][Neuron][YT_col_channel / YT_col_tile])
{
#pragma HLS DATAFLOW

#pragma HLS INTERFACE m_axi port = W_row_index bundle = dmem0
#pragma HLS INTERFACE m_axi port = YT_global_0 bundle = hmem0
#pragma HLS INTERFACE m_axi port = YT_global_1 bundle = hmem1
#pragma HLS INTERFACE m_axi port = YT_global_2 bundle = hmem2
#pragma HLS INTERFACE m_axi port = YT_global_3 bundle = hmem3
#pragma HLS INTERFACE m_axi port = YT_global_4 bundle = hmem4
#pragma HLS INTERFACE m_axi port = YT_global_5 bundle = hmem5
#pragma HLS INTERFACE m_axi port = YT_global_6 bundle = hmem6
#pragma HLS INTERFACE m_axi port = YT_global_7 bundle = hmem7
#pragma HLS INTERFACE m_axi port = YT_global_8 bundle = hmem8
#pragma HLS INTERFACE m_axi port = YT_global_9 bundle = hmem9
#pragma HLS INTERFACE m_axi port = YT_global_10 bundle = hmem10
#pragma HLS INTERFACE m_axi port = YT_global_11 bundle = hmem11
#pragma HLS INTERFACE m_axi port = YT_global_12 bundle = hmem12
#pragma HLS INTERFACE m_axi port = YT_global_13 bundle = hmem13
#pragma HLS INTERFACE m_axi port = YT_global_14 bundle = hmem14
#pragma HLS INTERFACE m_axi port = YT_global_15 bundle = hmem15
#pragma HLS INTERFACE m_axi port = YT_global_16 bundle = hmem16
#pragma HLS INTERFACE m_axi port = YT_global_17 bundle = hmem17
#pragma HLS INTERFACE m_axi port = YT_global_18 bundle = hmem18
#pragma HLS INTERFACE m_axi port = YT_global_19 bundle = hmem19
#pragma HLS INTERFACE m_axi port = YT_global_20 bundle = hmem20
#pragma HLS INTERFACE m_axi port = YT_global_21 bundle = hmem21
#pragma HLS INTERFACE m_axi port = YT_global_22 bundle = hmem22
#pragma HLS INTERFACE m_axi port = YT_global_23 bundle = hmem23
#pragma HLS INTERFACE m_axi port = YT_global_24 bundle = hmem24
#pragma HLS INTERFACE m_axi port = YT_global_25 bundle = hmem25
#pragma HLS INTERFACE m_axi port = YT_global_26 bundle = hmem26
#pragma HLS INTERFACE m_axi port = YT_global_27 bundle = hmem27
#pragma HLS INTERFACE m_axi port = YT_global_28 bundle = hmem28
#pragma HLS INTERFACE m_axi port = YT_global_29 bundle = hmem29
#pragma HLS INTERFACE m_axi port = YT_global_30 bundle = hmem30
#pragma HLS INTERFACE m_axi port = YT_global_31 bundle = hmem31

    // ap_uint<32 * YT_col_tile>(*YT_globals[32])[Neuron][YT_col_channel / YT_col_tile] =
    //     {
    //         YT_globals[0] = YT_global_0,
    //         YT_globals[1] = YT_global_1,
    //         YT_globals[2] = YT_global_2,
    //         YT_globals[3] = YT_global_3,
    //         YT_globals[4] = YT_global_4,
    //         YT_globals[5] = YT_global_5,
    //         YT_globals[6] = YT_global_6,
    //         YT_globals[7] = YT_global_7,
    //         YT_globals[8] = YT_global_8,
    //         YT_globals[9] = YT_global_9,
    //         YT_globals[10] = YT_global_10,
    //         YT_globals[11] = YT_global_11,
    //         YT_globals[12] = YT_global_12,
    //         YT_globals[13] = YT_global_13,
    //         YT_globals[14] = YT_global_14,
    //         YT_globals[15] = YT_global_15,
    //         YT_globals[16] = YT_global_16,
    //         YT_globals[17] = YT_global_17,
    //         YT_globals[18] = YT_global_18,
    //         YT_globals[19] = YT_global_19,
    //         YT_globals[20] = YT_global_20,
    //         YT_globals[21] = YT_global_21,
    //         YT_globals[22] = YT_global_22,
    //         YT_globals[23] = YT_global_23,
    //         YT_globals[24] = YT_global_24,
    //         YT_globals[25] = YT_global_25,
    //         YT_globals[26] = YT_global_26,
    //         YT_globals[27] = YT_global_27,
    //         YT_globals[28] = YT_global_28,
    //         YT_globals[29] = YT_global_29,
    //         YT_globals[30] = YT_global_30,
    //         YT_globals[31] = YT_global_31,
    //     };

    hls::vector<int, Mem_channel> num_nnz_cols = YT_col_channel;


    for (int layer = 0; layer < 4; layer++)
    {
    PE_GEN_LOOP:
//         for (int pe = 0; pe < Mem_channel; pe++)
//         {
// #pragma HLS UNROLL
//             PE(W_row_index[layer], YT_globals[pe],layer);
//         }
            PE(W_row_index[layer], YT_global_0 ,layer);
            PE(W_row_index[layer], YT_global_1 ,layer);
            PE(W_row_index[layer], YT_global_2 ,layer);
            PE(W_row_index[layer], YT_global_3 ,layer);
            PE(W_row_index[layer], YT_global_4 ,layer);
            PE(W_row_index[layer], YT_global_5 ,layer);
            PE(W_row_index[layer], YT_global_6 ,layer);
            PE(W_row_index[layer], YT_global_7 ,layer);            
            PE(W_row_index[layer], YT_global_8 ,layer);
            PE(W_row_index[layer], YT_global_9 ,layer);
            PE(W_row_index[layer], YT_global_10,layer);
            PE(W_row_index[layer], YT_global_11,layer);
            PE(W_row_index[layer], YT_global_12,layer);
            PE(W_row_index[layer], YT_global_13,layer);
            PE(W_row_index[layer], YT_global_14,layer);
            PE(W_row_index[layer], YT_global_15,layer);
            PE(W_row_index[layer], YT_global_16,layer);
            PE(W_row_index[layer], YT_global_17,layer);
            PE(W_row_index[layer], YT_global_18,layer);
            PE(W_row_index[layer], YT_global_19,layer);
            PE(W_row_index[layer], YT_global_20,layer);
            PE(W_row_index[layer], YT_global_21,layer);
            PE(W_row_index[layer], YT_global_22,layer);
            PE(W_row_index[layer], YT_global_23,layer);
            PE(W_row_index[layer], YT_global_24,layer);
            PE(W_row_index[layer], YT_global_25,layer);
            PE(W_row_index[layer], YT_global_26,layer);
            PE(W_row_index[layer], YT_global_27,layer);
            PE(W_row_index[layer], YT_global_28,layer);
            PE(W_row_index[layer], YT_global_29,layer);
            PE(W_row_index[layer], YT_global_30,layer);
            PE(W_row_index[layer], YT_global_31,layer);
    }
}
