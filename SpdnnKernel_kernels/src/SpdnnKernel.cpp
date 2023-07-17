
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
	const ap_uint<32> step,
	const ap_uint<32> YT_col_part)
{
#pragma HLS INLINE off
    ap_uint<32 * YT_col_tile> write_buf;
    ap_uint<32> R_write_start=0;
    R_write_start=(step<2)?ap_uint<32>(0):ap_uint<32>(step<<4);

WriteBack_LOOP_1:
    for (int i = 0; i < WT_row_tile; i++)
    {
    WriteBack_LOOP_2:
        for (int j = 0; j < YT_col_tile; j++)
        {
            uint2float tmp = {.f = (float)R_buf[i][j]};
            write_buf.range(31 + j * 32, j * 32) = tmp.u;
            if (j == YT_col_tile - 1)
                YT_global[R_write_start + i][YT_col_part] = write_buf;
        }
    }
}

void PrepareNextY(
    ap_uint<32 * Index_per_group> W_row_index[Neuron / 16 * Index_groups],
    ap_uint<32 * YT_col_tile> YT_global[Neuron][YT_col_channel / YT_col_tile],
    float Y_buf[YT_row_tile][YT_col_tile],
    const ap_uint<32 > step,
	const ap_uint<32> YT_col_part)
{
#pragma HLS INLINE off
ap_uint<32 * YT_col_tile> read_buf;

ap_uint<32> W_index_buf_index;
W_index_buf_index=(step<1)?ap_uint<32>(0):ap_uint<32>(step<<2);
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
        ap_uint<32 * YT_col_tile> Y_global_r[Neuron][YT_col_channel / YT_col_tile],ap_uint<32 * YT_col_tile> Y_global_w[Neuron][YT_col_channel / YT_col_tile],const int layer)
{

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
     float Y_buf0[YT_row_tile][YT_col_tile];
     float Y_buf1[YT_row_tile][YT_col_tile];
     float R_buf0[WT_row_tile][YT_col_tile];
     float R_buf1[YT_row_tile][YT_col_tile];
//     float Y_buf[YT_row_tile][YT_col_tile];
//     float R_buf[WT_row_tile][YT_col_tile];
// #pragma HLS ARRAY_PARTITION variable=R_buf type=complete dim=1
// #pragma HLS ARRAY_PARTITION variable=Y_buf type=complete dim=1
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=W
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=R_buf0
#pragma HLS ARRAY_PARTITION dim=0 type=complete variable=R_buf1
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=Y_buf0
#pragma HLS ARRAY_PARTITION dim=2 type=complete variable=Y_buf1


     ap_uint<1> pp = 0;
     ap_uint<32> W_row_index_start = 0;



     for (ap_uint<32> YT_col_part = 0; YT_col_part < Neuron/YT_col_tile; YT_col_part++)
     {

         for(ap_uint<32> step=0;step<Neuron / YT_col_tile ;step++)
         {
#pragma HLS PIPELINE II=1
#pragma HLS ALLOCATION function instances=PrepareNextY limit=1
#pragma HLS ALLOCATION function instances=MM limit=1
#pragma HLS ALLOCATION function instances=WriteBack limit=1
             if (pp == 0)
             {

            	 PrepareNextY(W_row_index, Y_global_r, Y_buf0, step, YT_col_part);
            	 MM(Y_buf1, W, R_buf1);
            	 WriteBack(R_buf0, Y_global_w,step, YT_col_part);
             }
             else
             {
            	 PrepareNextY(W_row_index, Y_global_r, Y_buf1, step, YT_col_part);
                 MM(Y_buf0, W, R_buf0);
                 WriteBack(R_buf1, Y_global_w,step, YT_col_part);
             }
             pp = 1 - pp;

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


    ap_uint<32 * Index_per_group> W_row_index_buff[32][Layer][Neuron / 16 * Index_groups];
#pragma HLS ARRAY_PARTITION dim=1 type=complete variable=W_row_index_buff
    for(int i=0;i<Layer;i++)
    	for(int j=0;j<Neuron / 16 * Index_groups;j++)
    	{
            for(int k=0;k<32;k++)
    		W_row_index_buff[k][i][j]=W_row_index[i][j];
    	}


    for (int layer = 0; layer < Layer; layer++)
    {
    PE_GEN_LOOP:
        if(layer%2==0)
        {
            PE(W_row_index_buff[0 ][layer], YT_global_0 [0],YT_global_0 [1],layer);
            PE(W_row_index_buff[1 ][layer], YT_global_1 [0],YT_global_1 [1],layer);
            PE(W_row_index_buff[2 ][layer], YT_global_2 [0],YT_global_2 [1],layer);
            PE(W_row_index_buff[3 ][layer], YT_global_3 [0],YT_global_3 [1],layer);
            PE(W_row_index_buff[4 ][layer], YT_global_4 [0],YT_global_4 [1],layer);
            PE(W_row_index_buff[5 ][layer], YT_global_5 [0],YT_global_5 [1],layer);
            PE(W_row_index_buff[6 ][layer], YT_global_6 [0],YT_global_6 [1],layer);
            PE(W_row_index_buff[7 ][layer], YT_global_7 [0],YT_global_7 [1],layer);
            PE(W_row_index_buff[8 ][layer], YT_global_8 [0],YT_global_8 [1],layer);
            PE(W_row_index_buff[9 ][layer], YT_global_9 [0],YT_global_9 [1],layer);
            PE(W_row_index_buff[10][layer], YT_global_10[0],YT_global_10[1],layer);
            PE(W_row_index_buff[11][layer], YT_global_11[0],YT_global_11[1],layer);
            PE(W_row_index_buff[12][layer], YT_global_12[0],YT_global_12[1],layer);
            PE(W_row_index_buff[13][layer], YT_global_13[0],YT_global_13[1],layer);
            PE(W_row_index_buff[14][layer], YT_global_14[0],YT_global_14[1],layer);
            PE(W_row_index_buff[15][layer], YT_global_15[0],YT_global_15[1],layer);
            PE(W_row_index_buff[16][layer], YT_global_16[0],YT_global_16[1],layer);
            PE(W_row_index_buff[17][layer], YT_global_17[0],YT_global_17[1],layer);
            PE(W_row_index_buff[18][layer], YT_global_18[0],YT_global_18[1],layer);
            PE(W_row_index_buff[19][layer], YT_global_19[0],YT_global_19[1],layer);
            PE(W_row_index_buff[20][layer], YT_global_20[0],YT_global_20[1],layer);
            PE(W_row_index_buff[21][layer], YT_global_21[0],YT_global_21[1],layer);
            PE(W_row_index_buff[22][layer], YT_global_22[0],YT_global_22[1],layer);
            PE(W_row_index_buff[23][layer], YT_global_23[0],YT_global_23[1],layer);
            PE(W_row_index_buff[24][layer], YT_global_24[0],YT_global_24[1],layer);
            PE(W_row_index_buff[25][layer], YT_global_25[0],YT_global_25[1],layer);
            PE(W_row_index_buff[26][layer], YT_global_26[0],YT_global_26[1],layer);
            PE(W_row_index_buff[27][layer], YT_global_27[0],YT_global_27[1],layer);
            PE(W_row_index_buff[28][layer], YT_global_28[0],YT_global_28[1],layer);
            PE(W_row_index_buff[29][layer], YT_global_29[0],YT_global_29[1],layer);
            PE(W_row_index_buff[30][layer], YT_global_30[0],YT_global_30[1],layer);
            PE(W_row_index_buff[31][layer], YT_global_31[0],YT_global_31[1],layer);

        }
        else
        {
            PE(W_row_index_buff[0 ][layer], YT_global_0 [1],YT_global_0 [0],layer);
            PE(W_row_index_buff[1 ][layer], YT_global_1 [1],YT_global_1 [0],layer);
            PE(W_row_index_buff[2 ][layer], YT_global_2 [1],YT_global_2 [0],layer);
            PE(W_row_index_buff[3 ][layer], YT_global_3 [1],YT_global_3 [0],layer);
            PE(W_row_index_buff[4 ][layer], YT_global_4 [1],YT_global_4 [0],layer);
            PE(W_row_index_buff[5 ][layer], YT_global_5 [1],YT_global_5 [0],layer);
            PE(W_row_index_buff[6 ][layer], YT_global_6 [1],YT_global_6 [0],layer);
            PE(W_row_index_buff[7 ][layer], YT_global_7 [1],YT_global_7 [0],layer);
            PE(W_row_index_buff[8 ][layer], YT_global_8 [1],YT_global_8 [0],layer);
            PE(W_row_index_buff[9 ][layer], YT_global_9 [1],YT_global_9 [0],layer);
            PE(W_row_index_buff[10][layer], YT_global_10[1],YT_global_10[0],layer);
            PE(W_row_index_buff[11][layer], YT_global_11[1],YT_global_11[0],layer);
            PE(W_row_index_buff[12][layer], YT_global_12[1],YT_global_12[0],layer);
            PE(W_row_index_buff[13][layer], YT_global_13[1],YT_global_13[0],layer);
            PE(W_row_index_buff[14][layer], YT_global_14[1],YT_global_14[0],layer);
            PE(W_row_index_buff[15][layer], YT_global_15[1],YT_global_15[0],layer);
            PE(W_row_index_buff[16][layer], YT_global_16[1],YT_global_16[0],layer);
            PE(W_row_index_buff[17][layer], YT_global_17[1],YT_global_17[0],layer);
            PE(W_row_index_buff[18][layer], YT_global_18[1],YT_global_18[0],layer);
            PE(W_row_index_buff[19][layer], YT_global_19[1],YT_global_19[0],layer);
            PE(W_row_index_buff[20][layer], YT_global_20[1],YT_global_20[0],layer);
            PE(W_row_index_buff[21][layer], YT_global_21[1],YT_global_21[0],layer);
            PE(W_row_index_buff[22][layer], YT_global_22[1],YT_global_22[0],layer);
            PE(W_row_index_buff[23][layer], YT_global_23[1],YT_global_23[0],layer);
            PE(W_row_index_buff[24][layer], YT_global_24[1],YT_global_24[0],layer);
            PE(W_row_index_buff[25][layer], YT_global_25[1],YT_global_25[0],layer);
            PE(W_row_index_buff[26][layer], YT_global_26[1],YT_global_26[0],layer);
            PE(W_row_index_buff[27][layer], YT_global_27[1],YT_global_27[0],layer);
            PE(W_row_index_buff[28][layer], YT_global_28[1],YT_global_28[0],layer);
            PE(W_row_index_buff[29][layer], YT_global_29[1],YT_global_29[0],layer);
            PE(W_row_index_buff[30][layer], YT_global_30[1],YT_global_30[0],layer);
            PE(W_row_index_buff[31][layer], YT_global_31[1],YT_global_31[0],layer);
        }

    }
}
