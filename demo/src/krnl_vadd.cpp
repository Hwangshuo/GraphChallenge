
#define Neuron 1024
#define Image 60000
#include <stdint.h>

typedef struct Y_datatype
{
    float data[Image * Neuron / 32];
} Y_dt;

typedef struct W_datatype
{
    float data[Neuron * Neuron];
} W_dt;

typedef struct Z_datatype
{
    float data[Image * Neuron / 32];
} Z_dt;

void PE(const Y_dt *inY, const W_dt *inW, const unsigned int index, Z_dt *outZ)
{

    //! Y矩阵Image X (Neuron / 32)，W矩阵Neuron X Neuron
    //! 首先计算出需要用到的W矩阵的行范围
    //! 换言之 只需要用到第W矩阵的W_col_rang_low到W_col_rang_high行
    int W_col_rang_low = index * Neuron / 32;
    int W_col_rang_high = (index + 1) * Neuron / 32 - 1;

    float W[Neuron / 32][Neuron / 32];
    for (int i = 0; i < Neuron / 32; i++)
    {
        for (int j = 0; j < Neuron / 32; j++)
        {
            W[i][j] = inW->data[(W_col_rang_low + i) * 32 + j];
        }
    }

    float Y_global[Image][Neuron / 32];
    for (int i = 0; i < Image; i++)
    {
        for (int j = 0; j < Neuron / 32; j++)
        {
            Y_global[i][j] = inY->data[i * Neuron / 32 + j];
        }
    }

    int start = 0;
    // float temp_Z[Neuron / 32][Neuron];

    for (int i = 0; i < Image; i++)
    {
        for (int j = 0; j < Neuron / 32; j++)
        {
            double temp = 0;

            for (int k = 0; k < Neuron / 32; k++)
            {
                temp += Y_global[i][k] * W[k][i];
            }

            // temp_Z[i][j] = temp;
            outZ->data[start] = temp;
            start++;
        }
    }
    //! 将矩阵运算的结果 Image x Neuron / 32 按行存放到outZ中

    // }
}

// void PE(int layer, , int W_row_index[Neuron / 16 * 32], outol data_prepare)
// {
// }
extern "C"
{
    void krnl_vadd(
        const Y_dt *inY0,
        const Y_dt *inY1,
        const Y_dt *inY2,
        const Y_dt *inY3,
        const Y_dt *inY4,
        const Y_dt *inY5,
        const Y_dt *inY6,
        const Y_dt *inY7,
        const Y_dt *inY8,
        const Y_dt *inY9,
        const Y_dt *inY10,
        const Y_dt *inY11,
        const Y_dt *inY12,
        const Y_dt *inY13,
        const Y_dt *inY14,
        const Y_dt *inY15,
        const Y_dt *inY16,
        const Y_dt *inY17,
        const Y_dt *inY18,
        const Y_dt *inY19,
        const Y_dt *inY20,
        const Y_dt *inY21,
        const Y_dt *inY22,
        const Y_dt *inY23,
        const Y_dt *inY24,
        const Y_dt *inY25,
        const Y_dt *inY26,
        const Y_dt *inY27,
        const Y_dt *inY28,
        const Y_dt *inY29,
        const Y_dt *inY30,
        const Y_dt *inY31,

        Z_dt *outZ0,
        Z_dt *outZ1,
        Z_dt *outZ2,
        Z_dt *outZ3,
        Z_dt *outZ4,
        Z_dt *outZ5,
        Z_dt *outZ6,
        Z_dt *outZ7,
        Z_dt *outZ8,
        Z_dt *outZ9,
        Z_dt *outZ10,
        Z_dt *outZ11,
        Z_dt *outZ12,
        Z_dt *outZ13,
        Z_dt *outZ14,
        Z_dt *outZ15,
        Z_dt *outZ16,
        Z_dt *outZ17,
        Z_dt *outZ18,
        Z_dt *outZ19,
        Z_dt *outZ20,
        Z_dt *outZ21,
        Z_dt *outZ22,
        Z_dt *outZ23,
        Z_dt *outZ24,
        Z_dt *outZ25,
        Z_dt *outZ26,
        Z_dt *outZ27,
        Z_dt *outZ28,
        Z_dt *outZ29,
        Z_dt *outZ30,
        Z_dt *outZ31,

        const W_dt *inW)
    {
#pragma HLS INTERFACE m_axi port = inY0 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = inY1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = inY2 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = inY3 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = inY4 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = inY5 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = inY6 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = inY7 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = inY8 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = inY9 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = inY10 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = inY11 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = inY12 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = inY13 offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = inY14 offset = slave bundle = gmem14
#pragma HLS INTERFACE m_axi port = inY15 offset = slave bundle = gmem15
#pragma HLS INTERFACE m_axi port = inY16 offset = slave bundle = gmem16
#pragma HLS INTERFACE m_axi port = inY17 offset = slave bundle = gmem17
#pragma HLS INTERFACE m_axi port = inY18 offset = slave bundle = gmem18
#pragma HLS INTERFACE m_axi port = inY19 offset = slave bundle = gmem19
#pragma HLS INTERFACE m_axi port = inY20 offset = slave bundle = gmem20
#pragma HLS INTERFACE m_axi port = inY21 offset = slave bundle = gmem21
#pragma HLS INTERFACE m_axi port = inY22 offset = slave bundle = gmem22
#pragma HLS INTERFACE m_axi port = inY23 offset = slave bundle = gmem23
#pragma HLS INTERFACE m_axi port = inY24 offset = slave bundle = gmem24
#pragma HLS INTERFACE m_axi port = inY25 offset = slave bundle = gmem25
#pragma HLS INTERFACE m_axi port = inY26 offset = slave bundle = gmem26
#pragma HLS INTERFACE m_axi port = inY27 offset = slave bundle = gmem27
#pragma HLS INTERFACE m_axi port = inY28 offset = slave bundle = gmem28
#pragma HLS INTERFACE m_axi port = inY29 offset = slave bundle = gmem29
#pragma HLS INTERFACE m_axi port = inY30 offset = slave bundle = gmem30
#pragma HLS INTERFACE m_axi port = inY31 offset = slave bundle = gmem31

#pragma HLS INTERFACE m_axi port = inW offset = slave bundle = gmem_DDR0

#pragma HLS INTERFACE m_axi port = outZ0 offset = slave bundle = gmem0
#pragma HLS INTERFACE m_axi port = outZ1 offset = slave bundle = gmem1
#pragma HLS INTERFACE m_axi port = outZ2 offset = slave bundle = gmem2
#pragma HLS INTERFACE m_axi port = outZ3 offset = slave bundle = gmem3
#pragma HLS INTERFACE m_axi port = outZ4 offset = slave bundle = gmem4
#pragma HLS INTERFACE m_axi port = outZ5 offset = slave bundle = gmem5
#pragma HLS INTERFACE m_axi port = outZ6 offset = slave bundle = gmem6
#pragma HLS INTERFACE m_axi port = outZ7 offset = slave bundle = gmem7
#pragma HLS INTERFACE m_axi port = outZ8 offset = slave bundle = gmem8
#pragma HLS INTERFACE m_axi port = outZ9 offset = slave bundle = gmem9
#pragma HLS INTERFACE m_axi port = outZ10 offset = slave bundle = gmem10
#pragma HLS INTERFACE m_axi port = outZ11 offset = slave bundle = gmem11
#pragma HLS INTERFACE m_axi port = outZ12 offset = slave bundle = gmem12
#pragma HLS INTERFACE m_axi port = outZ13 offset = slave bundle = gmem13
#pragma HLS INTERFACE m_axi port = outZ14 offset = slave bundle = gmem14
#pragma HLS INTERFACE m_axi port = outZ15 offset = slave bundle = gmem15
#pragma HLS INTERFACE m_axi port = outZ16 offset = slave bundle = gmem16
#pragma HLS INTERFACE m_axi port = outZ17 offset = slave bundle = gmem17
#pragma HLS INTERFACE m_axi port = outZ18 offset = slave bundle = gmem18
#pragma HLS INTERFACE m_axi port = outZ19 offset = slave bundle = gmem19
#pragma HLS INTERFACE m_axi port = outZ20 offset = slave bundle = gmem20
#pragma HLS INTERFACE m_axi port = outZ21 offset = slave bundle = gmem21
#pragma HLS INTERFACE m_axi port = outZ22 offset = slave bundle = gmem22
#pragma HLS INTERFACE m_axi port = outZ23 offset = slave bundle = gmem23
#pragma HLS INTERFACE m_axi port = outZ24 offset = slave bundle = gmem24
#pragma HLS INTERFACE m_axi port = outZ25 offset = slave bundle = gmem25
#pragma HLS INTERFACE m_axi port = outZ26 offset = slave bundle = gmem26
#pragma HLS INTERFACE m_axi port = outZ27 offset = slave bundle = gmem27
#pragma HLS INTERFACE m_axi port = outZ28 offset = slave bundle = gmem28
#pragma HLS INTERFACE m_axi port = outZ29 offset = slave bundle = gmem29
#pragma HLS INTERFACE m_axi port = outZ30 offset = slave bundle = gmem30
#pragma HLS INTERFACE m_axi port = outZ31 offset = slave bundle = gmem31

#pragma HLS INTERFACE s_axilite port = inY0
#pragma HLS INTERFACE s_axilite port = inY1
#pragma HLS INTERFACE s_axilite port = inY2
#pragma HLS INTERFACE s_axilite port = inY3
#pragma HLS INTERFACE s_axilite port = inY4
#pragma HLS INTERFACE s_axilite port = inY5
#pragma HLS INTERFACE s_axilite port = inY6
#pragma HLS INTERFACE s_axilite port = inY7
#pragma HLS INTERFACE s_axilite port = inY8
#pragma HLS INTERFACE s_axilite port = inY9
#pragma HLS INTERFACE s_axilite port = inY10
#pragma HLS INTERFACE s_axilite port = inY11
#pragma HLS INTERFACE s_axilite port = inY12
#pragma HLS INTERFACE s_axilite port = inY13
#pragma HLS INTERFACE s_axilite port = inY14
#pragma HLS INTERFACE s_axilite port = inY15
#pragma HLS INTERFACE s_axilite port = inY16
#pragma HLS INTERFACE s_axilite port = inY17
#pragma HLS INTERFACE s_axilite port = inY18
#pragma HLS INTERFACE s_axilite port = inY19
#pragma HLS INTERFACE s_axilite port = inY20
#pragma HLS INTERFACE s_axilite port = inY21
#pragma HLS INTERFACE s_axilite port = inY22
#pragma HLS INTERFACE s_axilite port = inY23
#pragma HLS INTERFACE s_axilite port = inY24
#pragma HLS INTERFACE s_axilite port = inY25
#pragma HLS INTERFACE s_axilite port = inY26
#pragma HLS INTERFACE s_axilite port = inY27
#pragma HLS INTERFACE s_axilite port = inY28
#pragma HLS INTERFACE s_axilite port = inY29
#pragma HLS INTERFACE s_axilite port = inY30
#pragma HLS INTERFACE s_axilite port = inY31

#pragma HLS INTERFACE s_axilite port = outZ0
#pragma HLS INTERFACE s_axilite port = outZ1
#pragma HLS INTERFACE s_axilite port = outZ2
#pragma HLS INTERFACE s_axilite port = outZ3
#pragma HLS INTERFACE s_axilite port = outZ4
#pragma HLS INTERFACE s_axilite port = outZ5
#pragma HLS INTERFACE s_axilite port = outZ6
#pragma HLS INTERFACE s_axilite port = outZ7
#pragma HLS INTERFACE s_axilite port = outZ8
#pragma HLS INTERFACE s_axilite port = outZ9
#pragma HLS INTERFACE s_axilite port = outZ10
#pragma HLS INTERFACE s_axilite port = outZ11
#pragma HLS INTERFACE s_axilite port = outZ12
#pragma HLS INTERFACE s_axilite port = outZ13
#pragma HLS INTERFACE s_axilite port = outZ14
#pragma HLS INTERFACE s_axilite port = outZ15
#pragma HLS INTERFACE s_axilite port = outZ16
#pragma HLS INTERFACE s_axilite port = outZ17
#pragma HLS INTERFACE s_axilite port = outZ18
#pragma HLS INTERFACE s_axilite port = outZ19
#pragma HLS INTERFACE s_axilite port = outZ20
#pragma HLS INTERFACE s_axilite port = outZ21
#pragma HLS INTERFACE s_axilite port = outZ22
#pragma HLS INTERFACE s_axilite port = outZ23
#pragma HLS INTERFACE s_axilite port = outZ24
#pragma HLS INTERFACE s_axilite port = outZ25
#pragma HLS INTERFACE s_axilite port = outZ26
#pragma HLS INTERFACE s_axilite port = outZ27
#pragma HLS INTERFACE s_axilite port = outZ28
#pragma HLS INTERFACE s_axilite port = outZ29
#pragma HLS INTERFACE s_axilite port = outZ30
#pragma HLS INTERFACE s_axilite port = outZ31

#pragma HLS INTERFACE s_axilite port = inW

#pragma HLS INTERFACE s_axilite port = return

        PE(inY0, inW, 0, outZ0);
        PE(inY1, inW, 1, outZ1);
        PE(inY2, inW, 2, outZ2);
        PE(inY3, inW, 3, outZ3);
        PE(inY4, inW, 4, outZ4);
        PE(inY5, inW, 5, outZ5);
        PE(inY6, inW, 6, outZ6);
        PE(inY7, inW, 7, outZ7);
        PE(inY8, inW, 8, outZ8);
        PE(inY9, inW, 9, outZ9);
        PE(inY10, inW, 10, outZ10);
        PE(inY11, inW, 11, outZ11);
        PE(inY12, inW, 12, outZ12);
        PE(inY13, inW, 13, outZ13);
        PE(inY14, inW, 14, outZ14);
        PE(inY15, inW, 15, outZ15);
        PE(inY16, inW, 16, outZ16);
        PE(inY17, inW, 17, outZ17);
        PE(inY18, inW, 18, outZ18);
        PE(inY19, inW, 19, outZ19);
        PE(inY20, inW, 20, outZ20);
        PE(inY21, inW, 21, outZ21);
        PE(inY22, inW, 22, outZ22);
        PE(inY23, inW, 23, outZ23);
        PE(inY24, inW, 24, outZ24);
        PE(inY25, inW, 25, outZ25);
        PE(inY26, inW, 26, outZ26);
        PE(inY27, inW, 27, outZ27);
        PE(inY28, inW, 28, outZ28);
        PE(inY29, inW, 29, outZ29);
        PE(inY30, inW, 30, outZ30);
        PE(inY31, inW, 31, outZ31);
    }
}
