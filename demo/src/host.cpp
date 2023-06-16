/**
 * Copyright (C) 2019-2021 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

/********************************************************************************************
 * Description:
 *
 * Xilinx's High Bandwidth Memory (HBM)-enabled FPGA are the clear solution for
 * providing
 * massive memory bandwidth within the lowest power, footprint, and system cost
 * envolopes.
 *
 * This example is designed to show a simple use case to understand how to use
 * HBM memory.
 *
 * There are total 32 memory resources referenced as HBM[0:31] by V++ and each
 * has a
 * capacity of storing 256MB of data.
 *
 * This example showcases two use cases to differentiate how efficiently one can
 * use HBM banks.
 *
 * CASE 1:
 *          +-----------+                   +-----------+
 *          |           | ---- Input1 ----> |           |
 *          |           |                   |           |
 *          |   HBM0    | ---- Input2 ----> |   KERNEL  |
 *          |           |                   |           |
 *          |           | <---- Output ---- |           |
 *          +-----------+                   +-----------+
 *
 *  In this case only one HBM Bank, i.e. HBM0, has been used for both the input
 *  vectors and the processed output vector.
 *
 *  CASE 2:
 *          +-----------+                   +-----------+
 *          |           |                   |           |
 *          |   HBM1    | ---- Input1 ----> |           |
 *          |           |                   |           |
 *          +-----------+                   |           |
 *                                          |           |
 *          +-----------+                   |           |
 *          |           |                   |   KERNEL  |
 *          |   HBM2    | ---- Input2 ----> |           |
 *          |           |                   |           |
 *          +-----------+                   |           |
 *                                          |           |
 *          +-----------+                   |           |
 *          |           |                   |           |
 *          |   HBM3    | <---- Output ---- |           |
 *          |           |                   |           |
 *          +-----------+                   +-----------+
 *
 *  In this case three different HBM Banks, i.e. HBM1, HBM2 and HBM3, have been
 * used for input
 *  vectors and the processed output vector.
 *  The banks HBM1 & HBM2 are used for input vectors whereas HBM3 is used for
 *  processed output vector.
 *
 *  The use case highlights significant change in the data transfer throughput
 * (in terms of
 *  Gigabytes per second) when a single and multiple HBM banks are used for the
 *  same application.
 *
 *  *****************************************************************************************/
#include "cmdlineparser.h"
#include <iostream>
#include <cstring>

// XRT includes
#include "experimental/xrt_bo.h"
#include "experimental/xrt_device.h"
#include "experimental/xrt_kernel.h"

#include <stdint.h>
#include <stdlib.h>

#include <string>

#include <fstream>
#include <sstream>
using namespace std;

#define Neuron 1024
#define Image 60000

float Y_T_global[Image][Neuron] = {0};
float W_T_global[Neuron][Neuron] = {0};

void read_Y(std::string fname)
{
    ifstream fp(fname);
    string line;
    stringstream ss;

    int i;
    int j;
    float v;

    while (getline(fp, line))
    {
        ss.str("");
        ss.clear();
        ss << line;

        ss >> i;
        ss >> j;
        ss >> v;
        Y_T_global[i][j] = v;
    }
    return;
}

void read_W(std::string fname)
{
    ifstream fp(fname);
    string line;
    stringstream ss;

    int i;
    int j;
    float v;

    while (getline(fp, line))
    {
        ss.str("");
        ss.clear();
        ss << line;

        ss >> i;
        ss >> j;
        ss >> v;
        W_T_global[i - 1][j - 1] = v;
    } // 权重矩阵
    return;
}

double run_krnl(xrtDeviceHandle device, xrt::kernel &krnl)
{

    //! 计划将6000x1024个Y分成32个bank存放
    //! 将二维数组转换为一维数组存放，每个子矩阵Y_part实际上要存放的信息是 6w*(Neuron/32) 即6wx32
    int Y_part_col_tile_in = Neuron / 32;                               //! 列数
    int Z_part_col_tile_in = Neuron / 32;                               //! 列数
    int Y_part_size_bytes = sizeof(float) * Y_part_col_tile_in * Image; // !按列划分每个Y_part，每个part需要的字节数

    std::cout << "Allocate Buffer in Global Memory\n";
    auto boY0_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(0));
    auto boY1_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(1));
    auto boY2_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(2));
    auto boY3_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(3));
    auto boY4_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(4));
    auto boY5_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(5));
    auto boY6_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(6));
    auto boY7_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(7));
    auto boY8_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(8));
    auto boY9_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(9));
    auto boY10_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(10));
    auto boY11_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(11));
    auto boY12_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(12));
    auto boY13_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(13));
    auto boY14_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(14));
    auto boY15_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(15));
    auto boY16_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(16));
    auto boY17_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(17));
    auto boY18_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(18));
    auto boY19_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(19));
    auto boY20_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(20));
    auto boY21_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(21));
    auto boY22_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(22));
    auto boY23_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(23));
    auto boY24_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(24));
    auto boY25_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(25));
    auto boY26_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(26));
    auto boY27_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(27));
    auto boY28_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(28));
    auto boY29_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(29));
    auto boY30_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(30));
    auto boY31_in = xrt::bo(device, Y_part_size_bytes, krnl.group_id(31));

    int Z_part_size_bytes = sizeof(float) * Z_part_col_tile_in * Image;

    auto boZ0_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(32));
    auto boZ1_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(33));
    auto boZ2_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(34));
    auto boZ3_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(35));
    auto boZ4_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(36));
    auto boZ5_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(37));
    auto boZ6_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(38));
    auto boZ7_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(39));
    auto boZ8_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(40));
    auto boZ9_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(41));
    auto boZ10_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(42));
    auto boZ11_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(43));
    auto boZ12_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(44));
    auto boZ13_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(45));
    auto boZ14_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(46));
    auto boZ15_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(47));
    auto boZ16_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(48));
    auto boZ17_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(49));
    auto boZ18_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(50));
    auto boZ19_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(51));
    auto boZ20_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(52));
    auto boZ21_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(53));
    auto boZ22_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(54));
    auto boZ23_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(55));
    auto boZ24_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(56));
    auto boZ25_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(57));
    auto boZ26_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(58));
    auto boZ27_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(59));
    auto boZ28_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(60));
    auto boZ29_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(61));
    auto boZ30_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(62));
    auto boZ31_out = xrt::bo(device, Z_part_size_bytes, krnl.group_id(63));

    //! W会全部存放进DDR中，为了表示二维，只需要列数即可，与Y的列数相同，均为Neuron
    auto boW_in = xrt::bo(device, Neuron * Neuron * sizeof(float), krnl.group_id(64));

    std::cout << "Map the contents of the buffer object into host memory \n";
    auto boY0_in_map = boY0_in.map<float *>();
    auto boY1_in_map = boY1_in.map<float *>();
    auto boY2_in_map = boY2_in.map<float *>();
    auto boY3_in_map = boY3_in.map<float *>();
    auto boY4_in_map = boY4_in.map<float *>();
    auto boY5_in_map = boY5_in.map<float *>();
    auto boY6_in_map = boY6_in.map<float *>();
    auto boY7_in_map = boY7_in.map<float *>();
    auto boY8_in_map = boY8_in.map<float *>();
    auto boY9_in_map = boY9_in.map<float *>();
    auto boY10_in_map = boY10_in.map<float *>();
    auto boY11_in_map = boY11_in.map<float *>();
    auto boY12_in_map = boY12_in.map<float *>();
    auto boY13_in_map = boY13_in.map<float *>();
    auto boY14_in_map = boY14_in.map<float *>();
    auto boY15_in_map = boY15_in.map<float *>();
    auto boY16_in_map = boY16_in.map<float *>();
    auto boY17_in_map = boY17_in.map<float *>();
    auto boY18_in_map = boY18_in.map<float *>();
    auto boY19_in_map = boY19_in.map<float *>();
    auto boY20_in_map = boY20_in.map<float *>();
    auto boY21_in_map = boY21_in.map<float *>();
    auto boY22_in_map = boY22_in.map<float *>();
    auto boY23_in_map = boY23_in.map<float *>();
    auto boY24_in_map = boY24_in.map<float *>();
    auto boY25_in_map = boY25_in.map<float *>();
    auto boY26_in_map = boY26_in.map<float *>();
    auto boY27_in_map = boY27_in.map<float *>();
    auto boY28_in_map = boY28_in.map<float *>();
    auto boY29_in_map = boY29_in.map<float *>();
    auto boY30_in_map = boY30_in.map<float *>();
    auto boY31_in_map = boY31_in.map<float *>();

    auto boZ0_out_map = boZ0_out.map<float *>();
    auto boZ1_out_map = boZ1_out.map<float *>();
    auto boZ2_out_map = boZ2_out.map<float *>();
    auto boZ3_out_map = boZ3_out.map<float *>();
    auto boZ4_out_map = boZ4_out.map<float *>();
    auto boZ5_out_map = boZ5_out.map<float *>();
    auto boZ6_out_map = boZ6_out.map<float *>();
    auto boZ7_out_map = boZ7_out.map<float *>();
    auto boZ8_out_map = boZ8_out.map<float *>();
    auto boZ9_out_map = boZ9_out.map<float *>();
    auto boZ10_out_map = boZ10_out.map<float *>();
    auto boZ11_out_map = boZ11_out.map<float *>();
    auto boZ12_out_map = boZ12_out.map<float *>();
    auto boZ13_out_map = boZ13_out.map<float *>();
    auto boZ14_out_map = boZ14_out.map<float *>();
    auto boZ15_out_map = boZ15_out.map<float *>();
    auto boZ16_out_map = boZ16_out.map<float *>();
    auto boZ17_out_map = boZ17_out.map<float *>();
    auto boZ18_out_map = boZ18_out.map<float *>();
    auto boZ19_out_map = boZ19_out.map<float *>();
    auto boZ20_out_map = boZ20_out.map<float *>();
    auto boZ21_out_map = boZ21_out.map<float *>();
    auto boZ22_out_map = boZ22_out.map<float *>();
    auto boZ23_out_map = boZ23_out.map<float *>();
    auto boZ24_out_map = boZ24_out.map<float *>();
    auto boZ25_out_map = boZ25_out.map<float *>();
    auto boZ26_out_map = boZ26_out.map<float *>();
    auto boZ27_out_map = boZ27_out.map<float *>();
    auto boZ28_out_map = boZ28_out.map<float *>();
    auto boZ29_out_map = boZ29_out.map<float *>();
    auto boZ30_out_map = boZ30_out.map<float *>();
    auto boZ31_out_map = boZ31_out.map<float *>();

    auto boW_in_map = boW_in.map<float *>();

    std::cout << "initial Global Memory \n";
    std::fill(boY0_in_map, boY0_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY1_in_map, boY1_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY2_in_map, boY2_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY3_in_map, boY3_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY4_in_map, boY4_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY5_in_map, boY5_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY6_in_map, boY6_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY7_in_map, boY7_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY8_in_map, boY8_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY9_in_map, boY9_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY10_in_map, boY10_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY11_in_map, boY11_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY12_in_map, boY12_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY13_in_map, boY13_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY14_in_map, boY14_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY15_in_map, boY15_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY16_in_map, boY16_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY17_in_map, boY17_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY18_in_map, boY18_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY19_in_map, boY19_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY20_in_map, boY20_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY21_in_map, boY21_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY22_in_map, boY22_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY23_in_map, boY23_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY24_in_map, boY24_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY25_in_map, boY25_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY26_in_map, boY26_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY27_in_map, boY27_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY28_in_map, boY28_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY29_in_map, boY29_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY30_in_map, boY30_in_map + Y_part_col_tile_in * Image, 0);
    std::fill(boY31_in_map, boY31_in_map + Y_part_col_tile_in * Image, 0);

    std::fill(boZ0_out_map, boZ0_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ1_out_map, boZ1_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ2_out_map, boZ2_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ3_out_map, boZ3_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ4_out_map, boZ4_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ5_out_map, boZ5_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ6_out_map, boZ6_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ7_out_map, boZ7_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ8_out_map, boZ8_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ9_out_map, boZ9_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ10_out_map, boZ10_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ11_out_map, boZ11_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ12_out_map, boZ12_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ13_out_map, boZ13_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ14_out_map, boZ14_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ15_out_map, boZ15_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ16_out_map, boZ16_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ17_out_map, boZ17_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ18_out_map, boZ18_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ19_out_map, boZ19_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ20_out_map, boZ20_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ21_out_map, boZ21_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ22_out_map, boZ22_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ23_out_map, boZ23_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ24_out_map, boZ24_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ25_out_map, boZ25_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ26_out_map, boZ26_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ27_out_map, boZ27_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ28_out_map, boZ28_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ29_out_map, boZ29_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ30_out_map, boZ30_out_map + Z_part_col_tile_in * Image, 0);
    std::fill(boZ31_out_map, boZ31_out_map + Z_part_col_tile_in * Image, 0);

    std::fill(boW_in_map, boW_in_map + Neuron * Neuron, 0);
    std::cout << "read Graph success to host memory \n";
    string y_fname = "../../../dataset/sparse-images-1024.tsv";
    string w_fname = "../../../dataset/hash_neuron1024/n1024-l1.tsv";
    read_Y(y_fname);
    read_W(w_fname);
    std::cout << "read success \n";
    std::cout << "Create the test data Y \n";
    // Create the test data

    //! 按列分割 将二维数组Y_T_global[i][j]以Image x Neuron/32为1块传送到FPGA中
    for (int i = 0; i < Image; ++i)
    {
        for (int j = 0; j < Y_part_col_tile_in; ++j)
        {
            boY0_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j];
            boY1_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 1 * Y_part_col_tile_in];
            boY2_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 2 * Y_part_col_tile_in];
            boY3_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 3 * Y_part_col_tile_in];
            boY4_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 4 * Y_part_col_tile_in];
            boY5_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 5 * Y_part_col_tile_in];
            boY6_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 6 * Y_part_col_tile_in];
            boY7_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 7 * Y_part_col_tile_in];
            boY8_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 8 * Y_part_col_tile_in];
            boY9_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 9 * Y_part_col_tile_in];
            boY10_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 10 * Y_part_col_tile_in];
            boY11_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 11 * Y_part_col_tile_in];
            boY12_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 12 * Y_part_col_tile_in];
            boY13_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 13 * Y_part_col_tile_in];
            boY14_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 14 * Y_part_col_tile_in];
            boY15_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 15 * Y_part_col_tile_in];
            boY16_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 16 * Y_part_col_tile_in];
            boY17_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 17 * Y_part_col_tile_in];
            boY18_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 18 * Y_part_col_tile_in];
            boY19_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 19 * Y_part_col_tile_in];
            boY20_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 20 * Y_part_col_tile_in];
            boY21_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 21 * Y_part_col_tile_in];
            boY22_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 22 * Y_part_col_tile_in];
            boY23_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 23 * Y_part_col_tile_in];
            boY24_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 24 * Y_part_col_tile_in];
            boY25_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 25 * Y_part_col_tile_in];
            boY26_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 26 * Y_part_col_tile_in];
            boY27_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 27 * Y_part_col_tile_in];
            boY28_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 28 * Y_part_col_tile_in];
            boY29_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 29 * Y_part_col_tile_in];
            boY30_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 30 * Y_part_col_tile_in];
            boY31_in_map[i * Y_part_col_tile_in + j] = Y_T_global[i][j + 31 * Y_part_col_tile_in];
        }
    }
    std::cout << "Create the test data Y success\n";
    std::cout << "Create the test data W \n";
    for (int i = 0; i < Neuron; ++i)
    {
        for (int j = 0; j < Neuron; ++j)
        {
            boW_in_map[i * Neuron + j] = W_T_global[i][j];
        }
    }
    std::cout << "Create the test data W success\n";
    // Synchronize buffer content with device side
    std::cout << "synchronize input buffer data to device global memory\n";

    boY0_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY1_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY2_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY3_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY4_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY5_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY6_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY7_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY8_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY9_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY10_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY11_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY12_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY13_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY14_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY15_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY16_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY17_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY18_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY19_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY20_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY21_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY22_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY23_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY24_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY25_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY26_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY27_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY28_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY29_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY30_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    boY31_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    boW_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::chrono::duration<double> kernel_time(0);

    std::cout << "Execution of the kernel\n";
    auto kernel_start = std::chrono::high_resolution_clock::now();
    auto run = krnl(boY0_in, boY1_in, boY2_in, boY3_in, boY4_in, boY5_in, boY6_in, boY7_in, boY8_in, boY9_in,
                    boY10_in, boY11_in, boY12_in, boY13_in, boY14_in, boY15_in, boY16_in, boY17_in, boY18_in, boY19_in,
                    boY20_in, boY21_in, boY22_in, boY23_in, boY24_in, boY25_in, boY26_in, boY27_in, boY28_in, boY29_in,
                    boY30_in, boY31_in,
                    boZ0_out, boZ0_out, boZ0_out, boZ0_out, boZ4_out, boZ5_out, boZ6_out, boZ7_out, boZ8_out, boZ9_out,
                    boZ10_out, boZ11_out, boZ12_out, boZ13_out, boZ14_out, boZ15_out, boZ16_out, boZ17_out, boZ18_out, boZ19_out,
                    boZ20_out, boZ21_out, boZ22_out, boZ23_out, boZ24_out, boZ25_out, boZ26_out, boZ27_out, boZ28_out, boZ29_out,
                    boZ30_out, boZ31_out,
                    boW_in);
    run.wait();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    std::cout << "Execute the kernel success\n";
    kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
    // Get the output;
    std::cout << "Get the output data from the device" << std::endl;

    boZ0_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ1_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ2_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ3_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ4_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ5_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ6_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ7_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ8_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ9_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ10_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ11_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ12_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ13_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ14_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ15_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ16_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ17_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ18_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ19_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ20_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ21_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ22_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ23_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ24_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ25_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ26_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ27_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ28_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ29_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ30_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    boZ31_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    std::cout << "Create the reference result" << std::endl;
    float Reference_boZ0[Z_part_col_tile_in * Image];
    float Reference_boZ1[Z_part_col_tile_in * Image];
    float Reference_boZ2[Z_part_col_tile_in * Image];
    float Reference_boZ3[Z_part_col_tile_in * Image];
    float Reference_boZ4[Z_part_col_tile_in * Image];
    float Reference_boZ5[Z_part_col_tile_in * Image];
    float Reference_boZ6[Z_part_col_tile_in * Image];
    float Reference_boZ7[Z_part_col_tile_in * Image];
    float Reference_boZ8[Z_part_col_tile_in * Image];
    float Reference_boZ9[Z_part_col_tile_in * Image];
    float Reference_boZ10[Z_part_col_tile_in * Image];
    float Reference_boZ11[Z_part_col_tile_in * Image];
    float Reference_boZ12[Z_part_col_tile_in * Image];
    float Reference_boZ13[Z_part_col_tile_in * Image];
    float Reference_boZ14[Z_part_col_tile_in * Image];
    float Reference_boZ15[Z_part_col_tile_in * Image];
    float Reference_boZ16[Z_part_col_tile_in * Image];
    float Reference_boZ17[Z_part_col_tile_in * Image];
    float Reference_boZ18[Z_part_col_tile_in * Image];
    float Reference_boZ19[Z_part_col_tile_in * Image];
    float Reference_boZ20[Z_part_col_tile_in * Image];
    float Reference_boZ21[Z_part_col_tile_in * Image];
    float Reference_boZ22[Z_part_col_tile_in * Image];
    float Reference_boZ23[Z_part_col_tile_in * Image];
    float Reference_boZ24[Z_part_col_tile_in * Image];
    float Reference_boZ25[Z_part_col_tile_in * Image];
    float Reference_boZ26[Z_part_col_tile_in * Image];
    float Reference_boZ27[Z_part_col_tile_in * Image];
    float Reference_boZ28[Z_part_col_tile_in * Image];
    float Reference_boZ29[Z_part_col_tile_in * Image];
    float Reference_boZ30[Z_part_col_tile_in * Image];
    float Reference_boZ31[Z_part_col_tile_in * Image];
    int start = 0;
    for (int i = 0; i < Image; ++i)
    {
        for (int j = 0; j < Neuron / 32; j++)
        {
            double temp0 = 0;
            double temp1 = 0;
            double temp2 = 0;
            double temp3 = 0;
            double temp4 = 0;
            double temp5 = 0;
            double temp6 = 0;
            double temp7 = 0;
            double temp8 = 0;
            double temp9 = 0;
            double temp10 = 0;
            double temp11 = 0;
            double temp12 = 0;
            double temp13 = 0;
            double temp14 = 0;
            double temp15 = 0;
            double temp16 = 0;
            double temp17 = 0;
            double temp18 = 0;
            double temp19 = 0;
            double temp20 = 0;
            double temp21 = 0;
            double temp22 = 0;
            double temp23 = 0;
            double temp24 = 0;
            double temp25 = 0;
            double temp26 = 0;
            double temp27 = 0;
            double temp28 = 0;
            double temp29 = 0;
            double temp30 = 0;
            double temp31 = 0;

            for (int k = 0; k < Neuron / 32; k++)
            {
                temp0 += Y_T_global[i][k] * W_T_global[k][i];
                temp1 += Y_T_global[i][k + 1 * Y_part_col_tile_in] * W_T_global[k + 1 * Y_part_col_tile_in][i];
                temp2 += Y_T_global[i][k + 2 * Y_part_col_tile_in] * W_T_global[k + 2 * Y_part_col_tile_in][i];
                temp3 += Y_T_global[i][k + 3 * Y_part_col_tile_in] * W_T_global[k + 3 * Y_part_col_tile_in][i];
                temp4 += Y_T_global[i][k + 4 * Y_part_col_tile_in] * W_T_global[k + 4 * Y_part_col_tile_in][i];
                temp5 += Y_T_global[i][k + 5 * Y_part_col_tile_in] * W_T_global[k + 5 * Y_part_col_tile_in][i];
                temp6 += Y_T_global[i][k + 6 * Y_part_col_tile_in] * W_T_global[k + 6 * Y_part_col_tile_in][i];
                temp7 += Y_T_global[i][k + 7 * Y_part_col_tile_in] * W_T_global[k + 7 * Y_part_col_tile_in][i];
                temp8 += Y_T_global[i][k + 8 * Y_part_col_tile_in] * W_T_global[k + 8 * Y_part_col_tile_in][i];
                temp9 += Y_T_global[i][k + 9 * Y_part_col_tile_in] * W_T_global[k + 9 * Y_part_col_tile_in][i];
                temp10 += Y_T_global[i][k + 10 * Y_part_col_tile_in] * W_T_global[k + 10 * Y_part_col_tile_in][i];
                temp11 += Y_T_global[i][k + 11 * Y_part_col_tile_in] * W_T_global[k + 11 * Y_part_col_tile_in][i];
                temp12 += Y_T_global[i][k + 12 * Y_part_col_tile_in] * W_T_global[k + 12 * Y_part_col_tile_in][i];
                temp13 += Y_T_global[i][k + 13 * Y_part_col_tile_in] * W_T_global[k + 13 * Y_part_col_tile_in][i];
                temp14 += Y_T_global[i][k + 14 * Y_part_col_tile_in] * W_T_global[k + 14 * Y_part_col_tile_in][i];
                temp15 += Y_T_global[i][k + 15 * Y_part_col_tile_in] * W_T_global[k + 15 * Y_part_col_tile_in][i];
                temp16 += Y_T_global[i][k + 16 * Y_part_col_tile_in] * W_T_global[k + 16 * Y_part_col_tile_in][i];
                temp17 += Y_T_global[i][k + 17 * Y_part_col_tile_in] * W_T_global[k + 17 * Y_part_col_tile_in][i];
                temp18 += Y_T_global[i][k + 18 * Y_part_col_tile_in] * W_T_global[k + 18 * Y_part_col_tile_in][i];
                temp19 += Y_T_global[i][k + 19 * Y_part_col_tile_in] * W_T_global[k + 19 * Y_part_col_tile_in][i];
                temp20 += Y_T_global[i][k + 20 * Y_part_col_tile_in] * W_T_global[k + 20 * Y_part_col_tile_in][i];
                temp21 += Y_T_global[i][k + 21 * Y_part_col_tile_in] * W_T_global[k + 21 * Y_part_col_tile_in][i];
                temp22 += Y_T_global[i][k + 22 * Y_part_col_tile_in] * W_T_global[k + 22 * Y_part_col_tile_in][i];
                temp23 += Y_T_global[i][k + 23 * Y_part_col_tile_in] * W_T_global[k + 23 * Y_part_col_tile_in][i];
                temp24 += Y_T_global[i][k + 24 * Y_part_col_tile_in] * W_T_global[k + 24 * Y_part_col_tile_in][i];
                temp25 += Y_T_global[i][k + 25 * Y_part_col_tile_in] * W_T_global[k + 25 * Y_part_col_tile_in][i];
                temp26 += Y_T_global[i][k + 26 * Y_part_col_tile_in] * W_T_global[k + 26 * Y_part_col_tile_in][i];
                temp27 += Y_T_global[i][k + 27 * Y_part_col_tile_in] * W_T_global[k + 27 * Y_part_col_tile_in][i];
                temp28 += Y_T_global[i][k + 28 * Y_part_col_tile_in] * W_T_global[k + 28 * Y_part_col_tile_in][i];
                temp29 += Y_T_global[i][k + 29 * Y_part_col_tile_in] * W_T_global[k + 29 * Y_part_col_tile_in][i];
                temp30 += Y_T_global[i][k + 30 * Y_part_col_tile_in] * W_T_global[k + 30 * Y_part_col_tile_in][i];
                temp31 += Y_T_global[i][k + 31 * Y_part_col_tile_in] * W_T_global[k + 31 * Y_part_col_tile_in][i];
            }
            Reference_boZ0[start] = temp0;
            Reference_boZ1[start] = temp1;
            Reference_boZ2[start] = temp2;
            Reference_boZ3[start] = temp3;
            Reference_boZ4[start] = temp4;
            Reference_boZ5[start] = temp5;
            Reference_boZ6[start] = temp6;
            Reference_boZ7[start] = temp7;
            Reference_boZ8[start] = temp8;
            Reference_boZ9[start] = temp9;
            Reference_boZ10[start] = temp10;
            Reference_boZ11[start] = temp11;
            Reference_boZ12[start] = temp12;
            Reference_boZ13[start] = temp13;
            Reference_boZ14[start] = temp14;
            Reference_boZ15[start] = temp15;
            Reference_boZ16[start] = temp16;
            Reference_boZ17[start] = temp17;
            Reference_boZ18[start] = temp18;
            Reference_boZ19[start] = temp19;
            Reference_boZ20[start] = temp20;
            Reference_boZ21[start] = temp21;
            Reference_boZ22[start] = temp22;
            Reference_boZ23[start] = temp23;
            Reference_boZ24[start] = temp24;
            Reference_boZ25[start] = temp25;
            Reference_boZ26[start] = temp26;
            Reference_boZ27[start] = temp27;
            Reference_boZ28[start] = temp28;
            Reference_boZ29[start] = temp29;
            Reference_boZ30[start] = temp30;
            Reference_boZ31[start] = temp31;
            start++;
        }
    }
    std::cout << "Vertify the output data from device and the reference result" << std::endl;
    for (int i = 0; i < Image * Z_part_col_tile_in; i++)
    {
        if (Reference_boZ0[i] != boZ0_out_map[i])
        {
            cout << "wrong!" << endl;
            break;
        }
    }

    std::cout << "finish" << std::endl;
    return kernel_time.count();
}

int main(int argc, char *argv[])
{
    // Command Line Parser
    sda::utils::CmdLineParser parser;

    parser.addSwitch("--xclbin_file", "-x", "input binary file string", "");
    parser.addSwitch("--device_id", "-d", "device index", "0");
    parser.parse(argc, argv);

    // Read settings
    std::string binaryFile = parser.value("xclbin_file");
    int device_index = stoi(parser.value("device_id"));

    if (argc < 3)
    {
        parser.printHelp();
        return EXIT_FAILURE;
    }

    std::cout << "Open the device" << device_index << std::endl;
    auto device = xrt::device(device_index);
    std::cout << "Load the xclbin " << binaryFile << std::endl;
    auto uuid = device.load_xclbin(binaryFile);

    auto krnl = xrt::kernel(device, uuid, "krnl_vadd");

    unsigned int dataSize = Neuron * Image;
    double kernel_time_in_sec = 0, result = 0;

    kernel_time_in_sec = run_krnl(device, krnl); // 要处理的字的个数

    result = 3 * dataSize * sizeof(float);
    result /= (1000 * 1000 * 1000); // to GB
    result /= kernel_time_in_sec;   // to GBps

    std::cout << "THROUGHPUT = " << result << " GB/s " << std::endl;

    std::cout << "TEST PASSED" << std::endl;
    return 0;
}
