#pragma once

#define Neuron 1024
#define Image 60000
#define Mem_channel 32
#define YT_col_channel (Image / Mem_channel)
#define YT_col_tile 16
#define YT_row_tile 32
#define WT_row_tile 16
#define Layer 1024

#define LOG2_WT_row_tile 4
#define LOG2_YT_col_tile 4

#define Index_per_group 4
#define Index_groups (32 / Index_per_group)
#define all_active 65535 // YT_col_tile bits all 1

#define w 0.0625
#define bias -0.3

union uint2float
{
	unsigned int u;
	float f;
};

enum
{
	row_tile_finish=1,
	col_tile_finish=2,
	layer_finish=4,
	all_finish=8
};
// const float w = 0.0625;
// const map<int, float> bias_map = {
// 	{1024, -0.3},
// 	{4096, -0.35},
// 	{16384, -0.4},
// 	{65536, -0.45}};
// const float bias = bias_map.at(Neuron);
