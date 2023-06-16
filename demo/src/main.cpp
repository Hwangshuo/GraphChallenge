#include <stdio.h>
#include <fstream>
#include <sstream>
#include <map>
#include <omp.h>

#define Neuron 1024
#define Image 60000
#define Y_row_channel 60000
#define Y_row_tile 8
#define Y_col_tile 32
#define W_col_tile 16
#define Layer 10

using namespace std;

float Y_T_global[2][Neuron][Image] = {0};
float W_T_global[Neuron][Neuron] = {0};
float R_test[Neuron][Image] = {0};
int W_row_index[Neuron / 16 * 32] = {0};
const float w = 0.0625;
const map<int, float> bias_map = {
	{1024, -0.3},
	{4096, -0.35},
	{16384, -0.4},
	{65536, -0.45}};
const float bias = bias_map.at(Neuron);

void read_Y(string fname)
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
		Y_T_global[0][j - 1][i - 1] = v;
		// ! 60000 x Neuron的矩阵 在内存中以转置存放
	}
}

void read_W(string fname)
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
		W_T_global[j][i] = v;
		// ! Neuron x Neuron的矩阵 在内存中以转置存放
	}

	int total = 0;
	for (i = 0; i < Neuron; i += 16)
	{
		int nnz = 0;
		for (j = 0; j < Neuron; j++)
		{
			if (W_T_global[i][j] != 0)
			{
				W_row_index[total] = j; // !记录转置矩阵中每个非0的列号，转置矩阵中每一行总共32个非0值，只需要记录首部即可，
				total++;				// W中总的非0个数
				nnz++;
			}

			if (nnz == 32)
				break;
		}
	}
}

void MM(const float intput_Y[Y_row_tile][Y_col_tile],
		const float input_W[Y_col_tile][W_col_tile],
		float R_buf[W_col_tile][Y_row_tile])
{
	float output_buf[Y_row_tile][W_col_tile];
	int k;
	float sum = 0;
	for (int i = 0; i < Y_row_tile; i++)
		for (int j = 0; j < W_col_tile; j++)
		{
			for (sum = 0, k = 0; k < Y_col_tile; k++)
				sum += intput_Y[i][k] * input_W[k][j];
			sum += bias;
			output_buf[i][j] = sum > 32 ? 32 : sum < 0 ? 0
													   : sum;
		}

	for (int j = 0; j < W_col_tile; j++)
		for (int i = 0; i < Y_row_tile; i++)
			R_buf[j][i] = output_buf[i][j];
}

void WriteBack(int Y_row_start,
			   int W_col_start,
			   const float R_buf[W_col_tile][Y_row_tile],
			   float output_Y[Neuron][Image])
{
	for (int j = 0; j < W_col_tile; j++)
		for (int i = 0; i < Y_row_tile; i++)
		{
			output_Y[W_col_start + j][Y_row_start + i] = R_buf[j][i];
		}
}

void PrepareNextW(int W_col_start,
				  int W_row_index[Neuron / 16 * 32],
				  int W_index_buf[Y_col_tile])
{
	for (int i = 0; i < Y_col_tile; i++)
		W_index_buf[i] = W_row_index[W_col_start + i];
}

void PrepareNextY(int Y_row_start,
				  int W_index_buf[Y_col_tile],
				  float Y_global[Neuron][Y_row_channel],
				  float Y_buf[Y_row_tile][Y_col_tile])
{
	for (int i = 0; i < Y_col_tile; i++)
		for (int j = 0; j < Y_row_tile; j++)
		{
			Y_buf[j][i] = Y_global[W_index_buf[i]][Y_row_start + j];
			// printf("fetch row %d, col %d, val %f\n", Y_row_start+i, W_index_buf[i], Y_global[W_index_buf[i]][j]);
			// if(Y_global[W_index_buf[i]][j]!=0)
			// 	getchar();
		}
}

void PE(int layer, float Y_global[2][Neuron][Y_row_channel], int W_row_index[Neuron / 16 * 32], bool data_prepare)
{
	static float Y_buf[2][Y_row_tile][Y_col_tile];
	static float W[Y_col_tile][W_col_tile];
	static int W_index_buf[Y_col_tile];
	static float R_buf[W_col_tile][Y_row_tile];

	if (data_prepare)
	{
		for (int i = 0; i < Y_col_tile; i++)
			for (int j = 0; j < W_col_tile; j++)
				W[i][j] = w;//! 生成一个32x16的矩阵W

		for (int i = 0; i < Y_col_tile; i++)
			W_index_buf[i] = W_row_index[i];//! 读取转置矩阵W第一行的32个所在列数

		for (int i = 0; i < Y_col_tile; i++)
			for (int j = 0; j < Y_row_tile; j++)
			{
				Y_buf[0][j][i] = Y_global[0][W_index_buf[i]][j]; //! 生成一个8 X 32 的矩阵
				// printf("fetch row %d, col %d, val %f\n", i, W_index_buf[i], Y_global[0][W_index_buf[i]][j]);
			}

		return;
	}

	for (int Y_row_start = 0; Y_row_start < 60000; Y_row_start += Y_row_tile)
	{
		for (int W_part = 0; W_part < Neuron / W_col_tile; W_part++)//! 将W进行分块 为 1024/16
		{
			// printf("%d\n", W_part);
			// for (int i = 0; i < Y_col_tile; i++)
			// 	printf("%d ", W_index_buf[i]);
			// printf("\n");

			MM(Y_buf[W_part % 2], W, R_buf);//! 8x32的矩阵与32x16的矩阵相乘，结果写入R_buff中，结果是16x8
			WriteBack(Y_row_start, W_part * W_col_tile, R_buf, Y_global[(layer + 1) % 2]);//! 将结果写回Y的转置中

			bool last_W_tile = W_part == Neuron / W_col_tile - 1;
			bool last_Y_tile = Y_row_start == 60000 - Y_row_tile;

			float(*next_Y_global)[60000] = (last_Y_tile && last_W_tile) ? Y_global[(layer + 1) % 2] : Y_global[layer % 2];
			int next_Y_row_start = last_W_tile ? Y_row_start + Y_row_tile : Y_row_start;
			int next_W_col_start = last_W_tile ? 0 : (W_part + 1) * Y_col_tile;
			// !进行下一轮的计算，将Y和W都做偏移 8x32 32x16
			PrepareNextW(next_W_col_start, W_row_index, W_index_buf);
			PrepareNextY(next_Y_row_start, W_index_buf, next_Y_global, Y_buf[(W_part + 1) % 2]);
		}
	}
}

void result_validate()
{
	for (int i = 0; i < Neuron; i++)
	{
#pragma omp parallel for num_threads(24)
		for (int j = 0; j < Image; j++)
		{
			float sum = 0;
			for (int k = 0; k < Neuron; k++)
				sum += W_T_global[i][k] * Y_T_global[0][k][j];
			sum += bias;
			R_test[i][j] = sum > 32 ? 32 : sum < 0 ? 0
												   : sum;
		}
	}

	for (int i = 0; i < Neuron; i++)
		for (int j = 0; j < Image; j++)
		{
			if (R_test[i][j] != Y_T_global[1][i][j])
			{
				printf("(%d, %d): R_test %f, PE %f\n", i, j, R_test[i][j], Y_T_global[1][i][j]);
				getchar();
			}
		}
}

void top_function(bool data_prepare)
{
}

int main()
{
	// perform W_T * Y_T
	string y_fname = "./dataset/sparse-images-1024.tsv";
	string w_fname = "./dataset/hash_neuron1024/n1024-l1.tsv";
	read_Y(y_fname);
	read_W(w_fname);

	PE(0, Y_T_global, W_row_index, 1);
	//
	PE(0, Y_T_global, W_row_index, 0);

	printf("PE done\n");
	result_validate();
}