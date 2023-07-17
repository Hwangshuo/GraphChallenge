
#include "cmdlineparser.h"
#include "gc.h"
#include "xrt/xrt_kernel.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_bo.h"
#include <fstream>
#include <string>
#include <sstream>
#include <ap_int.h>
using namespace std;

float Y_T_global[Mem_channel][2][Neuron][YT_col_channel] = {0};
float W_T_global[Layer][Neuron][Neuron] = {0};
// ap_uint<32> W_row_index[Layer][Neuron / 16 * 2] = {0};
int W_row_index[Layer][Neuron / 16 * 32] = {0};
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
		Y_T_global[i / YT_col_channel][0][j][i % YT_col_channel] = v;
		// Y_global[i / Y_row_channel][0][i % Y_row_channel][j] = v;
	}
}

void read_W(int layer, char *fname)
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
		W_T_global[layer][j][i] = v;
	}

	int total = 0;
	int index_group = 0;
	// 	for (i = 0; i < Neuron; i += 16)
	// 	{
	// 		int nnz = 0;

	// 		for (j = 0; j < Neuron; j++)
	// 		{

	// 			if (W_T_global[layer][i][j] != 0)
	// 			{
	// 				W_row_index[layer][index_group].range(31 + 32 * (nnz % Index_per_group), 32 * (nnz % Index_per_group)) = j;
	// 				total++;
	// 				nnz++;
	// 			}
	// 			if (total % Index_per_group == 0)
	// 				index_group++;
	// 			if (nnz == 32)
	// 				break;
	// 		}
	// 	}
	for (i = 0; i < Neuron; i += 16)
	{
		int nnz = 0;

		for (j = 0; j < Neuron; j++)
		{

			if (W_T_global[layer][i][j] != 0)
			{
				W_row_index[layer][total] = j;
				total++;
				nnz++;
			}
			if (nnz == 32)
				break;
		}
	}
}

int main(int argc, char **argv)
{

	// Command Line Parser
	sda::utils::CmdLineParser parser;

	// Switches
	//**************//"<Full Arg>",  "<Short Arg>", "<Description>", "<Default>"
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

	auto krnl = xrt::kernel(device, uuid, "SpdnnKernel");

	string y_fname = "../../../GC23/dataset/hash-sparse-images-1024.tsv";
	read_Y(y_fname);
	char w_fname[256];
	for (int i = 0; i < Layer; i++)
	{
		sprintf(w_fname, "../../../GC23/dataset/hash_neuron%d/n%d-l%d.tsv", Neuron, Neuron, i + 1);
		read_W(i, w_fname);
	}
	printf("Read done\n");

	//	string xclbinFile(argv[1]);
	//	xrt::device fpga = xrt::device(0);
	//	xrt::uuid xclbin = fpga.load_xclbin(xclbinFile);
	//
	//	xrt::kernel krnl = xrt::kernel(fpga, uuid, "SpdnnKernel");

	xrt::run exec = xrt::run(krnl);

	xrt::bo W_row_index_bo;
	xrt::bo Y_T_global_bos[Mem_channel];

	W_row_index_bo = xrt::bo(device, sizeof(W_row_index), krnl.group_id(0));
	W_row_index_bo.write(W_row_index);
	W_row_index_bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
	exec.set_arg(0, W_row_index_bo);

	for (int i = 0; i < Mem_channel; i++)
	{
		Y_T_global_bos[i] = xrt::bo(device, sizeof(Y_T_global[i]), krnl.group_id(i + 1));
		Y_T_global_bos[i].write(Y_T_global[i]);
		Y_T_global_bos[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
		exec.set_arg(i + 1, Y_T_global_bos[i]);
	}
	std::chrono::duration<double> kernel_time(0);
	auto kernel_start = std::chrono::high_resolution_clock::now();
	exec.start();
	exec.wait();
	auto kernel_end = std::chrono::high_resolution_clock::now();
	kernel_time = std::chrono::duration<double>(kernel_end - kernel_start);
	std::cout << kernel_time.count() << std::endl;
}
