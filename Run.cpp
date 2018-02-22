#include "cudaSpread.cuh"

#include "Header.h"

int tmain()
{
	string path = "wiki.txt";
	Network network(8300, path, "IC");
	//network.ShowRelation();
	vector<int> seedSet = vector<int>{ 30 };
	const int blocks = 400;
	const int threads = 1;
	vector<int> result(blocks*threads);

	cudaError_t cudaStatus = spreadWithCuda(result, network, seedSet, blocks, threads);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		getchar();
		return 1;
	}



	// cudaDeviceReset must be called before exiting in order for profiling and
	//tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		getchar();
		return 1;
	}
	getchar();
	return 0;
}