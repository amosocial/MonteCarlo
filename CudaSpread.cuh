#pragma once
#ifndef CudaSpread_H
#define CudaSpread_H

#include <iostream>
#include <set>
#include <list>
#include <cmath>
#include <math.h>
#include <queue>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include "Network.h"
#include <thrust/device_vector.h>
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <curand.h>
#include <curand_kernel.h>
#include <ctime>





void CurandInit(curandState*,int, int);
__global__ void CurandSetup(curandState* states, const unsigned long seed);

class GpuVs
{
public:
	int* d_result = 0, *dev_result = 0;
	

	int* d_neighbor = 0,		*dev_neighbor = 0;;
	int* d_neighbor_size = 0,	*dev_neighbor_size = 0;
	int* d_neighbor_index = 0,	*dev_neighbor_index = 0;
	int* d_infos = 0,			*dev_infos = 0;
	int* d_newActive = 0,		*dev_newActive = 0;
	bool* d_state = 0,			*dev_state = 0;

	bool* dev_states = 0;
	int* dev_newActives = 0;
	int* dev_tnewActives = 0;

	int blocks;
	int threads;
	int times;

	GpuVs(int blocks, int threads, Network network, vector<int> newActive, bool state[]);
};

cudaError_t spreadWithCuda(vector<int> &result, Network &network, vector<int> &seedSet, int, int);
cudaError_t g_marginal(float &result, Network &network, int cseed, GpuVs gpuvs);

#endif