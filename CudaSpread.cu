#include "cudaSpread.cuh"

#include "Header.h"
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

using namespace std;


inline void gpuAssert(cudaError_t code, char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

GpuVs::GpuVs(int b, int t, Network network, vector<int> newActive, bool state[]) : blocks(b), threads(t)
{

	times = b*t;
	d_result = new int[times];


	d_neighbor_size = new int[network.vNum];
	d_neighbor_index = new int[network.vNum];
	d_neighbor = new int[network.eNum];
	d_infos = new int[3];
	d_newActive = new int[newActive.size()];
	d_state = new bool[network.vNum];

	dev_states = new bool[times*network.vNum];
	dev_newActives = new int[times*network.vNum];
	dev_tnewActives = new int[times*network.vNum];

	d_infos[0] = network.vNum; //info_0: number of nodes
	d_infos[1] = newActive.size(); //info_1: num of seed nodes
	d_infos[2] = times; //info_2: times of spread

	//copy current active nodes
	for (size_t i = 0; i < newActive.size(); i++)
	{
		d_newActive[i] = newActive[i];
		//cout << "d_seedSet[i] " << d_seedSet[i] << endl;
	}
	//copy state
	for (size_t i = 0; i < network.vNum; i++)
	{
		d_state[i] = state[i];
	}
	//copy relations
	int count = 0;
	for (size_t i = 0; i < network.vNum; i++)
	{
		//d_neighbor[i] = new int[network.neighbor[i].size()];

		d_neighbor_size[i] = network.neighbor[i].size();
		//cout << d_neighbor_size[i] << endl;
		d_neighbor_index[i] = count;
		for (size_t j = 0; j < d_neighbor_size[i]; j++)
		{
			if (count > network.eNum)
			{
				cout << "count > network.eNum" << endl;
			}
			d_neighbor[count] = network.neighbor[i][j];
			count++;
		}
	}
	//initiate shared records: d_states d_newActives d_tnewActives
	
	cudaError_t cudaStatus;

	//----------------------------
	//cudaMallocManaged(&d_seedSet, seedSet.size() * sizeof(int));
	//---------------------------------
	// Allocate GPU buffers for four variables  .
	
	
	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_neighbor, network.eNum * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_neighbor, d_neighbor, network.eNum * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_neighbor cudaMemcpy failed!");
	}

	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_neighbor_size, network.vNum * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_neighbor_size, d_neighbor_size, network.vNum * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_neighbor_size cudaMemcpy failed!");
	}

	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_neighbor_index, network.vNum * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	cudaStatus = cudaMemcpy(dev_neighbor_index, d_neighbor_index, network.vNum * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_neighbor_index cudaMemcpy failed!");
	}

	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_infos, 3 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_infos, d_infos, 3 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
	}

	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_state, 3 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_state, d_state, 3 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
	}
	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_newActive, 3 * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMemcpy(dev_newActive, d_newActive, 3 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
	}



	//-----------------------------------------------
	cudaStatus = cudaMalloc((void**)&dev_result, times * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_states, network.vNum*times * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_newActives, network.vNum*times * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_tnewActives, network.vNum*times * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
	}
	
}






__global__ void initCurand(curandState *state, unsigned long seed) {
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	curand_init(seed+ idx, 0, 0, &state[idx]);
}

__global__ void marginalKernel(int cseed, int* result, int* neighbor, int* neighbor_size, int* neighbor_index, int* infos, int* newActive, bool* state, bool* states, int* newActives, int* tnewActives, curandState* rstates)
{
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int sindex = id*infos[0];
	int newActive_size = 0;
	int tnewActive_size = 0;

	int aNum = 0;
	for (size_t i = 0; i < infos[0]; i++)
	{
		states[id+i] = false;
		newActives[id+i] = 0;
		tnewActives[id+i] = 0;
	}

	for (size_t i = 0; i <infos[1]; i++)
	{
		states[sindex + newActive[i]] = true;
		newActives[sindex + i] = newActive[i];
		newActive_size++;
		//aNum++;
	}


	while (newActive_size>0)
	{
		for (size_t i = 0; i < newActive_size; i++)
		{
			int seed = newActives[sindex + i];
			//printf("seed %d and size %d \n", seed, d_neighbor_size[seed]);
			for (size_t j = 0; j < neighbor_size[seed]; j++)
			{
				int j_index = neighbor_index[seed] + j;
				int seede = neighbor[j_index];
				//printf("activate %d and %d \n", seede, seede);
				//cout << seed << ' ' << neighbor[seed][j] << endl;


				//curand_init((unsigned long long)clock() , id, 0, &s);
				float rand = curand_uniform(&rstates[id]);
				//float rand = 0.01;
				float prob = 0.1;//???
								 //printf("rand %f \n", rand);
				if (rand < prob && !states[sindex + seede])
				{
					states[sindex + seede] = true;
					tnewActives[sindex + tnewActive_size] = seede;
					tnewActive_size++;
					//aNum++;
				}
			}
			//cout << tActive.size() << endl;
		}
		//copy new active nodes
		for (size_t i = 0; i < tnewActive_size; i++)
		{
			newActives[sindex + i] = tnewActives[sindex + i];
		}
		newActive_size = tnewActive_size;
		tnewActive_size = 0;
	}

	if (!states[sindex + cseed])
	{
		newActives[sindex + 0] = cseed;
		states[sindex + cseed] = true;
		newActive_size++;
		aNum++;

		while (newActive_size>0)
		{
			for (size_t i = 0; i < newActive_size; i++)
			{
				int seed = newActives[sindex + i];
				for (size_t j = 0; j < neighbor_size[seed]; j++)
				{
					int j_index = neighbor_index[seed] + j;
					int seede = neighbor[j_index];
					float rand = curand_uniform(&rstates[id]);
					//float rand = 0.01;
					float prob = 0.1;//???
									 //printf("rand %f \n", rand);
					if (rand < prob && !states[sindex + seede])
					{
						states[sindex + seede] = true;
						tnewActives[sindex + tnewActive_size] = seede;
						tnewActive_size++;
						//aNum++;
					}
				}
				//cout << tActive.size() << endl;
			}
			//copy new active nodes
			for (size_t i = 0; i < tnewActive_size; i++)
			{
				newActives[sindex + i] = tnewActives[sindex + i];
			}
			newActive_size = tnewActive_size;
			tnewActive_size = 0;
		}
	}
	//printf("Hello from thread %d %d----------------------- %d \n", blockIdx.x, threadIdx.x, aNum);

	result[id] = aNum;
}


__global__ void addKernel(int* d_result, int* d_neighbor, int* d_neighbor_size, int* d_neighbor_index, int* d_seedSet, int* infos, bool* states, int* newActives, int* tnewActives, curandState* rstates)
{
	
	int id = blockDim.x*blockIdx.x + threadIdx.x;
	int sindex = id*infos[0];
	//printf("Hello from thread %d and %d \n", blockIdx.x, threadIdx.x);
	//bool* state = new bool[infos[0]];
	
	//int* newActive= new int[infos[0]];
	int newActive_size = 0;
	
	//int* tnewActive = new int[infos[0]];
	int tnewActive_size = 0;

	int aNum = 0;
	/*
	for (size_t i = 0; i <infos[0]; i++)
	{
		state[i] = false;
		newActive[i] = 0;
		tnewActive[i] = 0;
	}*/
	
	for (size_t i = 0; i <infos[1]; i++)
	{
		states[sindex+d_seedSet[i]] = true;
		newActives[sindex+i]= d_seedSet[i];
		newActive_size++;
		aNum++;
	}
	
	//curandState s;
	//curand_init((unsigned long long)clock() +id, 0, 0, &s);
	//curandState localState = rstates[id];
	//curand_init(id, 0, 0, &localState);
	//for (size_t i = 0; i < 10; i++)
	//{
	//	float rand = curand_uniform(&rstates[id]);
	//	printf("rand %f \n", rand);
	//}
	
	while (newActive_size>0)
	{
		for (size_t i = 0; i < newActive_size; i++)
		{
			int seed = newActives[sindex+i];
			//printf("seed %d and size %d \n", seed, d_neighbor_size[seed]);
			for (size_t j = 0; j < d_neighbor_size[seed]; j++)
			{
				int j_index = d_neighbor_index[seed] + j;
				int seede = d_neighbor[j_index];
				//printf("activate %d and %d \n", seede, seede);
				//cout << seed << ' ' << neighbor[seed][j] << endl;

				
				//curand_init((unsigned long long)clock() , id, 0, &s);
				float rand = curand_uniform(&rstates[id]);
				//float rand = 0.01;
				float prob = 0.1;//???
				//printf("rand %f \n", rand);
				if (rand < prob && !states[sindex+seede])
				{
					states[sindex+seede] = true;
					tnewActives[sindex+tnewActive_size]= seede;
					tnewActive_size++;
					aNum++;
				}
			}
			//cout << tActive.size() << endl;
		}
		//copy new active nodes
		for (size_t i = 0; i < tnewActive_size; i++)
		{
			newActives[sindex+i] = tnewActives[sindex+i];
		}
		newActive_size= tnewActive_size;
		tnewActive_size=0;
	}
	//printf("Hello from thread %d %d----------------------- %d \n", blockIdx.x, threadIdx.x, aNum);
	
	d_result[id] = aNum;

}


cudaError_t g_marginal(float &result, Network &network, int cseed, GpuVs gpuvs)
{
	cout << "g_marginal is running " << endl;
	cudaError_t cudaStatus;

	curandState* rstates;	cudaMalloc((void**)&rstates, gpuvs.times * sizeof(curandState));
	initCurand << <gpuvs.blocks, gpuvs.threads >> >(rstates, 1);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	marginalKernel << < gpuvs.blocks, gpuvs.threads >> > (cseed,
		gpuvs.dev_result,
		gpuvs.dev_neighbor, 
		gpuvs.dev_neighbor_size,
		gpuvs.dev_neighbor_index,
		gpuvs.dev_infos,
		gpuvs.dev_newActive,
		gpuvs.dev_state,
		gpuvs.dev_states,
		gpuvs.dev_newActives,
		gpuvs.dev_tnewActives,
		rstates);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "marginalKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		getchar();
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		getchar();

	}

	// Copy output vector from GPU buffer to host memory.

	cudaStatus = cudaMemcpy(gpuvs.d_result, gpuvs.dev_result, gpuvs.times * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
	}
	//cout << times << endl;
	for (size_t i = 0; i < gpuvs.times; i++)
	{
		//cout << i << " " << gpuvs.d_result[i] << endl;
		result = result + gpuvs.d_result[i];
	}
	// process required information
	
	result = result / gpuvs.times;
	return cudaStatus;

}

// Helper function for using CUDA to add vectors in parallel.

cudaError_t spreadWithCuda(vector<int> &result, Network &network, vector<int> &seedSet, int blocks, int threads)
{
	
	

	//blocks and threads
	int times = blocks*threads;

	//setup crandstates
	curandState* rstates;	gpuErrchk(cudaMalloc((void**)&rstates, times * sizeof(curandState)));

	float *d_a;             gpuErrchk(cudaMalloc((void**)&d_a, times * sizeof(float)));

	initCurand << <blocks , threads >> >(rstates, 1);
	gpuErrchk(cudaPeekAtLastError());
	gpuErrchk(cudaDeviceSynchronize());

	//testrand1 << <blocks, threads >> >(rstates, d_a);
	//gpuErrchk(cudaPeekAtLastError());
	//gpuErrchk(cudaDeviceSynchronize());

	//getchar();
	//network.ShowRelation();
	int* dev_result = 0;
	int* dev_seedSet = 0;

	int* dev_neighbor = 0;
	int* dev_neighbor_size = 0;
	int* dev_neighbor_index = 0;
	int* dev_infos= 0;

	bool* dev_states = 0;
	int* dev_newActives = 0;
	int* dev_tnewActives = 0;

	cudaError_t cudaStatus;

	
	// process required information
	int* d_result= new int[times];
	int* d_seedSet = new int[seedSet.size()];

	int* d_neighbor_size = new int[network.vNum];
	int* d_neighbor_index = new int[network.vNum];
	int* d_neighbor = new int[network.eNum];
	int* d_infos = new int[3];

	bool* d_states = new bool[times*network.vNum];
	int* d_newActives = new int[times*network.vNum];
	int* d_tnewActives = new int[times*network.vNum];

	d_infos[0] = network.vNum; //info_0: number of nodes
	d_infos[1] = seedSet.size(); //info_1: num of seed nodes
	d_infos[2] = times; //info_2: times of spread


	for (size_t i = 0; i < seedSet.size(); i++)
	{
		d_seedSet[i] = seedSet[i];
		cout << "d_seedSet[i] " << d_seedSet[i] << endl;
	}
	 
	int count = 0;
	for (size_t i = 0; i < network.vNum; i++)
	{
		//d_neighbor[i] = new int[network.neighbor[i].size()];
		
		d_neighbor_size[i] =  network.neighbor[i].size();
		//cout << d_neighbor_size[i] << endl;
		d_neighbor_index[i] = count;
		for (size_t j = 0; j < d_neighbor_size[i]; j++)
		{
			if (count > network.eNum)
			{
				cout << "count > network.eNum" << endl;
			}
			d_neighbor[count] = network.neighbor[i][j];
			count++;
		}
	}

	for (size_t i = 0; i < times*network.vNum; i++)
	{
		d_states[i] = false;
		d_newActives[i] = 0;
		d_tnewActives[i] = 0;
	}

	

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	//----------------------------
	//cudaMallocManaged(&d_seedSet, seedSet.size() * sizeof(int));
	//---------------------------------
	// Allocate GPU buffers for four variables  .
	cudaStatus = cudaMalloc((void**)&dev_result, times*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_seedSet, seedSet.size() * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_neighbor, network.eNum*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_neighbor_size, network.vNum * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_neighbor_index, network.vNum * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_infos, 3*sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_states, network.vNum*times * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_newActives, network.vNum*times * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	cudaStatus = cudaMalloc((void**)&dev_tnewActives, network.vNum*times * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}


	// Copy input vectors from host memory to GPU buffers.
	
	cudaStatus = cudaMemcpy(dev_seedSet, d_seedSet, seedSet.size() * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_seedSet cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_neighbor, d_neighbor, network.eNum * sizeof(int) , cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_neighbor cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_neighbor_size, d_neighbor_size, network.vNum * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_neighbor_size cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_neighbor_index, d_neighbor_index, network.vNum * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_neighbor_index cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_infos, d_infos, 3*sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_states, d_states, network.vNum* times * sizeof(bool), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_newActives, d_newActives, network.vNum* times * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(dev_tnewActives, d_tnewActives, network.vNum* times * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "dev_infos cudaMemcpy failed!");
		goto Error;
	}


	//allocate more memory
	/*
	int states_size = network.vNum * sizeof(bool);
	int active_size = network.vNum * sizeof(int);
	int tactive_size = network.vNum * sizeof(int);
	int extra_size = network.vNum * sizeof(int);
	cudaStatus = cudaDeviceSetLimit(cudaLimitMallocHeapSize, states_size+ active_size+ tactive_size+extra_size);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSetLimit failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}*/
	//cudaMalloc((void**)&rstates, blocks * threads * sizeof(curandState));
	// Launch a kernel on the GPU with one thread for each element.
	addKernel <<< blocks, threads >>> (dev_result, dev_neighbor, dev_neighbor_size, dev_neighbor_index, dev_seedSet, dev_infos, dev_states, dev_newActives, dev_tnewActives, rstates);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	
	cudaStatus = cudaMemcpy(d_result, dev_result, times*sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}
	//cout << times << endl;
	for (size_t i = 0; i < times; i++)
	{
		cout << i<<" "<<d_result[i] << endl;
	}

Error:
	cudaFree(dev_result);
	cudaFree(dev_neighbor);
	cudaFree(dev_seedSet);
	cudaFree(dev_neighbor_size);
	cudaFree(dev_neighbor_index);
	cudaFree(dev_infos);
	cudaFree(dev_states);
	cudaFree(dev_newActives);
	cudaFree(dev_tnewActives);

	return cudaStatus;
}

