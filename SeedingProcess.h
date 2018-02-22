#pragma once
#ifndef SEEDINGPROCESS_H
#define SEEDINGPROCESS_H

#include "network.h"
#include "CudaSpread.cuh"
using namespace std;

class MonteTimes {
public:
	int blocks;
	int threads;

	MonteTimes(int b, int t);
};



class Pattern{
public:
	int period;
	int budget;
	Pattern(int p,int b);
	void show();

};

class SeedingProcess
{
public:
	static void multiGo(Network network, Pattern pattern, int times, MonteTimes monteTimes, double result[]);
	
private:
	static void singleGo(Network network, Pattern pattern, MonteTimes monteTimes, double c_result[]);
};

class DiffusionState {
public:
	bool* state;
	vector<int> newActive;

	DiffusionState(int vNum);
};

int Greedy(Network &network, DiffusionState &dstate, sortedVector &nodesVector, MonteTimes monteTimes);

float marginal(Network &network, DiffusionState &dstate, int seed, int monteTimes);

float marginalOnce(Network &network, DiffusionState &dstate, int seed);

//float g_marginalOnce(Network &network, DiffusionState &dstate, int seed, int blocks, int threads);



#endif