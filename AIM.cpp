#include "Header.h"
#include "Network.h"
#include "SeedingProcess.h"

using namespace std;

int main()
{
	string path = "wiki.txt";
	Network network(8300, path, "IC");
	Pattern pattern(1, 1);
	pattern.show();
	SeedingProcess sp;
	int times = 1;
	MonteTimes monteTimes(10, 1);
	//int monteTimes = 1000;
	double* result = new double[network.vNum];
	sp.multiGo(network, pattern, times, monteTimes, result);

	for (size_t i = 0; i < network.vNum; i++)
	{
		cout << result[i] << endl;
		if (result[i] - result[i - 1] < 0.0001)
			break;
	}
	//network.ShowRelation();
	/*
	DiffusionState dstate(network.vNum);
	for (size_t i = 0; i < network.vNum; i++)
	{
		if(dstate.state[i])
			cout << dstate.state[i] << endl;
	}

	cout<<marginal(network, dstate, 30, monteTimes);

	for (size_t i = 0; i < network.vNum; i++)
	{
		if (dstate.state[i])
			cout << dstate.state[i] << endl;
	}*/
	getchar();
}