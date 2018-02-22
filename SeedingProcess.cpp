#include "Header.h"
#include "SeedingProcess.h"
using namespace std;

MonteTimes::MonteTimes(int b, int t) : blocks(b), threads(b) {
	return;
}



Pattern::Pattern(int p, int b) : period(p), budget(b) {
	return;
}

void Pattern::show() {
	cout << period << " " << budget << endl;
	return;
}



DiffusionState::DiffusionState(int n)
{
	state = new bool[n];
	for (size_t i = 0; i < n; i++)
	{
		state[i] = false;
	}
}

void SeedingProcess::multiGo(Network network, Pattern pattern, int times, MonteTimes monteTimes, double result[])
{
	cout << "multiGo is running " <<endl;
	for (size_t i = 0; i < network.vNum; i++)
	{
		result[i] = 0;
	}
	double *c_result=new double[network.vNum];

	for (size_t i = 0; i < times; i++)
	{
		singleGo(network, pattern, monteTimes, c_result);
		for (size_t j = 0; j < network.vNum; j++)
		{
			result[j] = result[j] + c_result[j];
		}
		cout << "wholetimes " << i << endl;
	}

	for (size_t i = 0; i < network.vNum; i++)
	{
		result[i] = result[i]/times;
	}
}

void SeedingProcess::singleGo(Network network, Pattern pattern, MonteTimes monteTimes, double c_result[])
{
	//cout << "singleGo is running " << endl;
	sortedVector nodesVector;

	for (int i = 0; i < network.vNum; i++)
	{
		nodesVector.pushback(i, network.vNum);
		//cout << "singleGo in 59 " << i <<endl;
	}
	
	
	DiffusionState dstate(network.vNum);
	int c_round = 0;
	int c_seed = 0;
	int aNum = 0;
	while (c_round < network.vNum)
	{
		
		if ( (c_round % pattern.period == 0 || dstate.newActive.size() == 0) && c_seed < pattern.budget )
		{
			int seed = Greedy(network, dstate, nodesVector, monteTimes);//???
			cout << "seed " << seed << endl;
			dstate.newActive.push_back(seed);
			dstate.state[seed] = true;
			aNum++;
			//it = mymap.find(seed);             // by iterator (b), leaves acdefghi.
			//mymap.erase(it);
			//mymap.insert(seed,0);
			nodesVector.update(seed,0);
			c_seed++;
		}

		//spread one round
		vector<int> tnewActive;
		for (size_t i = 0; i < dstate.newActive.size(); i++)
		{
			int seed = dstate.newActive[i];
			for (size_t j = 0; j < network.neighbor[seed].size(); j++)
			{
				int seede = network.neighbor[seed][j];
				float prob = network.GetProb(seed, seede);
				float rand = GetRand0_1();
				if (rand < prob && ! dstate.state[seede])
				{
					dstate.state[seede] = true;
					tnewActive.push_back(seede);
					aNum++;
				}
			}
		}
		//spread one round
		dstate.newActive.clear();
		for (size_t i = 0; i < tnewActive.size(); i++)
		{
			dstate.newActive.push_back(tnewActive[i]);
			//it = mymap.find(tnewActive[i]);             // by iterator (b), leaves acdefghi.
			//mymap.erase(it);
			//mymap.insert(tnewActive[i], 0);
			nodesVector.pushback(tnewActive[i], 0);
		}

		c_result[c_round] = aNum;
		c_round++;
		//cout << "c_round " << c_round << endl;

	}
}

int Greedy(Network &network, DiffusionState &dstate, sortedVector &nodesVector, MonteTimes monteTimes)
{
	cudaError_t  cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	GpuVs gpuvs(monteTimes.blocks, monteTimes.threads, network, dstate.newActive, dstate.state);


	//calculate current influence
	cout << "Greedy is running " << endl;
	float t_marginal = 0;
	int c_bound = network.vNum;

	bool *cstate = new bool[network.vNum];
	for (int i = 0; i < network.vNum; i++)
	{
		cstate[i] = false;
		//cout << "singleGo in 59 " << i <<endl;
	}

	while (c_bound > 0)
	{
		//int t_index = 0;
		int t_seed = nodesVector.getKeybyPosition(0);
		if (cstate[t_seed])
		{
			return t_seed;
		}
		else {
			cstate[t_seed] = true;
			
			//t_marginal = marginal(network, dstate, 3, monteTimes);
			 g_marginal(t_marginal, network, 3, gpuvs);
			 cout << t_marginal << endl;
			if (t_marginal > nodesVector.getValue(t_seed))
			{
				cout << "not submodular" << endl;
			}
			int c_bound = nodesVector.update(t_seed, t_marginal);
			//cout << t_marginal << " " << t_seed << " "<< c_bound<< " "<< double(end - begin) / CLOCKS_PER_SEC<<endl;
		}
		
	}
	delete[] cstate;
	return nodesVector.getKeybyPosition(0);
}

float marginal(Network &network, DiffusionState &dstate, int seed, int monteTimes)
{
	float result = 0;
	for (size_t i = 0; i < monteTimes; i++)
	{
		result = result + marginalOnce(network, dstate, seed);
		//cout << i << endl;
	}
	return result / monteTimes;
}


float marginalOnce(Network &network, DiffusionState &dstate, int seed)
{
	//vector<int> state = vector<int>(vNum, false);
	//vector<int> newActive = seedSet;
	float aNum = 0;
	vector<int> newActive= dstate.newActive;
	bool* state = new bool[network.vNum];
	for (size_t i = 0; i < network.vNum; i++)
	{
		state[i] = dstate.state[i];
	}
	vector<int> tActive;


	while (!newActive.empty())
	{
		for (size_t i = 0; i < newActive.size(); i++)
		{
			int seed = newActive[i];
			for (size_t j = 0; j < network.neighbor[seed].size(); j++)
			{
				int seede = network.neighbor[seed][j];
				//cout << seed << ' ' << neighbor[seed][j] << endl;
				float rand = GetRand0_1();
				float prob = network.GetProb(seed, seede);
				if (rand < prob && ! state[seede])
				{
					state[seede] = true;
					tActive.push_back(seede);
				}
			}
			//cout << tActive.size() << endl;
		}
		newActive = tActive;
		tActive.clear();
	}
	if (!state[seed])
	{
		
		
		newActive.clear();
		newActive.push_back(seed);
		aNum++;
		
		while (!newActive.empty())
		{
			
			
			for (size_t i = 0; i < newActive.size(); i++)
			{
				
				
				int cseed = newActive[i];
				for (size_t j = 0; j < network.neighbor[cseed].size(); j++)
				{
					
					
					
					//cout << network.neighbor[newActive[i]].size() << " ";
					int seede = network.neighbor[cseed][j];
					
					//cout << seed << ' ' << network.neighbor[seed][j] << endl;
					
					float rand = GetRand0_1();
					
					float prob = network.GetProb(cseed, seede);
					
					
					
					//float prob = 0.1;
					//cout << rand << ' ' << prob << " " << dstate.state[seede] << endl;
					
					if (rand < prob && !state[seede])
					{
						
						state[seede] = true;
						std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
						tActive.push_back(seede);
						std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
						std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() << std::endl;
						aNum++;
						
					}
					
					
				}
				
				//cout << tActive.size() << endl;
			}
			
			newActive = tActive;
			tActive.clear();
			
			
		}
		

		
	}
	delete[] state;
	//cout << aNum << endl;
	return aNum;
}


