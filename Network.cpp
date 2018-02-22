#include "Network.h"
#include "Header.h"
using namespace std;


Network::Network(const int n, string path, string type) : vNum(n)
{
	//initialize neighbor
	neighbor.resize(vNum);
	eNum = 0;
	//read relations
	ifstream datafile(path);

	int u, v;
	while (datafile >> u >> v)
	{
		// process pair (u,v)
		neighbor[u].push_back(v);
		eNum++;
	}

}

float Network::SpreadOnce(vector<int> seedSet)
{
	vector<int> state = vector<int>(vNum, false);
	vector<int> newActive=seedSet;
	double aNum = 0;
	for (size_t i = 0; i < seedSet.size(); i++)
	{
		state[seedSet[i]] = true;
		newActive.push_back(seedSet[i]);
		aNum++;
	}
	vector<int> tActive;

	
	while (!newActive.empty())
	{
		for (size_t i = 0; i < newActive.size(); i++)
		{
			int seed = newActive[i];
			for (size_t j = 0; j < neighbor[seed].size(); j++)
			{
				int seede = neighbor[seed][j];
				//cout << seed << ' ' << neighbor[seed][j] << endl;
				float rand=GetRand0_1();
				float prob = GetProb(seed,seede);
				if (rand < prob && !state[seede])
				{
					state[seede] = true;
					tActive.push_back(seede);
					aNum++;
				}
			}
			//cout << tActive.size() << endl;
		}
		newActive = tActive;
		tActive.clear();
	}

	return aNum;
}

float Network::GetProb(int u, int v)
{
	return 0.1;
}

void Network::ShowRelation()
{
	for (size_t i = 0; i < neighbor.size(); i++)
	{
		for (size_t j = 0; j < neighbor[i].size(); j++)
			cout << i << ' ' << neighbor[i][j] << endl;
	}
}
/*
void testmain()
{
	string path = "data\\wiki.txt";
	Network network(8300, path, "IC");
	//network.ShowRelation();
	
	
	getchar();
}*/
/*
void tmain()
{
	map<int, float> mymap;
	mymap[1] = 10;
	mymap[2] = 20;
	mymap[3] = 30;
	for (map<int, float>::iterator it = mymap.begin(); it != mymap.end(); ++it) {
		cout << it->first << "\n";
	}
	mymap[2] = 60;

	for (map<int, float>::iterator it = mymap.begin(); it != mymap.end(); ++it) {
		cout << it->first <<" "<< it->second<< "\n";
	}
	getchar();
	return;
}*/