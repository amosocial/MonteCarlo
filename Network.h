#pragma once
#ifndef NETWORK_H
#define NETWORK_H

#include "Header.h"

using namespace std;

class Network
{
public:
	int vNum;
	int eNum;
	vector<vector<int>> neighbor;
	vector<vector<int>> re_neighbor;
	vector<int> inDegree;
	//read
	Network(const int vNum, const string path, const string type);
	void ShowRelation();
	float SpreadOnce(vector<int> seedSet);
	float GetProb(int u, int v);
};

#endif