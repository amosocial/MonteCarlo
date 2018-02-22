#pragma once
#ifndef Header_H
#define Header_H

#include <iostream>
#include <set>
#include <list>
#include <cmath>
#include <queue>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <assert.h>
#include <ctime>
#include <chrono>

using namespace std;

float GetRand0_1();

class sortedVector
{
public:
	vector<int> keyList;
	map<int, float> mymap;
	float getValue(int index);
	int insert(int key, float value);
	int insert_d(int key, float value);
	int pushback(int key, float value);
	int getPosition(int key);
	int update(int key, float value);
	int getKeybyPosition(int position);

};

#endif

