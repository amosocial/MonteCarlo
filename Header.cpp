#include "Header.h"


using namespace std;


float GetRand0_1()
{
	return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
}

float sortedVector::getValue(int key)
{
	return mymap[key];
}

int sortedVector::insert(int key, float value)
{
	mymap[key] = value;
	for (size_t i = 0; i < keyList.size(); i++)
	{
		if (mymap[keyList[i]] <= value)
		{
			keyList.insert(keyList.begin() + i, key);
			return i;
		}
	}
	keyList.push_back(key);
	return keyList.size() - 1;
}

int sortedVector::pushback(int key, float value)
{
	mymap[key] = value;
	keyList.push_back(key);
	return keyList.size() - 1;
}



int sortedVector::insert_d(int key, float value)
{
	mymap[key] = value;
	for (size_t i = keyList.size(); i > -1; i--)
	{
		if (mymap[keyList[i]] >= value)
		{
			keyList.insert(keyList.begin() + i, key);
			return i;
		}
	}
	keyList.push_back(key);
	return keyList.size() - 1;
}

int sortedVector::getPosition(int key)
{
	vector<int>::iterator it;
	it = find(keyList.begin(), keyList.end(), key);
	if (it != keyList.end())
	{
		return it - keyList.begin();
	}
	else
	{
		assert(it != keyList.end() && "index is not found while getPosition");
		return -1;
	}
}

int sortedVector::getKeybyPosition(int position)
{
	if (position >= keyList.size())
	{
		cout << "getKeybyPosition our of bound";
		return -1;
	}
	else
	{
		
		return keyList[position];
	}
}
int sortedVector::update(int key, float value)
{
	mymap[key] = value;
	keyList.erase(keyList.begin()+getPosition(key));
	return insert(key, value);

}
