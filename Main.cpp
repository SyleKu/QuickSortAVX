#include <iostream>
#include "QuickSort.h"
#include <random>
#include <chrono>

using namespace std;

int main()
{
	// initialize size
	const uint64_t size = 32 * 1024 * 1024;

	// initialize random data
	QuickSort qs;
	auto serialData = qs.createRandomData(size);
	vector<int> avxData(serialData);

	// Serial QuickSort function
	double serialTime = qs.serialQuickSort(serialData);
	if (!qs.proove(serialData))
	{
		cout << "serial sort did not sort..." << endl;
		exit(-1);
	}

	// QuickSort with AVX
	double avxTime = qs.avxQuickSort(avxData);
	if (!qs.proove(avxData))
	{
		cout << "AVX sort did not sort..." << endl;
		exit(-1);
	}

	// print time results
	cout << "sorted serial in " << serialTime / 1000 << "ms" << endl;
	cout << "sorted AVX in " << avxTime / 1000 << "ms" << endl;
	cout << "speedup is " << serialTime / avxTime << endl;
}