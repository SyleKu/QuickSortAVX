#pragma once
#include <vector>
#include <immintrin.h>
#include <cmath>

// CUT_OFF / THRESHOLD for the insertion Sort function
constexpr auto CUT_OFF = 40;

class QuickSort
{
	// Functions for the serial QuickSort
	static void qsPartition(std::vector<int>& data, uint64_t beg, uint64_t end);
	static void qsSort(std::vector<int>& data, uint64_t beg, uint64_t end);
	static void insertionSort(std::vector<int>& data, uint64_t beg, uint64_t end);
	static void qsSwap(std::vector<int>& data, uint64_t i, uint64_t j);

	void serialQSort(std::vector<int>& data);

	// Functions for the AVX QuickSort
	static int* avxVectorizedPartitionInPlace(int* left, int* right);
	static void avxSwapIfGreater(int* left, int* right);
	static void avxSwap(int* left, int* right);
	static void avxPartitionBlock(int* dataPtr, __m256i pivots, int*& writeLeft, int*& writeRight);
	static void avxSort(int* left, int* right);
	static void avxInsertionSort(int* left, int* right);

	void avxQSort(std::vector<int>& data);

	// Function for measuring and running the provided function
	double measuredSort(std::vector<int>& data, void(QuickSort::* sortFunc)(std::vector<int>& data));

public:
	QuickSort(); // declare default constructor
	double serialQuickSort(std::vector<int>& data);
	double avxQuickSort(std::vector<int>& data);
	bool proove(std::vector<int>& data);
	std::vector<int> createRandomData(uint64_t size);
};