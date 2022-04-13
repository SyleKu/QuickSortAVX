#include "QuickSort.h"
#include "iostream"
#include <immintrin.h>
#include <chrono>
#include <cassert>
#include <random>
#include <intrin.h>

using namespace std;

// pointer to the helper array
int* _temp;

// start pointer position of the helper array
int* _TEMPSTART;

// end pointer position of the helper array
int* _TEMPEND;

// N_OFFSET is the amount of how many registers can do calculations at the same time in AVX
// Since we are working anyway with AVX2, we always have 256 bit registers. We also work here with int, that's why devide it here by 32 bit!
//
// In C#: "var N = Vector256<T>.Count"
// Vector256<T>.Count is dependent on the provided CPU architecture.
// Let's say we use Vector256<float>.Count. Float is 64 bit, and we have a CPU that has AVX2 with 256 bit. '.Count()' function will
// now give 4, since 64 fits 4 times in 256(!). In older architecture there might be possibilty that it only fits for example 2 times.
// With this "N" offset, we can always jump to the next array entry by this offset value!
const int N_OFFSET = 256 / 32;

// Constructor for initialzing the default values
QuickSort::QuickSort()
{
	// 96 bytes = 3 x 8 integers!
	// 3 times => 2 x 8 integers free space for each side of the data array, 
	//            1 x 8 integers for handling remaining values in data array
	_temp = new int[3 * N_OFFSET];
	_TEMPSTART = _temp;
	_TEMPEND = _temp + 3 * N_OFFSET;
}

// swap function
void QuickSort::qsSwap(std::vector<int>& data, uint64_t i, uint64_t j)
{
	uint64_t temp = data[i];
	data[i] = data[j];
	data[j] = temp;
}

// insertion function for the CUT_OFF / THRESHOLD part, same code as in algd1
void QuickSort::insertionSort(std::vector<int>& data, uint64_t beg, uint64_t end)
{
	for (size_t i = beg + 1; i <= end; i++)
	{
		size_t j = i;
		while (j > beg && data[j] < data[j - 1])
		{
			qsSwap(data, j, j - 1);
			j--;
		}
	}
}


// Partition function, same code as in algd1
void QuickSort::qsPartition(std::vector<int>& data, uint64_t beg, uint64_t end)
{
	uint64_t i = beg;
	uint64_t j = end;
	uint64_t pivot = (i + j) >> 1;

	while (i <= j)
	{
		while (data[i] < data[pivot])
		{
			i++;
		}

		while (data[pivot] < data[j])
		{
			j--;
		}

		if (i <= j)
		{
			if (i == pivot)
			{
				pivot = j;
			}
			else if (j == pivot)
			{
				pivot = i;
			}
			qsSwap(data, i, j);
			i++;
			j--;
		}
	}
	qsSort(data, beg, j);
	qsSort(data, i, end);
}

// QuickSort recursive function with CUT_OFF / THRESHOLD
void QuickSort::qsSort(std::vector<int>& data, uint64_t beg, uint64_t end)
{
	if (end - beg >= CUT_OFF)
	{
		qsPartition(data, beg, end);
	}
	else
	{
		insertionSort(data, beg, end);
	}
}

// Pre-Generated Permutation table
// 8 * 4 byte = 32 bytes per element
static const int PermTablePtr[] = {
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b00000000 (0)
	1, 2, 3, 4, 5, 6, 7, 0,  // 0b00000001 (1)
	0, 2, 3, 4, 5, 6, 7, 1,  // 0b00000010 (2)
	2, 3, 4, 5, 6, 7, 0, 1,  // 0b00000011 (3)
	0, 1, 3, 4, 5, 6, 7, 2,  // 0b00000100 (4)
	1, 3, 4, 5, 6, 7, 0, 2,  // 0b00000101 (5)
	0, 3, 4, 5, 6, 7, 1, 2,  // 0b00000110 (6)
	3, 4, 5, 6, 7, 0, 1, 2,  // 0b00000111 (7)
	0, 1, 2, 4, 5, 6, 7, 3,  // 0b00001000 (8)
	1, 2, 4, 5, 6, 7, 0, 3,  // 0b00001001 (9)
	0, 2, 4, 5, 6, 7, 1, 3,  // 0b00001010 (10)
	2, 4, 5, 6, 7, 0, 1, 3,  // 0b00001011 (11)
	0, 1, 4, 5, 6, 7, 2, 3,  // 0b00001100 (12)
	1, 4, 5, 6, 7, 0, 2, 3,  // 0b00001101 (13)
	0, 4, 5, 6, 7, 1, 2, 3,  // 0b00001110 (14)
	4, 5, 6, 7, 0, 1, 2, 3,  // 0b00001111 (15)
	0, 1, 2, 3, 5, 6, 7, 4,  // 0b00010000 (16)
	1, 2, 3, 5, 6, 7, 0, 4,  // 0b00010001 (17)
	0, 2, 3, 5, 6, 7, 1, 4,  // 0b00010010 (18)
	2, 3, 5, 6, 7, 0, 1, 4,  // 0b00010011 (19)
	0, 1, 3, 5, 6, 7, 2, 4,  // 0b00010100 (20)
	1, 3, 5, 6, 7, 0, 2, 4,  // 0b00010101 (21)
	0, 3, 5, 6, 7, 1, 2, 4,  // 0b00010110 (22)
	3, 5, 6, 7, 0, 1, 2, 4,  // 0b00010111 (23)
	0, 1, 2, 5, 6, 7, 3, 4,  // 0b00011000 (24)
	1, 2, 5, 6, 7, 0, 3, 4,  // 0b00011001 (25)
	0, 2, 5, 6, 7, 1, 3, 4,  // 0b00011010 (26)
	2, 5, 6, 7, 0, 1, 3, 4,  // 0b00011011 (27)
	0, 1, 5, 6, 7, 2, 3, 4,  // 0b00011100 (28)
	1, 5, 6, 7, 0, 2, 3, 4,  // 0b00011101 (29)
	0, 5, 6, 7, 1, 2, 3, 4,  // 0b00011110 (30)
	5, 6, 7, 0, 1, 2, 3, 4,  // 0b00011111 (31)
	0, 1, 2, 3, 4, 6, 7, 5,  // 0b00100000 (32)
	1, 2, 3, 4, 6, 7, 0, 5,  // 0b00100001 (33)
	0, 2, 3, 4, 6, 7, 1, 5,  // 0b00100010 (34)
	2, 3, 4, 6, 7, 0, 1, 5,  // 0b00100011 (35)
	0, 1, 3, 4, 6, 7, 2, 5,  // 0b00100100 (36)
	1, 3, 4, 6, 7, 0, 2, 5,  // 0b00100101 (37)
	0, 3, 4, 6, 7, 1, 2, 5,  // 0b00100110 (38)
	3, 4, 6, 7, 0, 1, 2, 5,  // 0b00100111 (39)
	0, 1, 2, 4, 6, 7, 3, 5,  // 0b00101000 (40)
	1, 2, 4, 6, 7, 0, 3, 5,  // 0b00101001 (41)
	0, 2, 4, 6, 7, 1, 3, 5,  // 0b00101010 (42)
	2, 4, 6, 7, 0, 1, 3, 5,  // 0b00101011 (43)
	0, 1, 4, 6, 7, 2, 3, 5,  // 0b00101100 (44)
	1, 4, 6, 7, 0, 2, 3, 5,  // 0b00101101 (45)
	0, 4, 6, 7, 1, 2, 3, 5,  // 0b00101110 (46)
	4, 6, 7, 0, 1, 2, 3, 5,  // 0b00101111 (47)
	0, 1, 2, 3, 6, 7, 4, 5,  // 0b00110000 (48)
	1, 2, 3, 6, 7, 0, 4, 5,  // 0b00110001 (49)
	0, 2, 3, 6, 7, 1, 4, 5,  // 0b00110010 (50)
	2, 3, 6, 7, 0, 1, 4, 5,  // 0b00110011 (51)
	0, 1, 3, 6, 7, 2, 4, 5,  // 0b00110100 (52)
	1, 3, 6, 7, 0, 2, 4, 5,  // 0b00110101 (53)
	0, 3, 6, 7, 1, 2, 4, 5,  // 0b00110110 (54)
	3, 6, 7, 0, 1, 2, 4, 5,  // 0b00110111 (55)
	0, 1, 2, 6, 7, 3, 4, 5,  // 0b00111000 (56)
	1, 2, 6, 7, 0, 3, 4, 5,  // 0b00111001 (57)
	0, 2, 6, 7, 1, 3, 4, 5,  // 0b00111010 (58)
	2, 6, 7, 0, 1, 3, 4, 5,  // 0b00111011 (59)
	0, 1, 6, 7, 2, 3, 4, 5,  // 0b00111100 (60)
	1, 6, 7, 0, 2, 3, 4, 5,  // 0b00111101 (61)
	0, 6, 7, 1, 2, 3, 4, 5,  // 0b00111110 (62)
	6, 7, 0, 1, 2, 3, 4, 5,  // 0b00111111 (63)
	0, 1, 2, 3, 4, 5, 7, 6,  // 0b01000000 (64)
	1, 2, 3, 4, 5, 7, 0, 6,  // 0b01000001 (65)
	0, 2, 3, 4, 5, 7, 1, 6,  // 0b01000010 (66)
	2, 3, 4, 5, 7, 0, 1, 6,  // 0b01000011 (67)
	0, 1, 3, 4, 5, 7, 2, 6,  // 0b01000100 (68)
	1, 3, 4, 5, 7, 0, 2, 6,  // 0b01000101 (69)
	0, 3, 4, 5, 7, 1, 2, 6,  // 0b01000110 (70)
	3, 4, 5, 7, 0, 1, 2, 6,  // 0b01000111 (71)
	0, 1, 2, 4, 5, 7, 3, 6,  // 0b01001000 (72)
	1, 2, 4, 5, 7, 0, 3, 6,  // 0b01001001 (73)
	0, 2, 4, 5, 7, 1, 3, 6,  // 0b01001010 (74)
	2, 4, 5, 7, 0, 1, 3, 6,  // 0b01001011 (75)
	0, 1, 4, 5, 7, 2, 3, 6,  // 0b01001100 (76)
	1, 4, 5, 7, 0, 2, 3, 6,  // 0b01001101 (77)
	0, 4, 5, 7, 1, 2, 3, 6,  // 0b01001110 (78)
	4, 5, 7, 0, 1, 2, 3, 6,  // 0b01001111 (79)
	0, 1, 2, 3, 5, 7, 4, 6,  // 0b01010000 (80)
	1, 2, 3, 5, 7, 0, 4, 6,  // 0b01010001 (81)
	0, 2, 3, 5, 7, 1, 4, 6,  // 0b01010010 (82)
	2, 3, 5, 7, 0, 1, 4, 6,  // 0b01010011 (83)
	0, 1, 3, 5, 7, 2, 4, 6,  // 0b01010100 (84)
	1, 3, 5, 7, 0, 2, 4, 6,  // 0b01010101 (85)
	0, 3, 5, 7, 1, 2, 4, 6,  // 0b01010110 (86)
	3, 5, 7, 0, 1, 2, 4, 6,  // 0b01010111 (87)
	0, 1, 2, 5, 7, 3, 4, 6,  // 0b01011000 (88)
	1, 2, 5, 7, 0, 3, 4, 6,  // 0b01011001 (89)
	0, 2, 5, 7, 1, 3, 4, 6,  // 0b01011010 (90)
	2, 5, 7, 0, 1, 3, 4, 6,  // 0b01011011 (91)
	0, 1, 5, 7, 2, 3, 4, 6,  // 0b01011100 (92)
	1, 5, 7, 0, 2, 3, 4, 6,  // 0b01011101 (93)
	0, 5, 7, 1, 2, 3, 4, 6,  // 0b01011110 (94)
	5, 7, 0, 1, 2, 3, 4, 6,  // 0b01011111 (95)
	0, 1, 2, 3, 4, 7, 5, 6,  // 0b01100000 (96)
	1, 2, 3, 4, 7, 0, 5, 6,  // 0b01100001 (97)
	0, 2, 3, 4, 7, 1, 5, 6,  // 0b01100010 (98)
	2, 3, 4, 7, 0, 1, 5, 6,  // 0b01100011 (99)
	0, 1, 3, 4, 7, 2, 5, 6,  // 0b01100100 (100)
	1, 3, 4, 7, 0, 2, 5, 6,  // 0b01100101 (101)
	0, 3, 4, 7, 1, 2, 5, 6,  // 0b01100110 (102)
	3, 4, 7, 0, 1, 2, 5, 6,  // 0b01100111 (103)
	0, 1, 2, 4, 7, 3, 5, 6,  // 0b01101000 (104)
	1, 2, 4, 7, 0, 3, 5, 6,  // 0b01101001 (105)
	0, 2, 4, 7, 1, 3, 5, 6,  // 0b01101010 (106)
	2, 4, 7, 0, 1, 3, 5, 6,  // 0b01101011 (107)
	0, 1, 4, 7, 2, 3, 5, 6,  // 0b01101100 (108)
	1, 4, 7, 0, 2, 3, 5, 6,  // 0b01101101 (109)
	0, 4, 7, 1, 2, 3, 5, 6,  // 0b01101110 (110)
	4, 7, 0, 1, 2, 3, 5, 6,  // 0b01101111 (111)
	0, 1, 2, 3, 7, 4, 5, 6,  // 0b01110000 (112)
	1, 2, 3, 7, 0, 4, 5, 6,  // 0b01110001 (113)
	0, 2, 3, 7, 1, 4, 5, 6,  // 0b01110010 (114)
	2, 3, 7, 0, 1, 4, 5, 6,  // 0b01110011 (115)
	0, 1, 3, 7, 2, 4, 5, 6,  // 0b01110100 (116)
	1, 3, 7, 0, 2, 4, 5, 6,  // 0b01110101 (117)
	0, 3, 7, 1, 2, 4, 5, 6,  // 0b01110110 (118)
	3, 7, 0, 1, 2, 4, 5, 6,  // 0b01110111 (119)
	0, 1, 2, 7, 3, 4, 5, 6,  // 0b01111000 (120)
	1, 2, 7, 0, 3, 4, 5, 6,  // 0b01111001 (121)
	0, 2, 7, 1, 3, 4, 5, 6,  // 0b01111010 (122)
	2, 7, 0, 1, 3, 4, 5, 6,  // 0b01111011 (123)
	0, 1, 7, 2, 3, 4, 5, 6,  // 0b01111100 (124)
	1, 7, 0, 2, 3, 4, 5, 6,  // 0b01111101 (125)
	0, 7, 1, 2, 3, 4, 5, 6,  // 0b01111110 (126)
	7, 0, 1, 2, 3, 4, 5, 6,  // 0b01111111 (127)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b10000000 (128)
	1, 2, 3, 4, 5, 6, 0, 7,  // 0b10000001 (129)
	0, 2, 3, 4, 5, 6, 1, 7,  // 0b10000010 (130)
	2, 3, 4, 5, 6, 0, 1, 7,  // 0b10000011 (131)
	0, 1, 3, 4, 5, 6, 2, 7,  // 0b10000100 (132)
	1, 3, 4, 5, 6, 0, 2, 7,  // 0b10000101 (133)
	0, 3, 4, 5, 6, 1, 2, 7,  // 0b10000110 (134)
	3, 4, 5, 6, 0, 1, 2, 7,  // 0b10000111 (135)
	0, 1, 2, 4, 5, 6, 3, 7,  // 0b10001000 (136)
	1, 2, 4, 5, 6, 0, 3, 7,  // 0b10001001 (137)
	0, 2, 4, 5, 6, 1, 3, 7,  // 0b10001010 (138)
	2, 4, 5, 6, 0, 1, 3, 7,  // 0b10001011 (139)
	0, 1, 4, 5, 6, 2, 3, 7,  // 0b10001100 (140)
	1, 4, 5, 6, 0, 2, 3, 7,  // 0b10001101 (141)
	0, 4, 5, 6, 1, 2, 3, 7,  // 0b10001110 (142)
	4, 5, 6, 0, 1, 2, 3, 7,  // 0b10001111 (143)
	0, 1, 2, 3, 5, 6, 4, 7,  // 0b10010000 (144)
	1, 2, 3, 5, 6, 0, 4, 7,  // 0b10010001 (145)
	0, 2, 3, 5, 6, 1, 4, 7,  // 0b10010010 (146)
	2, 3, 5, 6, 0, 1, 4, 7,  // 0b10010011 (147)
	0, 1, 3, 5, 6, 2, 4, 7,  // 0b10010100 (148)
	1, 3, 5, 6, 0, 2, 4, 7,  // 0b10010101 (149)
	0, 3, 5, 6, 1, 2, 4, 7,  // 0b10010110 (150)
	3, 5, 6, 0, 1, 2, 4, 7,  // 0b10010111 (151)
	0, 1, 2, 5, 6, 3, 4, 7,  // 0b10011000 (152)
	1, 2, 5, 6, 0, 3, 4, 7,  // 0b10011001 (153)
	0, 2, 5, 6, 1, 3, 4, 7,  // 0b10011010 (154)
	2, 5, 6, 0, 1, 3, 4, 7,  // 0b10011011 (155)
	0, 1, 5, 6, 2, 3, 4, 7,  // 0b10011100 (156)
	1, 5, 6, 0, 2, 3, 4, 7,  // 0b10011101 (157)
	0, 5, 6, 1, 2, 3, 4, 7,  // 0b10011110 (158)
	5, 6, 0, 1, 2, 3, 4, 7,  // 0b10011111 (159)
	0, 1, 2, 3, 4, 6, 5, 7,  // 0b10100000 (160)
	1, 2, 3, 4, 6, 0, 5, 7,  // 0b10100001 (161)
	0, 2, 3, 4, 6, 1, 5, 7,  // 0b10100010 (162)
	2, 3, 4, 6, 0, 1, 5, 7,  // 0b10100011 (163)
	0, 1, 3, 4, 6, 2, 5, 7,  // 0b10100100 (164)
	1, 3, 4, 6, 0, 2, 5, 7,  // 0b10100101 (165)
	0, 3, 4, 6, 1, 2, 5, 7,  // 0b10100110 (166)
	3, 4, 6, 0, 1, 2, 5, 7,  // 0b10100111 (167)
	0, 1, 2, 4, 6, 3, 5, 7,  // 0b10101000 (168)
	1, 2, 4, 6, 0, 3, 5, 7,  // 0b10101001 (169)
	0, 2, 4, 6, 1, 3, 5, 7,  // 0b10101010 (170)
	2, 4, 6, 0, 1, 3, 5, 7,  // 0b10101011 (171)
	0, 1, 4, 6, 2, 3, 5, 7,  // 0b10101100 (172)
	1, 4, 6, 0, 2, 3, 5, 7,  // 0b10101101 (173)
	0, 4, 6, 1, 2, 3, 5, 7,  // 0b10101110 (174)
	4, 6, 0, 1, 2, 3, 5, 7,  // 0b10101111 (175)
	0, 1, 2, 3, 6, 4, 5, 7,  // 0b10110000 (176)
	1, 2, 3, 6, 0, 4, 5, 7,  // 0b10110001 (177)
	0, 2, 3, 6, 1, 4, 5, 7,  // 0b10110010 (178)
	2, 3, 6, 0, 1, 4, 5, 7,  // 0b10110011 (179)
	0, 1, 3, 6, 2, 4, 5, 7,  // 0b10110100 (180)
	1, 3, 6, 0, 2, 4, 5, 7,  // 0b10110101 (181)
	0, 3, 6, 1, 2, 4, 5, 7,  // 0b10110110 (182)
	3, 6, 0, 1, 2, 4, 5, 7,  // 0b10110111 (183)
	0, 1, 2, 6, 3, 4, 5, 7,  // 0b10111000 (184)
	1, 2, 6, 0, 3, 4, 5, 7,  // 0b10111001 (185)
	0, 2, 6, 1, 3, 4, 5, 7,  // 0b10111010 (186)
	2, 6, 0, 1, 3, 4, 5, 7,  // 0b10111011 (187)
	0, 1, 6, 2, 3, 4, 5, 7,  // 0b10111100 (188)
	1, 6, 0, 2, 3, 4, 5, 7,  // 0b10111101 (189)
	0, 6, 1, 2, 3, 4, 5, 7,  // 0b10111110 (190)
	6, 0, 1, 2, 3, 4, 5, 7,  // 0b10111111 (191)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11000000 (192)
	1, 2, 3, 4, 5, 0, 6, 7,  // 0b11000001 (193)
	0, 2, 3, 4, 5, 1, 6, 7,  // 0b11000010 (194)
	2, 3, 4, 5, 0, 1, 6, 7,  // 0b11000011 (195)
	0, 1, 3, 4, 5, 2, 6, 7,  // 0b11000100 (196)
	1, 3, 4, 5, 0, 2, 6, 7,  // 0b11000101 (197)
	0, 3, 4, 5, 1, 2, 6, 7,  // 0b11000110 (198)
	3, 4, 5, 0, 1, 2, 6, 7,  // 0b11000111 (199)
	0, 1, 2, 4, 5, 3, 6, 7,  // 0b11001000 (200)
	1, 2, 4, 5, 0, 3, 6, 7,  // 0b11001001 (201)
	0, 2, 4, 5, 1, 3, 6, 7,  // 0b11001010 (202)
	2, 4, 5, 0, 1, 3, 6, 7,  // 0b11001011 (203)
	0, 1, 4, 5, 2, 3, 6, 7,  // 0b11001100 (204)
	1, 4, 5, 0, 2, 3, 6, 7,  // 0b11001101 (205)
	0, 4, 5, 1, 2, 3, 6, 7,  // 0b11001110 (206)
	4, 5, 0, 1, 2, 3, 6, 7,  // 0b11001111 (207)
	0, 1, 2, 3, 5, 4, 6, 7,  // 0b11010000 (208)
	1, 2, 3, 5, 0, 4, 6, 7,  // 0b11010001 (209)
	0, 2, 3, 5, 1, 4, 6, 7,  // 0b11010010 (210)
	2, 3, 5, 0, 1, 4, 6, 7,  // 0b11010011 (211)
	0, 1, 3, 5, 2, 4, 6, 7,  // 0b11010100 (212)
	1, 3, 5, 0, 2, 4, 6, 7,  // 0b11010101 (213)
	0, 3, 5, 1, 2, 4, 6, 7,  // 0b11010110 (214)
	3, 5, 0, 1, 2, 4, 6, 7,  // 0b11010111 (215)
	0, 1, 2, 5, 3, 4, 6, 7,  // 0b11011000 (216)
	1, 2, 5, 0, 3, 4, 6, 7,  // 0b11011001 (217)
	0, 2, 5, 1, 3, 4, 6, 7,  // 0b11011010 (218)
	2, 5, 0, 1, 3, 4, 6, 7,  // 0b11011011 (219)
	0, 1, 5, 2, 3, 4, 6, 7,  // 0b11011100 (220)
	1, 5, 0, 2, 3, 4, 6, 7,  // 0b11011101 (221)
	0, 5, 1, 2, 3, 4, 6, 7,  // 0b11011110 (222)
	5, 0, 1, 2, 3, 4, 6, 7,  // 0b11011111 (223)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11100000 (224)
	1, 2, 3, 4, 0, 5, 6, 7,  // 0b11100001 (225)
	0, 2, 3, 4, 1, 5, 6, 7,  // 0b11100010 (226)
	2, 3, 4, 0, 1, 5, 6, 7,  // 0b11100011 (227)
	0, 1, 3, 4, 2, 5, 6, 7,  // 0b11100100 (228)
	1, 3, 4, 0, 2, 5, 6, 7,  // 0b11100101 (229)
	0, 3, 4, 1, 2, 5, 6, 7,  // 0b11100110 (230)
	3, 4, 0, 1, 2, 5, 6, 7,  // 0b11100111 (231)
	0, 1, 2, 4, 3, 5, 6, 7,  // 0b11101000 (232)
	1, 2, 4, 0, 3, 5, 6, 7,  // 0b11101001 (233)
	0, 2, 4, 1, 3, 5, 6, 7,  // 0b11101010 (234)
	2, 4, 0, 1, 3, 5, 6, 7,  // 0b11101011 (235)
	0, 1, 4, 2, 3, 5, 6, 7,  // 0b11101100 (236)
	1, 4, 0, 2, 3, 5, 6, 7,  // 0b11101101 (237)
	0, 4, 1, 2, 3, 5, 6, 7,  // 0b11101110 (238)
	4, 0, 1, 2, 3, 5, 6, 7,  // 0b11101111 (239)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11110000 (240)
	1, 2, 3, 0, 4, 5, 6, 7,  // 0b11110001 (241)
	0, 2, 3, 1, 4, 5, 6, 7,  // 0b11110010 (242)
	2, 3, 0, 1, 4, 5, 6, 7,  // 0b11110011 (243)
	0, 1, 3, 2, 4, 5, 6, 7,  // 0b11110100 (244)
	1, 3, 0, 2, 4, 5, 6, 7,  // 0b11110101 (245)
	0, 3, 1, 2, 4, 5, 6, 7,  // 0b11110110 (246)
	3, 0, 1, 2, 4, 5, 6, 7,  // 0b11110111 (247)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11111000 (248)
	1, 2, 0, 3, 4, 5, 6, 7,  // 0b11111001 (249)
	0, 2, 1, 3, 4, 5, 6, 7,  // 0b11111010 (250)
	2, 0, 1, 3, 4, 5, 6, 7,  // 0b11111011 (251)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11111100 (252)
	1, 0, 2, 3, 4, 5, 6, 7,  // 0b11111101 (253)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11111110 (254)
	0, 1, 2, 3, 4, 5, 6, 7,  // 0b11111111 (255)
};


// Full 8-element partitioning block!
void QuickSort::avxPartitionBlock(int* dataPtr, __m256i pivots, int*& writeLeft, int*& writeRight)
{
	// cast writeLeft and writeRight into __m256i pointers
	__m256i* writeLeftPtr = (__m256i*) writeLeft;
	__m256i* writeRightPtr = (__m256i*) writeRight;

	// read first 8-element from the data starting at the dataPtr
	__m256i data = _mm256_lddqu_si256((__m256i*) dataPtr);

	// compare data with the pivots values
	__m256i cmpData = _mm256_cmpgt_epi32(data, pivots);

	// create mask based on the cmpData
	int mask = _mm256_movemask_ps(*(__m256*) & cmpData);

	// get the corresponding permutation values depending on the mask.
	// 8 times, since we are working with int ( 8 * 4 byte = 32 bytes)
	__m256i permData = _mm256_lddqu_si256((__m256i*) (PermTablePtr + (mask * 8)));

	// Permute data based on the permutation table
	data = _mm256_permutevar8x32_epi32(data, permData);

	// store data at the pointers postion from writeLeft and writeRight
	_mm256_storeu_si256(writeLeftPtr, data);
	_mm256_storeu_si256(writeRightPtr, data);

	// pop count
	int pc = _mm_popcnt_u32(mask);
	writeRight -= pc;
	writeLeft += 8 - pc;
}


// AVX QuickSort function, partition in-place
int* QuickSort::avxVectorizedPartitionInPlace(int* left, int* right)
{
	// define pointers positions
	auto writeLeft = left;
	auto writeRight = right - N_OFFSET;
	auto tmpLeft = _TEMPSTART;
	auto tmpRight = _TEMPEND - N_OFFSET;

	// get the pivot element which is always at the right side of the handled partition chunk
	auto pivot = *right;

	// broadcast pivot into a 8-elements of the same value
	__m256i p = _mm256_broadcastd_epi32(*(__m128i*) & pivot);

	// Make some room in the data array by doing partition and copy these values into the helperarray "_temp"
	avxPartitionBlock(left, p, tmpLeft, tmpRight);
	avxPartitionBlock(right - N_OFFSET, p, tmpLeft, tmpRight);

	// move pointers to the correct position where it should read next
	auto readLeft = left + N_OFFSET;
	auto readRight = right - 2 * N_OFFSET;

	// call partition function within the provided data
	while (readRight >= readLeft)
	{
		int* nextPtr;

		// check which side of the 8-element chunks is smaller and do partitioning on that 8-element chunk
		if ((readLeft - writeLeft) <= (writeRight - readRight))
		{
			nextPtr = readLeft;
			readLeft += N_OFFSET;
		}
		else {
			nextPtr = readRight;
			readRight -= N_OFFSET;
		}

		// do partition of the array and save result directly on the same data array in-place
		avxPartitionBlock(nextPtr, p, writeLeft, writeRight);
	}
	readRight += N_OFFSET;
	tmpRight += N_OFFSET;


	// handle remainder values inside the data array (which are in the middle of the array)
	while (readLeft < readRight) {
		auto v = *readLeft++;
		// copy values to the temp helper array
		if (v <= pivot)
		{
			*tmpLeft++ = v;
		}
		else
		{
			*--tmpRight = v;
		}
	}

	// copy values from the left side of the temp helper array back to the array
	auto leftTmpSize = (tmpLeft - _TEMPSTART);
	memcpy(writeLeft, _TEMPSTART, leftTmpSize * sizeof(int));
	writeLeft += leftTmpSize;

	// copy values from the right side of the temp helper array back to the array
	auto rightTmpSize = (_TEMPEND - tmpRight);
	memcpy(writeLeft, tmpRight, rightTmpSize * sizeof(int));

	// Since we swapped the pivot to the right side of the array, we move it back to the position of the "writeLeft" so that we can handle this
	// as a boundary!
	avxSwap(writeLeft, right);
	return writeLeft;
}


// swap function for pointers
void QuickSort::avxSwap(int* left, int* right)
{
	auto tmp = *left;
	*left = *right;
	*right = tmp;
}


// swap if left value is greater than the right value
void QuickSort::avxSwapIfGreater(int* left, int* right)
{
	if (*left <= *right)
	{
		// in this case the data is already in order.
		return;
	}
	// swaps the values
	avxSwap(left, right);
}

// insertion function for the CUT_OFF / THRESHOLD part, same code as in algd1 but with pointers!
void QuickSort::avxInsertionSort(int* left, int* right)
{
	for (int* i = left + 1; i <= right; i++)
	{
		int* j = i;
		while (j > left && *j < *(j - 1))
		{
			avxSwap(j, j - 1);
			j--;
		}
	}
}

// QuickSort with AVX recursive function
void QuickSort::avxSort(int* left, int* right)
{
	int length = right - left + 1;
	int* mid;

	// special cases for lengths of 0-3
	switch (length)
	{
	case 0:
	case 1:
		return;
	case 2:
		avxSwapIfGreater(left, right);
		return;
	case 3:
		mid = right - 1;
		avxSwapIfGreater(left, mid);
		avxSwapIfGreater(left, right);
		avxSwapIfGreater(mid, right);
		return;
	}

	// Do insertionSort if length is below the CUT_OFF / THRESHOLD
	if (length <= CUT_OFF)
	{
		avxInsertionSort(left, right);
		return;
	}

	// Compute median-of-three, of:
	// the first, mid and one before last elements
	mid = left + ((right - left) >> 1);
	avxSwapIfGreater(left, mid);
	avxSwapIfGreater(left, right - 1);
	avxSwapIfGreater(mid, right - 1);

	// Pivot is mid, place it in the right hand site
	// This prevents that the pivot in the middle might be the biggest value of the whole array, so that we cannot fulfill the QuickSort condition, that
	// in the next iteration we split the array into two segements for the "Divide and Conquer" approach.
	// By doing this we are making sure that we can split the array into two segments!
	avxSwap(mid, right);

	// do in-place partitioning
	int* boundary = avxVectorizedPartitionInPlace(left, right);

	// recursive avxSort call
	avxSort(left, boundary - 1);
	avxSort(boundary + 1, right);
}


// Serial QuickSort function, same code as in algd1
void QuickSort::serialQSort(std::vector<int>& data)
{
	qsSort(data, 0, data.size() - 1);
}

// AVX QuickSort function, partition in-place
void QuickSort::avxQSort(std::vector<int>& data)
{
	// refrence data on the first entry, since std::vector is not a pointer to its first element
	int* left = &data[0];
	avxSort(left, left + data.size() - 1);
}

// public facade function for measuring and running the serial QuickSort function
double QuickSort::serialQuickSort(std::vector<int>& data)
{
	return measuredSort(data, &QuickSort::serialQSort);
}

// public facade function for measuring and running the AVX QuickSort function
double QuickSort::avxQuickSort(std::vector<int>& data)
{
	return measuredSort(data, &QuickSort::avxQSort);
}

// Measure Sort function
double QuickSort::measuredSort(std::vector<int>& data, void(QuickSort::* sortFunc)(std::vector<int>& data))
{
	auto start = chrono::high_resolution_clock::now();
	(this->*sortFunc) (data);
	auto stop = chrono::high_resolution_clock::now();

	return (double)(stop - start).count();
}

// Check if array is sorted
bool QuickSort::proove(std::vector<int>& data)
{
	for (uint64_t i = 0; i < data.size() - 1; ++i)
	{
		if (data[i] > data[i + 1])
		{
			return false;
		}
	}
	return true;
}

// Generate random data
std::vector<int> QuickSort::createRandomData(uint64_t size)
{
	vector<int> data(size, 0);

	// random gen
	default_random_engine generator;
	uniform_int_distribution<int> distribution(0, (int)size);
	generator.seed(chrono::system_clock::now().time_since_epoch().count());

	for (int i = 0; i < size; ++i)
	{
		data[i] = distribution(generator);
	}
	return data;
}