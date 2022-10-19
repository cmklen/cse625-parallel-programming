// main.cpp
//
// This program shows you how to use three different timers to
// measure execution times of sequential and c++11 multi-threading
// dot product.

#include <iostream>
#include <vector>

float SequentialDot(const std::vector<float> &v1, const std::vector<float> &v2);
float SequentialDot2(const std::vector<float> &v1, const std::vector<float> &v2);
float ThreadDot(const std::vector<float> &v1, const std::vector<float> &v2);

float SequentialDot(float * v1, float * v2, int n)
{
	float result = 0;

	for (int i = 0; i < n; ++i)
		result += v1[i] * v2[i];

	return result;
}

#include "include/hpc_helpers.hpp"

void hpc_helpersTimer(int N)
{
	float dot;

	std::vector<float> v1(N, 1.0f), v2(N, 1.0f);

	TIMERSTART(dot)
	dot = SequentialDot(v1, v2);
	TIMERSTOP(dot)
	std::cout << "Sequential dot product result = " << dot << std::endl;

	TIMERSTART(thread_dot)
	dot = ThreadDot(v1, v2);
	TIMERSTOP(thread_dot)
	std::cout << "MultiThreading dot product result = " << dot << std::endl;
}

//
// Use the timer based on C++11 chrono functions (requiring chronoTimer.h)
//

#include "chronoTimer.h"

void chronoTimer(int N)
{
	float dot;

	std::vector<float> v1(N, 1.0f), v2(N, 1.0f);

	INIT_TIMER

	START_TIMER
	dot = SequentialDot(v1, v2);
	STOP_TIMER("Sequential time measured by chrono timer")
	std::cout << "Sequential dot product result = " << dot << std::endl;

	START_TIMER
	dot = ThreadDot(v1, v2);
	STOP_TIMER("MutiThreading time measured by chrono timer")
	std::cout << "MultiThreading dot product result = " << dot << std::endl;
}

//
// Use Windows timers (requiring timers.cpp and timers.h)
//
#include "timers.h"

void WindowsTimer(int N)
{
	double time1, time2;
	float dot;
	std::vector<float> v1(N, 1.0f), v2(N, 1.0f);

	StartCounter();
	dot = SequentialDot(v1, v2);
	time1 = GetCounter();
	std::cout << "Sequential time measured by Windows timer = " << time1 << " seconds \n";
	std::cout << "Sequential dot product result = " << dot << std::endl;

	StartCounter();
	dot = ThreadDot(v1, v2);
	time2 = GetCounter();
	std::cout << "MultiThreading time measured by Windows timer = " << time2 << " seconds\n";
	std::cout << "\tSpeed-up = " << time1 / time2 << "\n";
	std::cout << "MultiThreading dot product result = " << dot << std::endl;

}

//
// Use chTimer.h (a cross-platform timer)
//
#include "chTimer.h"

void chTimerTest(int N)
{
	double time;
	float dot;
	chTimerTimestamp start, stop;

	std::vector<float> v1(N, 1.0f), v2(N, 1.0f);

	chTimerGetTime(&start);
	dot = SequentialDot(v1, v2);
	chTimerGetTime(&stop);
	time = chTimerElapsedTime(&start, &stop);
	std::cout << "Sequential time measured by chTimer = " << time << " seconds\n";
	std::cout << "Sequential dot product result = " << dot << std::endl;

	chTimerGetTime(&start);
	dot = ThreadDot(v1, v2);
	chTimerGetTime(&stop);
	time = chTimerElapsedTime(&start, &stop);
	std::cout << "MutiThreading time measured by chTimer = " << time << " seconds\n";
	std::cout << "MutiThreading dot product result = " << dot << std::endl;
}

void seqDotTest(int N)
{
	double time;
	float dot;
	chTimerTimestamp start, stop;

	std::vector<float> v1(N, 1.0f), v2(N, 1.0f);

	chTimerGetTime(&start);
	dot = SequentialDot(v1, v2);
	chTimerGetTime(&stop);
	time = chTimerElapsedTime(&start, &stop);
	std::cout << "Sequential time measured by chTimer = " << time << " seconds\n";
	std::cout << "Sequential dot product result = " << dot << std::endl;
}

void avxDotTest(int N)
{
	double time;
	float dot;
	chTimerTimestamp start, stop;

	std::vector<float> v1(N, 1.0f), v2(N, 1.0f);

	chTimerGetTime(&start);
	//dot = AVXDot(v1, v2);
	chTimerGetTime(&stop);
	time = chTimerElapsedTime(&start, &stop);
	std::cout << "AVX time measured by chTimer = " << time << " seconds\n";
	std::cout << "AVX dot product result = " << dot << std::endl;
}

#define _N	(64000000)		// Number of vector elements

int main()
{
	std::cout << "Dot product of two ones float vectors of length " << _N << " ----->\n";

	std::cout << "\nhpc_helpers.hpp timer tests\n";
	//hpc_helpersTimer(_N);

	//std::cout << "\nchrono timer tests\n";
	//chronoTimer(_N);

	//std::cout << "\nWinodows timer tests\n";
	//WindowsTimer(_N);

	//std::cout << "\nchTimer tests\n";
	//chTimerTest(_N);

	seqDotTest(_N);

	std::cout << "\n============== Timer tests done. ============\n\n";

	system("PAUSE");

	return 0;
}
