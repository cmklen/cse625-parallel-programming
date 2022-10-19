// chronoTimer.cpp

#include <chrono>

using namespace std;

auto start = chrono::high_resolution_clock::now();

void StartTimer()
{
	start = chrono::high_resolution_clock::now();
}

double StopTimer() // in seconds
{
	auto end = chrono::high_resolution_clock::now();
	return 1e-9 * chrono::duration_cast<chrono::nanoseconds>(end - start).count();
}
