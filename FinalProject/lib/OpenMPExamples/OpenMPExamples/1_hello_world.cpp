// 1_hello_world.cpp

#include <iostream>
#include <cstdint>

// 
// OpenMP
//

#include <omp.h>
void omp_hello_world()
{
	std::cout << "The CPU has " << omp_get_num_procs() << " cores. \n\n";

	#pragma omp parallel
	{
		int id = omp_get_thread_num();
		std::cout << "Hello, world greeting from thread "
			<< id << std::endl;
		//printf("Hello, world greeting from thread %d \n", id);
	}
}

void omp_hello_world_2()
{
	int nthreads, tid;

	#pragma omp parallel private(tid)
	{
		/* Obtain and print thread id */
		tid = omp_get_thread_num();
		printf("Hello World from thread = %d\n", tid);

		/* Only master thread does this */
		if (tid == 0)
		{
			nthreads = omp_get_num_threads();
			printf("Number of threads = %d\n", nthreads);
		}

	}  /* All threads join master thread and terminate */

}



// 
// C++ Threads
//

#include <vector>
#include <thread>

void  hello_world()
{
	const uint64_t num_threads = 8;

	// this  function  is  called  by the  threads
	auto hello = [&](const uint64_t & id) -> void
	{
		std::cout << "Hello from thread: " << id << std::endl; 
	};

	std::vector<std::thread> threads;

	for (uint64_t id = 0; id < num_threads; id++)
		threads.emplace_back(hello, id);

	for (auto& thread : threads)
		thread.join();
}