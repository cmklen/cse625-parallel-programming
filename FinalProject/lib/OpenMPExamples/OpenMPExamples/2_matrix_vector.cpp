// 3_matrix_vector.cpp

#include "include/hpc_helpers.hpp"

#include <iostream>
#include <cstdint>
#include <vector>

//
// OpenMP
//
#include <omp.h>

void omp_matrix_vector(std::vector<float>& A,
    std::vector<float>& x,
    std::vector<float>& b,
    int m,
    int n)
{
    #pragma omp parallel for
    // notice in OpenMP 2.0 array index must be int type
    for (int row = 0; row < m; row++)
    {
        float accum = float(0);
        for (int col = 0; col < n; col++)
            accum += A[row * n + col] * x[col];
        b[row] = accum;
    }
}

//=====================================================================

// 
//
// C++ threads
//

#include <thread>

#include <algorithm> // for std::min function

void init(
    std::vector<float>& A,
    std::vector<float>& x,
    int m,
    int n) 
{
    for (int row = 0; row < m; row++)
        for (int col = 0; col < n; col++)
            A[row * n + col] = 1; // = row >= col ? 1 : 0;

    for (int col = 0; col < m; col++)
        x[col] = 1; // col;
}

void sequential_mult(
    std::vector<float>& A,
    std::vector<float>& x,
    std::vector<float>& b,
    int m,
    int n) 
{
    for (int row = 0; row < m; row++) 
    {
        float accum = float(0);
        for (int col = 0; col < n; col++)
            accum += A[row*n+col]*x[col];
        b[row] = accum;
    }
}

void cyclic_parallel_mult(
    std::vector<float>& A, // linear memory for A
    std::vector<float>& x, // to be mapped vector
    std::vector<float>& b, // result vector
    int m,               // number of rows
    int n,               // number of cols
    int num_threads=12) { // number of threads

    // this  function  is  called  by the  threads
    auto cyclic = [&] (const int& id) -> void 
    {
        // indices are incremented with a stride of p
        for (int row = id; row < m; row += num_threads) 
        {
            float accum = float(0);

	        for (int col = 0; col < n; col++)
                accum += A[row*n+col]*x[col];

            b[row] = accum;
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (int id = 0; id < num_threads; id++)
        threads.emplace_back(cyclic, id);

    for (auto& thread : threads)
        thread.join();
}

void block_parallel_mult(
    std::vector<float>& A,
    std::vector<float>& x,
    std::vector<float>& b,
    int m,
    int n,
    int num_threads=12) 
{
    // this function is called by the threads
    auto block = [&] (const int& id) -> void
	{    //       ^-- capture whole scope by reference

        // compute chunk size, lower and upper task id
        const int chunk = SDIV(m, num_threads);
        const int lower = id*chunk;
        const int upper = std::min(lower+chunk, m);

        // only computes rows between lower and upper
        for (int row = lower; row < upper; row++) {
            float accum = float(0);
            for (int col = 0; col < n; col++)
                accum += A[row*n+col]*x[col];
            b[row] = accum;
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (int id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto& thread : threads)
        thread.join();
}

void block_cyclic_parallel_mult(
    std::vector<float>& A,
    std::vector<float>& x,
    std::vector<float>& b,
    int m,
    int n,
    int num_threads=12,
    int chunk_size=64/sizeof(float)) 
{
    // this  function  is  called  by the  threads
    auto block_cyclic = [&] (const int& id) -> void 
    {

        // precomupute the stride
	    const int stride = num_threads*chunk_size;
	    const int offset = id*chunk_size;

        // for each block of size chunk_size in cyclic order
        for (int lower = offset; lower < m; lower += stride) 
        {

            // compute the upper border of the block
            const int upper = std::min(lower+chunk_size, m);

	        // for each row in the block
            for (int row = lower; row < upper; row++) 
            {
		        // accumulate the contributions
		        float accum = float(0);
		        for (int col = 0; col < n; col++)
                    accum += A[row*n+col]*x[col];

                b[row] = accum;
            }
	    }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (int id = 0; id < num_threads; id++)
        threads.emplace_back(block_cyclic, id);

    for (auto& thread : threads)
        thread.join();
}

void matrix_vector() 
{
    const uint64_t n = 1UL << 15;  // 32768
    const uint64_t m = 1UL << 15;
    int num_threads = std::thread::hardware_concurrency();

    std::cout << "Timing of matrix-vector multiplication of " << m << "x" << n 
        << " matrix using "<< num_threads << " threads\n\n";

    TIMERSTART(overall)

    TIMERSTART(alloc)
    std::vector<float> A(m * n);
    std::vector<float> x(n);
    std::vector<float> b(m);
    TIMERSTOP(alloc)

    TIMERSTART(init)
    init(A, x, m, n);
    TIMERSTOP(init)

    std::cout << "\nSequential:\n";
    TIMERSTART(Seq_mult)
    sequential_mult(A, x, b, m, n);
    std::cout << "sequential: b[10] = " << b[10] << "\n";
    TIMERSTOP(Seq_mult)

    std::cout << "\nOpenMP:\n";
    TIMERSTART(OpenMP_mult)
    omp_matrix_vector(A, x, b, m, n);
    std::cout << "OpenMP: b[10] = " << b[10] << "\n";
    TIMERSTOP(OpenMP_mult)

    std::cout << "\nBlock:\n";
    TIMERSTART(BP_mult)
    block_parallel_mult(A, x, b, m, n, num_threads);
    std::cout << "block: b[10] = " << b[10] << "\n";
    TIMERSTOP(BP_mult)

    std::cout << "\nCyclic:\n";
    TIMERSTART(CP_mult)
    cyclic_parallel_mult(A, x, b, m, n, num_threads);
    std::cout << "cyclic: b[10] = " << b[10] << "\n";
    TIMERSTOP(CP_mult)

    std::cout << "\nBlock_Cyclic:\n";
    TIMERSTART(BCP_mult)
    block_cyclic_parallel_mult(A, x, b, m, n, num_threads);
    std::cout << "block_cyclic: b[10] = " << b[10] << "\n";
    TIMERSTOP(BCP_mult)

    std::cout << "\n\n";
    TIMERSTOP(overall)

}
