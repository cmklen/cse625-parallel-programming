// 3_reduction.cpp

// C++11 parallel reduction

#include "include/hpc_helpers.hpp"

#include <vector>

#include <iostream>
#include <iomanip>      // std::setprecision
#include <cstdint>
#include <cmath>
//#define M_PI    3.141592653589793238462643383279502884L

//
// OpenMP reduction
//
#include <omp.h>

template <typename value_t>
value_t omp_reduction_sum(std::vector<value_t>& V, int m, int num_threads)
{
    value_t sum = 0;

    //omp_set_num_threads(num_threads);
    #pragma omp parallel shared(V, sum) num_threads(num_threads)
    {
        value_t localsum = 0;
        #pragma omp for
        for (int i = 0; i < m; i++)
            localsum += V[i];

        #pragma omp atomic
        sum += localsum;
    }

    return sum;
}


//
// sum with reduction clause; for reduction(+; sum) has this menaing:
//
// 1 OpenMP will make a copy of the reduction variable (e.g. sum) per thread,
//   initialized to the identity of the reduction operator, for instance 0 for +.
// 2 Each thread will then reduce into its local variable;
// 3 At the end of the loop, the local results are combined, again using the reduction
//   operator, into the global variable (e.g. sum)

template <typename value_t>
value_t omp_reduction_clause_sum(std::vector<value_t>& V, int m, int num_threads)
{
    value_t sum = 0;

    omp_set_num_threads(num_threads);
    #pragma omp parallel for shared(V) reduction(+:sum)
    for (int i = 0; i < m; i++)
        sum += V[i];

    return sum;
}

//====================================================================

//
// C++ Threads
//

#include <thread>
#include <mutex>
template <typename value_t>
value_t sequential_reduction_sum(std::vector<value_t>& V, int m)
{
    value_t sum = 0;
    for (int i = 0; i < m; ++i)
        sum += V[i];

    return sum;
}

template <typename value_t>
value_t parallel_sum_v1(std::vector<value_t>& V, int m, int num_threads = 6)
{
    std::mutex barrier;
    value_t sum = 0;

    // this function is called by the threads
    auto block = [&](const int& id) -> void
    {    //       ^-- capture whole scope by reference

        // compute chunk size, lower and upper (block thread scheduling)
        const int chunk = (m + num_threads - 1) / num_threads;
        const int lower = id * chunk;
        const int upper = std::min(lower + chunk, m);

        // only computes rows between lower and upper
        for (int i = lower; i < upper; ++i)
        {
            std::lock_guard<std::mutex> block(barrier);
            sum += V[i];
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (int id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto& thread : threads)
        thread.join();

    return sum;
}

template <typename value_t>
value_t parallel_sum_v2(std::vector<value_t>& V, int m, int num_threads = 6)
{
    std::mutex barrier;
    value_t sum = 0;

    // this function is called by the threads
    auto block = [&](const int& id) -> void
    {    //       ^-- capture whole scope by reference

        // compute chunk size, lower and upper task id
        const int chunk = (m + num_threads - 1) / num_threads;
        const int lower = id * chunk;
        const int upper = std::min(lower + chunk, m);

        value_t partial_sum = 0;

        // only computes rows between lower and upper
        for (int i = lower; i < upper; ++i)
            partial_sum += V[i];

        // Block other threads until this thread finishes
        std::lock_guard<std::mutex> block(barrier);
        //barrier.lock();
        sum += partial_sum;
        //barrier.unlock();
    };
    // business as usual
    std::vector<std::thread> threads;

    for (int id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto& thread : threads)
        thread.join();

    return sum;
}

template <typename value_t>
value_t parallel_max_v1(std::vector<value_t>& V, int m, int num_threads = 6)
{
    std::mutex barrier;
    value_t max = V[0];

    // this function is called by the threads
    auto block = [&](const int& id) -> void
    {    //       ^-- capture whole scope by reference

        // compute chunk size, lower and upper (block thread scheduling)
        const int chunk = (m + num_threads - 1) / num_threads;
        const int lower = id * chunk;
        const int upper = std::min(lower + chunk, m);

        // only computes rows between lower and upper
        for (int i = lower; i < upper; ++i)
        {
            std::lock_guard<std::mutex> block(barrier);
            if (V[i] > max) max = V[i];
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (int id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto& thread : threads)
        thread.join();

    return max;
}

void reduction_test()
{
    const int m = 1UL << 26;
    const int num_threads = 8;
    double sum;

    std::vector<double> V(m, 1);

    TIMERSTART(Seq_sum)
        sum = sequential_reduction_sum(V, m);
    TIMERSTOP(Seq_sum)
    std::cout << std::fixed << std::setprecision(9); // set float format to fixed and 9 decimal digits
    std::cout << "Sequential reduction sum = " << sum <<"\n\n";

    TIMERSTART(OpenMP_sum)
        sum = omp_reduction_sum(V, m, num_threads);
    TIMERSTOP(OpenMP_sum)
    std::cout << "OpenMP sum: sum = " << sum << " using " << num_threads
        << " threads.\n\n";

    TIMERSTART(OpenMP_reduction_clause_sum)
        sum = omp_reduction_clause_sum(V, m, num_threads);
    TIMERSTOP(OpenMP_reduction_clause_sum)
    std::cout << "OpenMP reduction sum: sum = " << sum << " using " << num_threads
        << " threads.\n\n";


    TIMERSTART(Parallel_sum_v1)
        sum = parallel_sum_v1(V, m, num_threads);
    TIMERSTOP(Parallel_sum_v1)
    std::cout << "C++ threads sum v1: sum = " << sum << " using " << num_threads
        << " threads.\n\n";

    TIMERSTART(Parallel_sum_v2)
        sum = parallel_sum_v2(V, m, num_threads);
    TIMERSTOP(Parallel_sum_v2)
    std::cout << "C++ threads sum v2: sum = " << sum << " using " << num_threads
        << " threads.\n\n";
}
