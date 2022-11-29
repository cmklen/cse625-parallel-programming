#include <mutex>
#include <atomic>
#include <iostream>
#include <cstdint>                 // uint64_t
#include <vector>                  // std::vector
#include <thread>                  // std::thread
#include <algorithm>               // std::min
#include "include/hpc_helpers.hpp" // timers, no_init_t
#include "include/binary_IO.hpp"   // load_binary
#include "omp.h"
#include "chronoTimer.h"

#define COLS 784 // 28 * 28;
#define ROWS 60000

typedef void (*AllPairsWorker_t)(
    std::vector<float> &,
    std::vector<float> &,
    uint64_t,
    uint64_t,
    uint64_t);

typedef struct
{
    std::vector<float> mnist;
    std::vector<float> allPairs;
    uint64_t rows;
    uint64_t threads;
    uint64_t chunksize;
} AllPairWorkerData_t;

// block work distribution
void block_all_pairs(
    std::vector<float> &mnist,
    std::vector<float> &all_pair,
    uint64_t rows,
    uint64_t num_threads = 64,
    uint64_t unused = 0)
{
    auto block = [&](const uint64_t &id) -> void
    {
        // pre-compute offset and stride
        uint64_t chunk_size = (rows + num_threads - 1) / num_threads;
        const uint64_t off = id * chunk_size;
        const uint64_t str = num_threads * chunk_size;

        // for each block of size chunk_size in cyclic order
        for (uint64_t lower = off; lower < rows; lower += str)
        {
            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower + chunk_size, rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < COLS; j++)
                    {
                        float residue = mnist[i * COLS + j] - mnist[I * COLS + j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i * rows + I] = all_pair[I * rows + i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto &thread : threads)
        thread.join();
}

// block cyclic work distribution
void block_cyclic_all_pairs(
    std::vector<float> &mnist,
    std::vector<float> &all_pair,
    uint64_t rows,
    uint64_t num_threads = 64,
    uint64_t chunk_size = 64 / sizeof(float))
{
    auto block_cyclic = [&](const uint64_t &id) -> void
    {
        // pre-compute offset and stride
        const uint64_t off = id * chunk_size;
        const uint64_t str = num_threads * chunk_size;

        // for each block of size chunk_size in cyclic order
        for (uint64_t lower = off; lower < rows; lower += str)
        {
            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower + chunk_size, rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < COLS; j++)
                    {
                        float residue = mnist[i * COLS + j] - mnist[I * COLS + j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i * rows + I] = all_pair[I * rows + i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(block_cyclic, id);

    for (auto &thread : threads)
        thread.join();
}

std::mutex mutex;

void dynamic_all_pairs(
    std::vector<float> &mnist,
    std::vector<float> &all_pair,
    uint64_t rows,
    uint64_t num_threads = 64,
    uint64_t chunk_size = 64 / sizeof(float))
{
    // declare mutex and current lower index
    uint64_t global_lower = 0;

    auto dynamic_block_cyclic = [&](const uint64_t &id) -> void
    {
        // assume we have not done anything
        uint64_t lower = 0;

        // while there are still rows to compute
        while (lower < rows)
        {
            // update lower row with global lower row
            {
                std::lock_guard<std::mutex> lock_guard(mutex);
                lower = global_lower;
                global_lower += chunk_size;
            }

            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower + chunk_size, rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < COLS; j++)
                    {
                        float residue = mnist[i * COLS + j] - mnist[I * COLS + j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i * rows + I] = all_pair[I * rows + i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(dynamic_block_cyclic, id);

    for (auto &thread : threads)
        thread.join();
}

// block work distribution
void OpenMP_block_all_pairs(
    std::vector<float> &mnist,
    std::vector<float> &all_pair,
    uint64_t rows,
    uint64_t num_threads = 64,
    uint64_t unused = 0)
{
    auto block = [&](const uint64_t &id) -> void
    {
        // pre-compute offset and stride
        uint64_t chunk_size = (rows + num_threads - 1) / num_threads;
        const uint64_t off = id * chunk_size;
        const uint64_t str = num_threads * chunk_size;

// for each block of size chunk_size in cyclic order
        for (uint64_t lower = off; lower < rows; lower += str)
        {
            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower + chunk_size, rows);
// for all entries below the diagonal (i'=I)
            #pragma omp parallel for
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < COLS; j++)
                    {
                        float residue = mnist[i * COLS + j] - mnist[I * COLS + j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i * rows + I] = all_pair[I * rows + i] = accum;
                }
            }
        }
    };

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        block(omp_get_thread_num());
    }
}

// block cyclic work distribution
void OpenMP_block_cyclic_all_pairs(
    std::vector<float>&mnist,
    std::vector<float>&all_pair,
    uint64_t rows,

    uint64_t num_threads = 64,
    uint64_t chunk_size = 64 / sizeof(float))
{
    auto block_cyclic = [&](const uint64_t &id) -> void
    {
        // pre-compute offset and stride
        const uint64_t off = id * chunk_size;
        const uint64_t str = num_threads * chunk_size;

        // for each block of size chunk_size in cyclic order
        for (uint64_t lower = off; lower < rows; lower += str)
        {
            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower + chunk_size, rows);

            // for all entries below the diagonal (i'=I)
            #pragma omp parallel for
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < COLS; j++)
                    {
                        float residue = mnist[i * COLS + j] - mnist[I * COLS + j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i * rows + I] = all_pair[I * rows + i] = accum;
                }
            }
        }
    };

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        block_cyclic(omp_get_thread_num());
    }
}

void OpenMP_dynamic_all_pairs(
    std::vector<float> &mnist,
    std::vector<float> &all_pair,
    uint64_t rows,
    uint64_t num_threads = 64,
    uint64_t chunk_size = 64 / sizeof(float))
{
    // declare mutex and current lower index
    uint64_t global_lower = 0;

    auto dynamic_block_cyclic = [&](const uint64_t &id) -> void
    {
        // assume we have not done anything
        uint64_t lower = 0;

        // while there are still rows to compute
        while (lower < rows)
        {
            // update lower row with global lower row
            {
                std::lock_guard<std::mutex> lock_guard(mutex);
                lower = global_lower;
                global_lower += chunk_size;
            }

            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower + chunk_size, rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < COLS; j++)
                    {
                        float residue = mnist[i * COLS + j] - mnist[I * COLS + j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i * rows + I] = all_pair[I * rows + i] = accum;
                }
            }
        }
    };

    omp_set_num_threads(num_threads);
    #pragma omp parallel
    {
        dynamic_block_cyclic(omp_get_thread_num());
    }
}

void printImagePixels(std::vector<float> image, int n)
{
    for (int i = 0; i < n; i++)
        std::cout << i << ": " << image[i] << "\n";
}

void printAndTimePairs(
    std::string name,
    AllPairsWorker_t Worker,
    AllPairWorkerData_t *data)
{
    std::cout << name << "...\n";
    StartTimer();
    Worker(data->mnist, data->allPairs, data->rows, data->threads, data->chunksize);
    std::cout << "\t " << name << " time = " << StopTimer() << "\n";
    std::cout << "\tall_pair[1000] = " << data->allPairs[1000] << "\n\n";
}

int all_pairs(uint64_t nRows = 60000)
{
    std::cout << "Load MNIST train-image dataset ......\n";
    StartTimer();
    std::vector<float> mnist(ROWS * COLS, 5); // values initialized to 5
    load_binary(mnist.data(), ROWS * COLS,
                "./data/train-images.bin");
    StopTimer();

    // validate data
    if ((int)(mnist[156] * 10000) != 4941) // the value should be 0.494118
        return 0;
    if (nRows < 1 || nRows > ROWS)
        return -1;

    std::vector<float> all_pair(nRows * nRows);

    AllPairWorkerData_t data;
    data.allPairs = all_pair;
    data.mnist = mnist;
    data.rows = nRows;
    data.threads = 20;
    data.chunksize = 2;

    std::cout << "\n\nCompute pair_wise_distance for first " << nRows << " MNIST train images (gcc) using " << data.threads << " threads \n\n";
    //printAndTimePairs("block_all_pairs", block_all_pairs, &data);
    printAndTimePairs("OpenMP_block_all_pairs", OpenMP_block_all_pairs, &data);
    printAndTimePairs("OpenMP_block_cyclic_all_pairs", OpenMP_block_cyclic_all_pairs, &data);
    printAndTimePairs("OpenMP_dynamic_all_pairs", OpenMP_dynamic_all_pairs, &data);
    // printAndTimePairs("block_cyclic_all_pairs", block_cyclic_all_pairs, &data);
    // printAndTimePairs("dynamic_all_pairs", dynamic_all_pairs, &data);

    return 0;
}
