#include <iostream>
#include <cstdint>						// uint64_t
#include <vector>						// std::vector
#include <thread>						// std::thread
#include <algorithm>					// std::min
#include "include/hpc_helpers.hpp"		// timers, no_init_t
#include "include/binary_IO.hpp"		// load_binary

void sequential_all_pairs(
    std::vector<float>& mnist,		    // rowsXcols matrix
    std::vector<float>& all_pair,		// rowsXrows matrix
    uint64_t rows,						// rows: number of  images
    uint64_t cols)						// cols: number of pixels per image
{
    // Compute distance for all entries below the diagonal
    for (uint64_t i = 0; i < rows; i++)
    {
        for (uint64_t I = 0; I <= i; I++)
        {
            // compute squared Euclidean distance
            float accum = float(0);
            for (uint64_t j = 0; j < cols; j++)
            {
                float residue = mnist[i*cols+j] - mnist[I*cols+j];
                accum += residue * residue;
            }

            // write Delta[i,i'] = Delta[i',i] = dist(i, i')
            all_pair[i*rows+I] = all_pair[I*rows+i] = accum;
        }
    }
}

// block work distribution
void block_all_pairs(
    std::vector<float>& mnist,
    std::vector<float>& all_pair,
    uint64_t rows,
    uint64_t cols,
    uint64_t num_threads = 64)
{
    auto block = [&] (const uint64_t& id) -> void
    {
        // pre-compute offset and stride
        uint64_t chunk_size = (rows + num_threads -1) / num_threads;
        const uint64_t off = id * chunk_size;
        const uint64_t str = num_threads * chunk_size;

        // for each block of size chunk_size in cyclic order
        for (uint64_t lower = off; lower < rows; lower += str)
        {
            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower+chunk_size,rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < cols; j++)
                    {
                        float residue = mnist[i*cols+j] - mnist[I*cols+j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i*rows+I] = all_pair[I*rows+i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(block, id);

    for (auto& thread : threads)
        thread.join();
}

// block cyclic work distribution
void block_cyclic_all_pairs(
    std::vector<float>& mnist,
    std::vector<float>& all_pair,
    uint64_t rows,
    uint64_t cols,
    uint64_t num_threads = 64,
    uint64_t chunk_size = 64 / sizeof(float))
{
    auto block_cyclic = [&] (const uint64_t& id) -> void
    {
        // pre-compute offset and stride
        const uint64_t off = id*chunk_size;
        const uint64_t str = num_threads*chunk_size;

        // for each block of size chunk_size in cyclic order
        for (uint64_t lower = off; lower < rows; lower += str)
        {
            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower+chunk_size,rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < cols; j++)
                    {
                        float residue = mnist[i*cols+j] - mnist[I*cols+j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i*rows+I] = all_pair[I*rows+i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(block_cyclic, id);

    for (auto& thread : threads)
        thread.join();
}

#include <mutex>
#include <atomic>

std::mutex mutex;

void dynamic_all_pairs(
    std::vector<float>& mnist,
    std::vector<float>& all_pair,
    uint64_t rows,
    uint64_t cols,
    uint64_t num_threads=64,
    uint64_t chunk_size=64/sizeof(float))
{
    // declare mutex and current lower index
    uint64_t global_lower = 0;

    auto dynamic_block_cyclic = [&] (const uint64_t& id ) -> void
    {
        // assume we have not done anything
        uint64_t lower = 0;

        // while there are still rows to compute
        while (lower < rows)
        {
            // update lower row with global lower row
            {
                std::lock_guard<std::mutex> lock_guard(mutex);
                       lower  = global_lower;
                global_lower += chunk_size;
            }

            // compute the upper border of the block (exclusive)
            const uint64_t upper = std::min(lower+chunk_size,rows);

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = lower; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < cols; j++)
                    {
                        float residue = mnist[i*cols+j] - mnist[I*cols+j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i*rows+I] = all_pair[I*rows+i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(dynamic_block_cyclic, id);

    for (auto& thread : threads)
        thread.join();
}

void dynamic_all_pairs_rev(
    std::vector<float>& mnist,
    std::vector<float>& all_pair,
    uint64_t rows,
    uint64_t cols,
    uint64_t num_threads=64,
    uint64_t chunk_size=64/sizeof(float))
{
    // declare mutex and current lower index
    uint64_t global_lower = 0;

    auto dynamic_block_cyclic = [&] (const uint64_t& id ) -> void
    {
        // assume we have not done anything
        uint64_t lower = 0;

        // while there are still rows to compute
        while (lower < rows)
        {
            // update lower row with global lower row
            {
                std::lock_guard<std::mutex> lock_guard(mutex);
                       lower  = global_lower;
                global_lower += chunk_size;
            }

            // compute the upper border of the block (exclusive)
            const uint64_t upper = rows  >= lower ? rows-lower : 0;
            const uint64_t LOWER = upper >= chunk_size ? upper-chunk_size : 0;

            // for all entries below the diagonal (i'=I)
            for (uint64_t i = LOWER; i < upper; i++)
            {
                for (uint64_t I = 0; I <= i; I++)
                {
                    // compute squared Euclidean distance
                    float accum = float(0);
                    for (uint64_t j = 0; j < cols; j++)
                    {
                        float residue = mnist[i*cols+j]
                                        - mnist[I*cols+j];
                        accum += residue * residue;
                    }

                    // write Delta[i,i'] = Delta[i',i]
                    all_pair[i*rows+I] = all_pair[I*rows+i] = accum;
                }
            }
        }
    };

    // business as usual
    std::vector<std::thread> threads;

    for (uint64_t id = 0; id < num_threads; id++)
        threads.emplace_back(dynamic_block_cyclic, id);

    for (auto& thread : threads)
        thread.join();
}

void printImagePixels (std::vector<float> image, int n)
{
    for (int i = 0; i < n; i++)
        std::cout << i << ": " << image[i] << "\n";

}

# include "chronoTimer.h"

int all_pairs(uint64_t nRows = 60000)
{
    // binary MNIST train-images dataset
    const uint64_t rows = 60000;    // number of train images
    const uint64_t cols = 28*28;    // resolution
                                    // pixel (gray) value is stored in a float

    // load MNIST train-image data from the binary file train-image.bin
    std::cout << "Load MNIST train-image dataset ......\n";
    StartTimer();
    std::vector<float> mnist(rows*cols, 5); // values initialized to 5
    load_binary(mnist.data(), rows*cols,
       "./data/train-images.bin");
    StopTimer();

    // verify the read data
    std::cout << "mnist[156] = " << mnist[156] << "\n";  // the value should be 0.494118
    //printImagePixels (mnist, 500);

	if (nRows < 1 || nRows > rows)
        return -1;

    std::cout << "\n\nCompute pair_wise_distance for first " << nRows <<
        " MNIST train images (gcc) ...\n\n";

	// For nRows = 60,000, it requires about 14 GB of memory
	std::vector<float> all_pair(nRows*nRows);

    // The sequential computing took about 30 min to complete
    std::cout << "Starting sequential_all_pairs ...\n";
	StartTimer ();
        //sequential_all_pairs(mnist, all_pair, nRows, cols);
	std::cout << "\tsequential_all_pairs time = " << StopTimer() << "\n";
	std::cout << "\tall_pair[1000] = " << all_pair[1000] << "\n\n";

	// This block parallel version took about 260 seconds to complete
    // the distance matrix computation for nRows = 60,000 using 12 threads
    // on the DC 119 computers (compared with sequential version, which took
    // about 30 minutes.

	std::cout << "Starting block_all_pairs ...\n";
    StartTimer();
        block_all_pairs(mnist, all_pair, nRows, cols, 12);
    std::cout << "\tBlock_all_pairs time = " << StopTimer() << "\n";
    std::cout << "\tall_pair[1000] = " << all_pair[1000] << "\n\n";

    // This block-cyclic-parallel version took about 150 seconds to complete
    // for nRows = 60,000 using 12 threads and chunk size 2 on DC119 computer
    std::cout << "Starting block_cyclic_all_pairs ...\n";
    StartTimer();
        block_cyclic_all_pairs(mnist, all_pair, nRows, cols, 12, 2);
    std::cout << "\tblock_cyclic_all_pairs time = " << StopTimer() << "\n";
    std::cout << "\tall_pair[1000] = " << all_pair[1000] << "\n\n";

    // This dynamic-parallel version took about 140 seconds to complete
    // for nRows = 60,000 using 12 threads and chunk size 2 on DC 119 computers
    std::cout << "Starting dynamic_all_pairs ...\n";
    StartTimer();
	dynamic_all_pairs(mnist, all_pair, nRows, cols,12,2);
    std::cout << "\tdynamic_all_pairs time = " << StopTimer() << "\n";
    std::cout << "\tall_pair[1000] = " << all_pair[1000] << "\n\n";

    //dump_binary(mnist.data(), rows*rows, "./all_pairs.bin");

	return 0;
}
