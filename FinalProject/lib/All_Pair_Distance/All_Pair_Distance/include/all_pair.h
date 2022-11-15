#include <iostream>
#include <cstdint>						// uint64_t
#include <vector>						// std::vector
#include <thread>						// std::thread
#include <algorithm>					// std::min
#include "include/hpc_helpers.hpp"		// timers, no_init_t
#include "include/binary_IO.hpp"		// load_binary

void sequential_all_pairs(
    std::vector<float>& mnist,		    
    std::vector<float>& all_pair,
    uint64_t rows,						
    uint64_t cols);