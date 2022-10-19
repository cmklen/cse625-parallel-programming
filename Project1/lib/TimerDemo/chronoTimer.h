
#include <iostream>
//#include <cunistd>	// sleep function for UNIX
//#include <windows.h>	// Sleep (milliseconds) function for Windows

#include <chrono>

#define TIMING

#ifdef TIMING
#define INIT_TIMER auto start = std::chrono::high_resolution_clock::now();
#define START_TIMER  start = std::chrono::high_resolution_clock::now();
#define STOP_TIMER(name)  std::cout << "Runtime of " << name << " = " << \
    std::chrono::duration_cast<std::chrono::microseconds>( \
     std::chrono::high_resolution_clock::now()-start).count()/1000000.0 << " seconds " << std::endl; 
#else
#define INIT_TIMER
#define START_TIMER
#define STOP_TIMER(name)
#endif

