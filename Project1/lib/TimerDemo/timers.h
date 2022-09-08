//	timers.h
//
//		Wall timer
// and
//		CPU timer
//

#ifndef _timers_h_
#define _timers_h_

#include <iostream> 
#include <Windows.h>	// for LARGE_INTEGER and QueryPerformanceFrequency function

//============================================================================= 
//
// Wall-time timer Variables and functions
//
// Usage:	StartCounter();
//			................ // time the elapsed wall-time in seconds of this region
//			double elapsedTime = GetCounter();
//

//Start Counter 
void StartCounter();

//Get Counter 
double GetCounter();

double get_wall_time();

#endif
