//	timers.cpp
//


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
//============================================================================= 
double PCFreq = 0.0; 
__int64 CounterStart = 0; 

//Start Counter 
void StartCounter() 
{ 
     LARGE_INTEGER li; 
     if(!QueryPerformanceFrequency(&li)) 
         std::cout << "QueryPerformanceFrequency failed!\n"; 
 
	//This gives us a value in milli seconds 
    //PCFreq = double(li.QuadPart)/1000.0; 
 
	 //This gives us a value in seconds 
     PCFreq = double(li.QuadPart); 
 
     QueryPerformanceCounter(&li); 
     CounterStart = li.QuadPart; 
} 

//Get Counter 
double GetCounter() 
{ 
    LARGE_INTEGER li; 
     QueryPerformanceCounter(&li); 
    return double(li.QuadPart-CounterStart)/PCFreq; 
}

double get_wall_time()
{
    LARGE_INTEGER time,freq;
    if (!QueryPerformanceFrequency(&freq)){
        //  Handle error
        return 0;
    }
    if (!QueryPerformanceCounter(&time)){
        //  Handle error
        return 0;
    }
    return (double)time.QuadPart / freq.QuadPart;
}

//////////////////////////////////////////////////////////////////////////////////////
//
//	CPU timer
//
//  Usage:	double start = cputimer();
//			................ // time the elapsed cpu-time in seconds of this region
//			double elapsedTime = cputimer() - start;
//
//////////////////////////////////////////////////////////////////////////////////////

double cputimer()
{
    FILETIME createTime;
    FILETIME exitTime;
    FILETIME kernelTime;
    FILETIME userTime;

    if ( GetProcessTimes( GetCurrentProcess( ),
        &createTime, &exitTime, &kernelTime, &userTime ) != -1 )
    {
        SYSTEMTIME userSystemTime;
        if ( FileTimeToSystemTime( &userTime, &userSystemTime ) != -1 )
            return (double)userSystemTime.wHour * 3600.0 +
            (double)userSystemTime.wMinute * 60.0 +
            (double)userSystemTime.wSecond +
            (double)userSystemTime.wMilliseconds / 1000.0;
    }
}

// Return user cpu time in seconds
double get_cpu_time()
{
    FILETIME a,b,c,d;
    if (GetProcessTimes(GetCurrentProcess(),&a,&b,&c,&d) != 0){
        //  Returns total user time.
        //  Can be tweaked to include kernel times as well.
        return
            (double)(d.dwLowDateTime | ((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
    }else{
        //  Handle error
        return 0;
    }
}

