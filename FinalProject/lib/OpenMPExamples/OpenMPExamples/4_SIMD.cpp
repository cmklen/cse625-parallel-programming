// 4_SIMD.cpp

#include <iostream>
#include <omp.h>
#include <math.h>

#define N 10000
float x[N][N], y[N][N];

void SIMD_Test()
{
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; i++)
        {
            #pragma omp simd safelen(18)
            for (int j = 18; j < N- 18; j += 18)
            {
                x[i][j] = x[i][j-18] + sinf(y[i][j]);
            }
        }
    }


}
