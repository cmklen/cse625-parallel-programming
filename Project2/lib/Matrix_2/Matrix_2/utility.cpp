// utility.cpp

#include <math.h>

// Compute distance between two float vectors of size n
double absolute_dist (float * A, float * B, int n)
{
    double dist = 0;

    for (int i = 0; i < n; ++i)
        dist += fabs(A[i]-B[i]);

    return dist;
}

double relative_dist (float * A, float * B, int n)
{
    double dist = 0, d;

    for (int i = 0; i < n; ++i)
    {
        d = fabs(A[i]) + fabs(B[i]);

        if (d > 0)
            d = fabs(A[i]-B[i]) / d;

        dist += d;
    }

    return dist;
}
