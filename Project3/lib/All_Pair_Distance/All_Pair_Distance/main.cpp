#include <iostream>


int all_pairs(uint64_t nRows) ;

int main()
{
    uint64_t nRows[6] = {400,800,10000,20000,30000,60000};

    // Compute all_pairs_distance between first nRows of MNIST train images
    for (int i =0; i < 6; i++)
    {
        all_pairs(nRows[i]);

    }

    return 0;
}
