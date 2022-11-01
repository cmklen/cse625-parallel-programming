#include <iostream>


int all_pairs(uint64_t nRows) ;

int main()
{
    const uint64_t nRows = 60000;

    // Compute all_pairs_distance between first nRows of MNIST train images
    all_pairs(nRows);

    return 0;
}
