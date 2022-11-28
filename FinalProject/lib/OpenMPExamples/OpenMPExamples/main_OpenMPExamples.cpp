// OpenMPExamples.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

void omp_hello_world();
void omp_hello_world_2();

void matrix_vector();

void reduction_test();
void SIMD_Test();


int main()
{
    omp_hello_world();
    //omp_hello_world_2();

    //matrix_vector();

    //reduction_test();
    //SIMD_Test();

    system("Pause");

    return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu
