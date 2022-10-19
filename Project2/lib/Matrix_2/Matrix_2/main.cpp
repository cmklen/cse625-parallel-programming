// main.cpp

#include <iostream>
#include <random>
#include "chronoTimer.h"

//============================================================================
//
//  External function declarations
//
//============================================================================

// Functions defined in matrix.cpp
void matrixMul_RowMajor (float* C, float* A, float* B, int RA, int CA, int CB);
void matrixMul_ColMajor (float* C, float* A, float* B, int RA, int CA, int CB);
void matrixMul_tmm (float* C, float* A, float* B, int RA, int CA, int CB);
void matrixMul_AVX_tmm(float *A, float *B, float *C, int RA, int CA, int CB,
                       bool ToTranspose = true);
float AVXDot(const std::vector<float> &v1, const std::vector<float> &v2);
float SequentialDot(const std::vector<float> &v1, const std::vector<float> &v2);

// Functions defined in gemm.cpp
void sgemm(char transa, char transb, int m, int n, int k,
	float alpha, float a[], int lda, float b[], int ldb, float beta,
	float c[], int ldc);
void CPU_sgemm (float* C, float* A, float* B, int RA, int CA, int CB);

// Functions defined in utility.cpp
double absolute_dist (float * A, float * B, int n);
double relative_dist (float * A, float * B, int n);

//============= End of External function declarations ===================



//=======================================================================
//
// Matrix Multiplication Timing Functions
//
//=======================================================================

void matrixMul_RowMajor_Timing(float *C, float *A, float* B,
                int ROWA, int COLA, int COLB)
{
    StartTimer();
        matrixMul_RowMajor(C, A, B, ROWA, COLA, COLB);
    std::cout << "matrixMul_ColMajor time: " << StopTimer() << " seconds\n";
    std::cout << "\t C[0] = " << C[0] << "\n\n";
}


void tmm_Timing(float* Ref_C, float *C, float *A, float* B,
                int ROWA, int COLA, int COLB, bool baseLine = false)
{
    if (baseLine)
    {
        // Base-line matrix multiplication timing
        StartTimer();
            matrixMul_RowMajor(C, A, B, ROWA, COLA, COLB);
        std::cout << "matrixMul_RowMajor time: " << StopTimer() << " seconds\n";
        std::cout << "\t C[0] = " << C[0] << "\n\n";
    }

    StartTimer();
        matrixMul_tmm(Ref_C, A, B, ROWA, COLA, COLB);
    std::cout << "matrixMul_tmm time: " << StopTimer() << " seconds\n";
    std::cout << "\t Ref_C[0] = " << Ref_C[0] << "\n\n";

    std::cout << "relative_distance (C, Ref_C) = " <<
        relative_dist (C, Ref_C, ROWA * COLA) << "\n";
        //absolute_dist (C, Ref_C, ROWA * COLA) << "\n";
}

void sgemm_Timing(float* Ref_C, float* C, float* A, float* B,
                 int ROWA, int COLA, int COLB, bool baseLine = false)
{
    if (baseLine)
    {
        // Base-line matrix multiplication timing
        StartTimer();
            matrixMul_ColMajor(C, A, B, ROWA, COLA, COLB);
        std::cout << "matrixMul_ColMajor time: " << StopTimer() << " seconds\n";
        std::cout << "\t C[0] = " << C[0] << "\n\n";
    }

    StartTimer();
        CPU_sgemm(Ref_C, A, B, ROWA, COLA, COLB);
	double sgemm_Time = StopTimer();
	std::cout << "sgemm time: " << sgemm_Time << " seconds\n";
	std::cout << "\t Ref_C[0] = " << Ref_C[0] << "\n\n";

	std::cout << "Distance (C, Ref_C) = " <<
        relative_dist (C, Ref_C, ROWA * COLA) << "\n\n";

}

void AVX_tmm_Timing (float* Ref_C, float *C, float *A, float* B,
                int ROWA, int COLA, int COLB, bool baseLine = false)
{
    if (baseLine)
    {
        // Base-line matrix multiplication timing
        StartTimer();
        matrixMul_RowMajor(C, A, B, ROWA, COLA, COLB);
        std::cout << "matrixMul_RowMajor time: " << StopTimer() << " seconds\n";
        std::cout << "\t C[0] = " << C[0] << "\n\n";
    }

    StartTimer();
        matrixMul_AVX_tmm (Ref_C, A, B, ROWA, COLA, COLB);
    std::cout << "matrixMul_AVX_tmm time: " << StopTimer() << " seconds\n";
    std::cout << "\t Ref_C[0] = " << Ref_C[0] << "\n\n";

    std::cout << "relative_distance (C, Ref_C) = " <<
        relative_dist (C, Ref_C, ROWA * COLA) << "\n";
        //absolute_dist (C, Ref_C, ROWA * COLA) << "\n";
}

void group_timing(float* Ref_C, float *C, float *A, float* B,
                int ROWA, int COLA, int COLB, bool baseLine = false)
{
    if (baseLine)
    {
        // Base-line matrix multiplication timing
        StartTimer();
        matrixMul_RowMajor(C, A, B, ROWA, COLA, COLB);
        std::cout << "matrixMul_RowMajor time: " << StopTimer() << " seconds\n";
        std::cout << "\t C[0] = " << C[0] << "\n\n";
    }

    StartTimer();
        matrixMul_tmm(Ref_C, A, B, ROWA, COLA, COLB);
    std::cout << "matrixMul_tmm time: " << StopTimer() << " seconds\n";
    std::cout << "\t Ref_C[0] = " << Ref_C[0] << "\n\n";

    StartTimer();
        matrixMul_AVX_tmm (Ref_C, A, B, ROWA, COLA, COLB);
    std::cout << "matrixMul_AVX_tmm time: " << StopTimer() << " seconds\n";
    std::cout << "\t Ref_C[0] = " << Ref_C[0] << "\n\n";

}

void AVXDot_timing(const std::vector<float> &v1, const std::vector<float> &v2)
{
     float dot;
    StartTimer();
     dot = AVXDot(v1,v2);
    std::cout << "AVXDot time: " << StopTimer() << " seconds\n";
    std::cout << "\t AVX Dot Result = " << dot << "\n\n";
}

void SeqDot_timing(std::vector<float> &v1, std::vector<float> &v2)
{
    float dot;
    StartTimer();
    dot = SequentialDot(v1, v2);
    std::cout << "SeqDot time: " ;//<< StopTimer() << " seconds\n";
    std::cout << "\t Seq Dot Result = " << dot << "\n\n";
}



//============== End of Timing Test functions definitions ================
#define VSIZE 6400000
#define ROWA 500
#define COLA 500
#define COLB 500
#define ROWB COLA
#define ROWC ROWA
#define COLC COLB

int main()
{

    float *A, *B, *C, *Ref_C;

	printf("single precision %dx%d matrix times %dx%d matrix:\n\n",
		ROWA, COLA, ROWB, COLB);

	StartTimer();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);

	// allocate 32 byte-aligned matrix memory
	A = (float *)_aligned_malloc(ROWA * COLA * sizeof(float), 32);
	B = (float*)_aligned_malloc(ROWB * COLB * sizeof(float), 32);
	C = (float*)_aligned_malloc(ROWA * COLA * sizeof(float), 32);
	Ref_C = (float*)_aligned_malloc(ROWA * COLA * sizeof(float), 32);

	// initialization
	for (int i = 0; i < ROWA * COLA; ++i)
		A[i] = (float)(dist(rd));
	for (int i = 0; i < ROWB * COLB; ++i)
		B[i] = (float)(dist(rd));
	for (int i = 0; i < ROWC * COLC; ++i)
		C[i] = Ref_C[i] = 0;

    std::vector<float> v1(VSIZE, 1.0f), v2(VSIZE, 1.0f);
	//A[0] = .1;
	//A[1] = .2;
	//A[2] = .3;

	//B[0] = .1;
	//B[1] = .5;
	//B[2] = .7;

	std::cout << "allocating and initializing matrices using uniform distribution(0,1) time: "
		<< StopTimer() << " seconds\n\n";

    // Base-line matrix multiplication timing
    //matrixMul_RowMajor_Timing(C, A, B, ROWA, COLA, COLB);

    //tmm_Timing(Ref_C, C, A, B, ROWA, COLA, COLB, false);

    //AVX_tmm_Timing(Ref_C, C, A, B, ROWA, COLA, COLB, false);

    //group_timing(Ref_C, C, A, B, ROWA, COLA, COLB, true);

    //sgemm_Timing(Ref_C, C, A, B, ROWA, COLA, COLB, true);
    SeqDot_timing(v1, v2);
    AVXDot_timing(v1, v2);

    _aligned_free(A);
    _aligned_free(B);
    _aligned_free(C);
    _aligned_free(Ref_C);

    system("pause");

    return 0;
}


