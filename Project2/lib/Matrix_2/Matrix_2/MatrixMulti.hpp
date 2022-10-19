// MatrixMulti.hpp

#pragma once
#include <iostream>
#include <cassert>

#include <random>
#include "chronoTimer.h"

//======================================================================================
// matrixMul_1 (value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
//
// CPU Matrix multiplication: C = A * B (Row-major)
// where A is a RAxCA matrix, B is a CAxCB matrix and C is a RAxCB matrix.
// Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
//
//=====================================================================================

template <typename value_t> void matrixMul_1(value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
{
	// index of the C matrix element 
	int row;
	int col;

	// Assuming row-major like in C (note CUBLAS and Fortran uses column-major)
	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			value_t Cvalue = 0;

			for (int k = 0; k < CA; k++)
			{
				Cvalue += A[row * CA + k] * B[k * CB + col];
			}

			C[row * CB + col] = Cvalue;
		}
	}
}

// Same as matrixMul_1 but with a slightly different indexing scheme
template<typename value_t> void matrixMul_2(value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
{
	// index of the C matrix element 
	int row, col;

	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			value_t Cvalue = 0;
			int indexA = row * CA;
			int indexB = col;

			// Assuming row-major like in C (note CUBLAS and Fortran uses column-major)
			for (int k = 0; k < CA; k++)
			{
				Cvalue += A[indexA++] * B[indexB];
				indexB += CB;
			}

			C[row * CB + col] = Cvalue;
		}
	}
}

//=========================================================================================
// matrixMul_3(value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
// 
// CPU Matrix multiplication: C = A * B (Columnn-major)
// where A is a RAxCA matrix, B is a CAxCB matrix, and C is a RAxCB matrix.
// Assume matrices are stored in column-major linear array, and matrix indexing is 0-based.
//
//=========================================================================================

template <typename value_t> void matrixMul_3(value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
{
	// index of the C matrix element 
	int row, col;

	// Assuming colimn-major like in CUBLAS and Fortran, but matrix indexing is 0-based
	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			value_t Cvalue = 0.0;

			for (int k = 0; k < CA; k++)
			{
				Cvalue += A[row + RA * k] * B[col * CA + k];
			}

			C[row + col * RA] = Cvalue;  // Note column-major !!
		}
	}
}

//======================================================================================
// matrixMul_tmm(value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
//
// CPU Matrix multiplication: C = A * B (Row-major), assuming square matrices 
// where A is a RAxCA matrix, B is a CAxCB matrix and C is a RAxCB matrix.
// Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
//
// Transpose B matrix before matrix multiply.
//
//=====================================================================================

template <typename value_t> void matrixMul_tmm(value_t* C, value_t* A, value_t* B, int RA, int CA, int CB)
{
	// index of the C matrix element 
	int row, col;

	// Transpose B matrix
	value_t temp;
	for (row = 0; row < CA; row++)
		for (col = 0; col < CB; col++)
		{
			if (row > col) continue;

			temp = B[row * CA + col];
			B[row * CA + col] = B[col * CA + row];
			B[col * CA + row] = temp;
		}

	// Matrix multiplication
	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			value_t Cvalue = 0;
			int indexA = row * CA;
			int indexB = col * CA;

			for (int k = 0; k < CA; k++)
				Cvalue += A[indexA++] * B[indexB++];

			C[row * CB + col] = Cvalue;
		}
	}

}
	
//======================================================================================
//  AVX utility functions
//=====================================================================================

#include <immintrin.h>

// Horizontal sum of vectors (__mm128 float, mm256 float, and  __mm256 double)
float hsum_ps_sse3(__m128 v) 
{
	__m128 shuf = _mm_movehdup_ps(v);        // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums);		// high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return        _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 v) 
{
	__m128 vlow = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
	vlow = _mm_add_ps(vlow, vhigh);     // add the low 128
	return hsum_ps_sse3(vlow);         // and inline the sse3 version, which is optimal for AVX
	// (no wasted instructions, and all of them are the 4B minimum)
}

double hsum256_double_avx(__m256d v) 
{
	__m128d vlow = _mm256_castpd256_pd128(v);
	__m128d vhigh = _mm256_extractf128_pd(v, 1);	// high 128
	vlow = _mm_add_pd(vlow, vhigh);					// reduce down to 128
	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
}

//======================================================================================
// matrixMul_AVX_tmm(double *A, double *B, double *C, int RA, int CA, int CB)
// 
// CPU Matrix multiplication: C = A * B (Row-major), assuming square matrices 
// where A is a RAxCA matrix, B is a CAxCB matrix and C is a RAxCB matrix.
// Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
//
// Transpose B matrix before matrix multiply.
//
//=====================================================================================
void matrixMul_AVX_tmm(float* C, float* A, float* B, int RA, int CA, int CB, 
	bool ToTranspose = true) 
{
	// Transpose B matrix
	if (ToTranspose)
	{
		double temp;
		for (int i = 0; i < CA; i++)
			for (int j = 0; j < CB; j++)
			{
				if (i > j) continue;

				temp = B[i * CA + j];
				B[i * CA + j] = B[j * CA + i];
				B[j * CA + i] = temp;
			}
	}
	for (int i = 0; i < RA; i++) 
	{
		for (int j = 0; j < CB; j++) 
		{
			__m256 X = _mm256_setzero_ps();
			for (int k = 0; k < CA; k += 8) 
			{
				const __m256 AV = _mm256_loadu_ps(A + i * CA + k);
				const __m256 BV = _mm256_loadu_ps(B + j * CA + k);
				X = _mm256_fmadd_ps(AV, BV, X);
			}
			C[i * RA + j] = hsum256_ps_avx(X);
		}
	}
}

void double_matrixMul_AVX_tmm(double* C, double* A, double* B, int RA, int CA, int CB,
	bool ToTranspose = true)
{
	// Transpose B matrix
	if (ToTranspose)
	{
		double temp;
		for (int i = 0; i < CA; i++)
			for (int j = 0; j < CB; j++)
			{
				if (i > j) continue;

				temp = B[i * CA + j];
				B[i * CA + j] = B[j * CA + i];
				B[j * CA + i] = temp;
			}
	}
	for (int i = 0; i < RA; i++)
	{
		for (int j = 0; j < CB; j++)
		{
			__m256d X = _mm256_setzero_pd();
			for (int k = 0; k < CA; k += 4)
			{
				const __m256d AV = _mm256_loadu_pd(A + i * CA + k);
				const __m256d BV = _mm256_loadu_pd(B + j * CA + k);
				X = _mm256_fmadd_pd(AV, BV, X);
			}
			C[i * RA + j] = hsum256_double_avx(X);
		}
	}
}
#include <omp.h>

void matrixMul_OpenMP_AVX_tmm(float* C, float* A, float* B, int RA, int CA, int CB,
	bool ToTranspose = true)
{
	// Transpose B matrix
	if (ToTranspose)
	{
		float temp;
		for (int i = 0; i < CA; i++)
			for (int j = 0; j < CB; j++)
			{
				if (i > j) continue;

				temp = B[i * CA + j];
				B[i * CA + j] = B[j * CA + i];
				B[j * CA + i] = temp;
			}
	}

	#pragma omp parallel shared(C,A,B,RA,CA,CB) num_threads(24)
	{
		#pragma omp for schedule(dynamic)
		for (int i = 0; i < RA; i++)
		{
			for (int j = 0; j < CB; j++)
			{
				__m256 X = _mm256_setzero_ps();
				for (int k = 0; k < CA; k += 8)
				{
					const __m256 AV = _mm256_loadu_ps(A + i * CA + k);
					const __m256 BV = _mm256_loadu_ps(B + j * CA + k);
					X = _mm256_fmadd_ps(AV, BV, X);
				}
				C[i * RA + j] = hsum256_ps_avx(X);
			}
		}
	}
}
void double_matrixMul_OpenMP_AVX_tmm(double* C, double* A, double* B, int RA, int CA, int CB,
	bool ToTranspose = true)
{
	// Transpose B matrix
	if (ToTranspose)
	{
		double temp;
		for (int i = 0; i < CA; i++)
			for (int j = 0; j < CB; j++)
			{
				if (i > j) continue;

				temp = B[i * CA + j];
				B[i * CA + j] = B[j * CA + i];
				B[j * CA + i] = temp;
			}
	}

#pragma omp parallel shared(C,A,B,RA,CA,CB) num_threads(24)
	{
#pragma omp for schedule(dynamic)
		for (int i = 0; i < RA; i++)
		{
			for (int j = 0; j < CB; j++)
			{
				__m256d X = _mm256_setzero_pd();
				for (int k = 0; k < CA; k += 4)
				{
					const __m256d AV = _mm256_loadu_pd(A + i * CA + k);
					const __m256d BV = _mm256_loadu_pd(B + j * CA + k);
					X = _mm256_fmadd_pd(AV, BV, X);
				}
				C[i * RA + j] = hsum256_double_avx(X);
			}
		}
	}
}
union U256d {
	__m256d v;
	double a[4];
};
double hsum_avx(__m256d vd)
{
	vd = _mm256_hadd_pd(vd, vd);
	vd = _mm256_hadd_pd(vd, vd);
	const U256d r = { vd };
	return r.a[0] / 4;
}
void spoody(double* C, double* A, double* B, int RA, int CA, int CB)
{
	int row, col;
	double temp;
	for (row = 0; row < CA; row++)
		for (col = 0; col < CB; col++)
		{
			if (row > col) continue;
			temp = B[row * CA + col];
			B[row * CA + col] = B[col * CA + row];
			B[col * CA + row] = temp;
		}
	int i, j, k;
    #pragma omp parallel shared(C,A,B,RA,CA,CB) num_threads(24) private(i,j,k)
	{
        #pragma omp for schedule(dynamic)
		for (i = 0; i < RA; i++)
		{
			for (j = 0; j < CB; j++)
			{
				__m256d X = _mm256_setzero_pd();
				for (k = 0; k < CA; k++)
				{
					const __m256d AV = _mm256_load_pd(A + i * CA + k);
					const __m256d BV = _mm256_load_pd(B + j * CA + k);
					X = _mm256_fmadd_pd(AV, BV, X);
				}
				C[i * CB + j] = hsum_avx(X);
			}
		}
	}
}
//============================================================================
template<typename value_t> void printMatrix(value_t* M, int RM, int CM)
{
	int col, row;

	int maxRow = RM <= 8 ? RM : 8;
	int maxCol = CM <= 8 ? CM : 8;
	for (row = 0; row < maxRow; row++)
	{
		for (col = 0; col < maxCol; col++)
		{
			std::cout << "\t" << M[row * CM + col];
		}
		std::cout << "\n";
	}
}

//============================ sgemm and dgemm ===========================================

void sgemm(char transa, char transb, int m, int n, int k,
	float alpha, float a[], int lda, float b[], int ldb, float beta,
	float c[], int ldc);

void dgemm(char transa, char transb, int m, int n, int k,
	double alpha, double a[], int lda, double b[], int ldb, double beta,
	double c[], int ldc);


//============================ Timing ===========================================

template <typename value_t> void MatrixMultiTiming(int RA, int CA, int CB)
{
	int RB = CA, RC = RA, CC = CB;
	value_t *A, *B, *C, *Ref_C;


	StartTimer();
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dist(0, 1);

	// allocate 32 byte-aligned matrix memory
	A = (value_t *)_aligned_malloc(RA * CA * sizeof(value_t), 32);
	B = (value_t*)_aligned_malloc(RB * CB * sizeof(value_t), 32);
	C = (value_t*)_aligned_malloc(RA * CA * sizeof(value_t), 32);
	Ref_C = (value_t*)_aligned_malloc(RA * CA * sizeof(value_t), 32);
	
	//A = new value_t[RA * CA]; std::cout << "A = " << (unsigned long)A << "\n\n";
	//B = new value_t[RB * CB];
	//C = new value_t[RC * CC];
	//Ref_C = new value_t[RC * CC];
	
	// initialization
	for (int i = 0; i < RA * CA; ++i)
		A[i] = (value_t)(dist(rd));
	for (int i = 0; i < RB * CB; ++i)
		B[i] = (value_t)(dist(rd));
	for (int i = 0; i < RC * CC; ++i)
		C[i] = Ref_C[i] = 0;

	std::cout << "allocating and initilizing matrices using uniform distribution(0,1) time: "
		<< StopTimer() << "\n\n";

	StartTimer();
	matrixMul_1<value_t>(C, A, B, RA, CA, CB);
	//double matrixMul_1_Time = StopTimer();
	std::cout << "\t matrixMul_1 time: " << StopTimer();
	std::cout << "\n\t C[2] = " << C[2] << "\n\n";

	StartTimer();
	matrixMul_tmm<value_t>(C, A, B, RA, CA, CB);
	double matrixMul_tmm_Time = StopTimer();
	std::cout << "\t matrixMul_tmm time: " << matrixMul_tmm_Time;
	std::cout << "\n\t C[2] = " << C[2] << "\n\n";

	StartTimer();
	matrixMul_AVX_tmm(C, A, B, RA, CA, CB, false);
	double matrixMul_AVX_tmm_Time = StopTimer();
	std::cout << "\t matrixMul_AVX_tmm time: " << matrixMul_AVX_tmm_Time;
	//double_matrixMul_OpenMP_AVX_tmm(C, A, B, RA, CA, CB);
	//double double_matrixMul_OpenMP_AVX_tmm_Time = StopTimer();
	//matrixMul_OpenMP_AVX_tmm(C, A, B, RA, CA, CB);
	//double "matrixMul_OpenMP_AVX_tmm_Time = StopTimer();
	std::cout << "\n\t C[2] = " << C[2] << "\n\n";

	StartTimer();
	//  Ref_C := alpha*op( A )*op( B ) + beta*C
	sgemm('n', 'n', RA, CB, CA, 1.0, A, CA, B, RB, 1.0, Ref_C, CB);
	double sgemm_Time = StopTimer();
	std::cout << "\t sgemm time: " << sgemm_Time;
	std::cout << "\n\t Ref_C[0] = " << Ref_C[0] << "\n\n";
/*

	StartTimer();
	//  Ref_C := alpha*op( A )*op( B ) + beta*C
	dgemm('n', 'n', RA, CB, CA, 1.0, A, CA, B, RB, 1.0, Ref_C, CB);
	double dgemm_Time = stopTimer();

	//printMatrix(C, RA, CB); std::cout << "\n";
	std::cout << "Ref_C[2] = " << Ref_C[2] << "\n\n";
*/	
	delete A;
	delete B;
	delete C;
	delete Ref_C;
}