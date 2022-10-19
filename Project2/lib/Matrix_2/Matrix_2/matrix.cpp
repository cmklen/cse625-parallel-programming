// matrix.cpp

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <immintrin.h>
#include <vector>
//======================================================================================
// matrixMul_RowMajor (float* C, float* A, float* B, int RA, int CA, int CB)
// Note: this is the base-line matrix multiplication C++ implementation on
//       which performance-improvement implementations will be made.

// Matrix multiplication: C = A * B (Row-major)
// where A is a RAxCA matrix, B is a CAxCB matrix and C is a RAxCB matrix.
// Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
//
//=====================================================================================
void matrixMul_RowMajor(float *C, float *A, float *B, int RA, int CA, int CB)
{
	// index of the C matrix element
	int row, col;

	// Assuming row-major like in C (note CUBLAS and Fortran uses column-major)
	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			float Cvalue = 0;

			for (int k = 0; k < CA; k++)
				Cvalue += A[row * CA + k] * B[k * CB + col];

			C[row * CB + col] = Cvalue;
		}
	}
}

//=========================================================================================
// matrixMul_ColMajor(float* C, float* A, float* B, int RA, int CA, int CB)
//
// CPU Matrix multiplication: C = A * B (Column-major)
// where A is a RAxCA matrix, B is a CAxCB matrix, and C is a RAxCB matrix.
// Assume matrices are stored in column-major linear array, and matrix indexing is 0-based.
//
//=========================================================================================

void matrixMul_ColMajor(float *C, float *A, float *B, int RA, int CA, int CB)
{
	// index of the C matrix element
	int row, col;

	// Assuming column-major like in CUBLAS and Fortran, but matrix indexing is 0-based
	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			float Cvalue = 0.0;

			for (int k = 0; k < CA; k++)
				Cvalue += A[row + RA * k] * B[col * CA + k];

			C[row + col * RA] = Cvalue; // Note column-major !!
		}
	}
}

//======================================================================================
// matrixMul_tmm(float* C, float* A, float* B, int RA, int CA, int CB)
// Note: B is modified (i.e., transpose B matrix before matrix multiply).

// CPU Matrix multiplication: C = A * B (Row-major), assuming square matrices
// where A is a RAxCA matrix, B is a CAxCB matrix and C is a RAxCB matrix.
// Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
//
//=====================================================================================
void transpose(float *B, int RB, int CB)
{
	int row, col;
	float temp;

	// Transpose
	for (row = 0; row < RB; row++)
		for (col = 0; col < CB; col++)
		{
			if (row > col)
				continue;

			temp = B[row * CB + col];
			B[row * CB + col] = B[col * CB + row];
			B[col * CB + row] = temp;
		}
}

void matrixMul_tmm(float *C, float *A, float *B, int RA, int CA, int CB)
{
	// index of the C matrix element
	int row, col;

	// Transpose B matrix
	transpose(B, CA, CB);

	// Matrix multiplication
	for (row = 0; row < RA; ++row)
	{
		for (col = 0; col < CB; ++col)
		{
			float Cvalue = 0;
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

// Horizontal sum of 8 floats in __m256 X
float hsum_avx(__m256 &X)
{
	float temp[8], hsum = 0;

	_mm256_storeu_ps(temp, X);

	for (int i = 0; i < 8; i++)
		hsum += temp[i];

	return hsum;
}
float hsum_avx_2(__m256 &X)
{
	float hsum = 0;
	float *x = (float *)&X;

	for (int i = 0; i < 8; i++)
		hsum += x[i];

	return hsum;
}

// Horizontal sum of vectors (__mm128 float, mm256 float, and  __mm256 double)
float hsum_ps_sse3(__m128 &v)
{
	__m128 shuf = _mm_movehdup_ps(v); // broadcast elements 3,1 to 2,0
	__m128 sums = _mm_add_ps(v, shuf);
	shuf = _mm_movehl_ps(shuf, sums); // high half -> low half
	sums = _mm_add_ss(sums, shuf);
	return _mm_cvtss_f32(sums);
}

float hsum256_ps_avx(__m256 &v)
{
	__m128 vlow = _mm256_castps256_ps128(v);
	__m128 vhigh = _mm256_extractf128_ps(v, 1); // high 128
	vlow = _mm_add_ps(vlow, vhigh);				// add the low 128
	return hsum_ps_sse3(vlow);					// and inline the sse3 version, which is optimal for AVX
												// (no wasted instructions, and all of them are the 4B minimum)
}

double hsum256_double_avx(__m256d &v)
{
	__m128d vlow = _mm256_castpd256_pd128(v);
	__m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
	vlow = _mm_add_pd(vlow, vhigh);				 // reduce down to 128
	__m128d high64 = _mm_unpackhi_pd(vlow, vlow);
	return _mm_cvtsd_f64(_mm_add_sd(vlow, high64)); // reduce to scalar
}

//======================================================================================
// matrixMul_AVX_tmm(float *A, float *B, float *C, int RA, int CA, int CB)
//  Note: B is modified (i.e., Transpose B matrix before matrix multiply).
//
// CPU Matrix multiplication: C = A * B (Row-major), assuming square matrices
// where A is a RAxCA matrix, B is a CAxCB matrix and C is a RAxCB matrix.
// Assume matrices are stored in row-major linear array, and matrix indexing is 0-based.
//
//=====================================================================================
void matrixMul_AVX_tmm(float *C, float *A, float *B, int RA, int CA, int CB,
					   bool ToTranspose = true)
{
	if (ToTranspose)
		// Transpose B matrix
		transpose(B, CA, CB);

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
			// C[i * RA + j] = hsum_avx(X);
			// C[i * RA + j] = hsum_avx_2(X);
		}
	}
}

float AVXDot(const std::vector<float> &v1, const std::vector<float> &v2)
{
	__m256 C = _mm256_setzero_ps();
	size_t length = (v1.size() <= v2.size() ? v1.size() : v2.size());
	float sum;
	
	for (int i = 0; i < length; i += 8)
	{
		__m256 X = _mm256_setzero_ps();
		const __m256 mmA = _mm256_loadu_ps((float *)&v1[i]);
		const __m256 mmB = _mm256_loadu_ps((float *)&v2[i]);
		X = _mm256_mul_ps(mmA, mmB);
		sum += hsum256_ps_avx(X);
	}
	return sum;
}

float SequentialDot(const std::vector<float> &v1, const std::vector<float> &v2)
{
	float result = 0;
	size_t length = (v1.size() <= v2.size() ? v1.size() : v2.size());
	for (int i = 0; i < length; ++i)
	{
		result += v1[i] * v2[i];
	}
	return result;
}
