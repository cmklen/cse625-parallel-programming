# include <stdlib.h>
# include <stdio.h>

int i4_max(int a, int b)
{
	return ((a >= b) ? a : b);
}

/******************************************************************************/
// 
// This function assumes Columnn-major 2D array storage
//

void dgemm ( char transa, char transb, int m, int n, int k, 
  double alpha, double a[], int lda, double b[], int ldb, double beta, 
  double c[], int ldc)

/******************************************************************************/
/*
  Purpose:

    DGEMM computes C = alpha * A * B and related operations.

  Discussion:

    DGEMM performs one of the matrix-matrix operations

     C := alpha * op ( A ) * op ( B ) + beta * C,

    where op ( X ) is one of

      op ( X ) = X   or   op ( X ) = X',

    ALPHA and BETA are scalars, and A, B and C are matrices, with op ( A )
    an M by K matrix, op ( B ) a K by N matrix and C an N by N matrix.

  Licensing:

    This code is distributed under the GNU LGPL license.
    
  Modified:

    10 February 2014

  Author:

    Original FORTRAN77 version by Jack Dongarra.
    C version by John Burkardt.

  Parameters:

    Input, char TRANSA, specifies the form of op( A ) to be used in
    the matrix multiplication as follows:
    'N' or 'n', op ( A ) = A.
    'T' or 't', op ( A ) = A'.
    'C' or 'c', op ( A ) = A'.

    Input, char TRANSB, specifies the form of op ( B ) to be used in
    the matrix multiplication as follows:
    'N' or 'n', op ( B ) = B.
    'T' or 't', op ( B ) = B'.
    'C' or 'c', op ( B ) = B'.

    Input, int M, the number of rows of the  matrix op ( A ) and of the  
    matrix C.  0 <= M.

    Input, int N, the number  of columns of the matrix op ( B ) and the 
    number of columns of the matrix C.  0 <= N.

    Input, int K, the number of columns of the matrix op ( A ) and the 
    number of rows of the matrix op ( B ).  0 <= K.

    Input, double ALPHA, the scalar multiplier 
    for op ( A ) * op ( B ).

    Input, double A(LDA,KA), where:
    if TRANSA is 'N' or 'n', KA is equal to K, and the leading M by K
    part of the array contains A;
    if TRANSA is not 'N' or 'n', then KA is equal to M, and the leading
    K by M part of the array must contain the matrix A.

    Input, int LDA, the first dimension of A as declared in the calling 
    routine.  When TRANSA = 'N' or 'n' then LDA must be at least max ( 1, M ), 
    otherwise LDA must be at least max ( 1, K ).

    Input, double B(LDB,KB), where:
    if TRANSB is 'N' or 'n', kB is N, and the leading K by N 
    part of the array contains B;
    if TRANSB is not 'N' or 'n', then KB is equal to K, and the leading
    N by K part of the array must contain the matrix B.

    Input, int LDB, the first dimension of B as declared in the calling 
    routine.  When TRANSB = 'N' or 'n' then LDB must be at least max ( 1, K ), 
    otherwise LDB must be at least max ( 1, N ).

    Input, double BETA, the scalar multiplier for C.

    Input/output, double C(LDC,N).
    Before entry, the leading M by N part of this array must contain the 
    matrix C, except when BETA is 0.0, in which case C need not be set 
    on entry.
    On exit, the array C is overwritten by the M by N matrix
      alpha * op ( A ) * op ( B ) + beta * C.

    Input, int LDC, the first dimension of C as declared in the calling 
    routine.  max ( 1, M ) <= LDC.
*/
{
  int i;
  int info;
  int j;
  int l;
  int ncola;
  int nrowa;
  int nrowb;
  int nota;
  int notb;
  double temp;
/*
  Set NOTA and NOTB as true if A and B respectively are not
  transposed and set NROWA, NCOLA and NROWB as the number of rows
  and columns of A and the number of rows of B respectively.
*/
  nota = ( ( transa == 'N' ) || ( transa == 'n' ) );

  if ( nota )
  {
    nrowa = m;
    ncola = k;
  }
  else
  {
    nrowa = k;
    ncola = m;
  }

  notb = ( ( transb == 'N' ) || ( transb == 'n' ) );

  if ( notb )
  {
    nrowb = k;
  }
  else
  {
    nrowb = n;
  }
/*
  Test the input parameters.
*/
  info = 0;

  if ( ! ( transa == 'N' || transa == 'n' ||
           transa == 'C' || transa == 'c' ||
           transa == 'T' || transa == 't' ) )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input TRANSA had illegal value.\n" );
    exit ( 1 );
  }

  if ( ! ( transb == 'N' || transb == 'n' ||
           transb == 'C' || transb == 'c' ||
           transb == 'T' || transb == 't' ) )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input TRANSB had illegal value.\n" );
    exit ( 1 );
  }

  if ( m < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input M had illegal value.\n" );
    exit ( 1 );
  }

  if ( n < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input N had illegal value.\n" );
    exit ( 1 );
  }

  if ( k  < 0 )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input K had illegal value.\n" );
    exit ( 1 );
  }

  if ( lda < i4_max ( 1, nrowa ) )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input LDA had illegal value.\n" );
    exit ( 1 );
  }

  if ( ldb < i4_max ( 1, nrowb ) )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input LDB had illegal value.\n" );
    exit ( 1 );
  }

  if ( ldc < i4_max ( 1, m ) )
  {
    fprintf ( stderr, "\n" );
    fprintf ( stderr, "DGEMM - Fatal error!\n" );
    fprintf ( stderr, "  Input LDC had illegal value.\n" );
    exit ( 1 );
  }
/*
  Quick return if possible.
*/
  if ( m == 0 )
  {
    return;
  }

  if ( n == 0 )
  {
    return;
  }

  if ( ( alpha == 0.0 || k == 0 ) && ( beta == 1.0 ) )
  {
    return;
  }
/*
  And if alpha is 0.0.
*/
  if ( alpha == 0.0 )
  {
    if ( beta == 0.0 )
    {
      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          c[i+j*ldc] = 0.0;
        }
      }
    }
    else
    {
      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          c[i+j*ldc] = beta * c[i+j*ldc];
        }
      }
    }
    return;
  }
/*
  Start the operations.
*/
  if ( notb )
  {
/*
  Form  C := alpha*A*B + beta*C.
*/
    if ( nota )
    {
      for ( j = 0; j < n; j++ )
      {
        if ( beta == 0.0 )
        {
          for ( i = 0; i < m; i++ )
          {
            c[i+j*ldc] = 0.0;
          }
        }
        else if ( beta != 1.0 )
        {
          for ( i = 0; i < m; i++ )
          {
            c[i+j*ldc] = beta * c[i+j*ldc];
          }
        }

        for ( l = 0; l < k; l++ )
        {
          if ( b[l+j*ldb] != 0.0 )
          {
            temp = alpha * b[l+j*ldb];
            for ( i = 0; i < m; i++ )
            {
              c[i+j*ldc] = c[i+j*ldc] + temp * a[i+l*lda];
            }
          }
        }

      }
    }
/*
  Form  C := alpha*A'*B + beta*C
*/
    else
    {
      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          temp = 0.0;
          for ( l = 0; l < k; l++ )
          {
            temp = temp + a[l+i*lda] * b[l+j*ldb];
          }

          if ( beta == 0.0 )
          {
            c[i+j*ldc] = alpha * temp;
          }
          else
          {
            c[i+j*ldc] = alpha * temp + beta * c[i+j*ldc];
          }
        }
      }
    }
  }
/*
  Form  C := alpha*A*B' + beta*C
*/
  else
  {
    if ( nota )
    {
      for ( j = 0; j < n; j++ )
      {
        if ( beta == 0.0 )
        {
          for ( i = 0; i < m; i++ )
          {
            c[i+j*ldc] = 0.0;
          }
        }
        else if ( beta != 1.0 )
        {
          for ( i = 0; i < m; i++ )
          {
            c[i+j*ldc] = beta * c[i+j*ldc];
          }
        }

        for ( l = 0; l < k; l++ )
        {
          if ( b[j+l*ldb] != 0.0 )
          {
            temp = alpha * b[j+l*ldb];
            for ( i = 0; i < m; i++ )
            {
              c[i+j*ldc] = c[i+j*ldc] + temp * a[i+l*lda];
            }
          }
        }
      }
    }
/*
  Form  C := alpha*A'*B' + beta*C
*/
    else
    {
      for ( j = 0; j < n; j++ )
      {
        for ( i = 0; i < m; i++ )
        {
          temp = 0.0;
          for ( l = 0; l < k; l++ )
          {
            temp = temp + a[l+i*lda] * b[j+l*ldb];
          }
          if ( beta == 0.0 )
          {
            c[i+j*ldc] = alpha * temp;
          }
          else
          {
            c[i+j*ldc] = alpha * temp + beta * c[i+j*ldc];
          }
        }
      }
    }
  }

  return;
}

#include <iostream>

void dgemmDemo()
{
	int rowA, colA, rowB, colB, rowC, colC;
	rowA = colA = rowB = colB = rowC = colC = 3200;

	double *A, *B, *C;
	double alpha = 1, beta = 1;

	A = (double *)malloc(rowA * colA * sizeof(double));
	B = (double *)malloc(rowB * colB * sizeof(double));
	C = (double *)malloc(rowA * colB * sizeof(double));

	for (int i = 0; i < rowA * colA; i++) A[i] = 1;
	for (int i = 0; i < rowB * colB; i++) B[i] = 1;
	for (int i = 0; i < rowC * colC; i++) C[i] = 0;

	for (int i = 0; i < rowC * colC; i++) C[i] = 0;		// need to reset C matrix

	/*  C := alpha*op( A )*op( B ) + beta*C, */
	dgemm('n', 'n', rowA, colB, colA, alpha, A, rowA, B, rowB, beta, C, rowC);


	// Print some C elements
	std::cout << "dgemm results: " << std::endl;
	std::cout << "\tC[10,10] = " << C[10010] << std::endl;
	std::cout << "\tC[100,10] = " << C[100010] << std::endl;

	free(A);
	free(B);
	free(C);
}

//================================================================================
// Single precision version
//
// This function assumes Columnn-major 2D array storage
//

void sgemm(char transa, char transb, int m, int n, int k,
	float alpha, float a[], int lda, float b[], int ldb, float beta,
	float c[], int ldc)

	/******************************************************************************/
	/*
	Purpose:

	DGEMM computes C = alpha * A * B and related operations.

	Discussion:

	DGEMM performs one of the matrix-matrix operations

	C := alpha * op ( A ) * op ( B ) + beta * C,

	where op ( X ) is one of

	op ( X ) = X   or   op ( X ) = X',

	ALPHA and BETA are scalars, and A, B and C are matrices, with op ( A )
	an M by K matrix, op ( B ) a K by N matrix and C an N by N matrix.

	Licensing:

	This code is distributed under the GNU LGPL license.

	Modified:

	10 February 2014

	Author:

	Original FORTRAN77 version by Jack Dongarra.
	C version by John Burkardt.

	Parameters:

	Input, char TRANSA, specifies the form of op( A ) to be used in
	the matrix multiplication as follows:
	'N' or 'n', op ( A ) = A.
	'T' or 't', op ( A ) = A'.
	'C' or 'c', op ( A ) = A'.

	Input, char TRANSB, specifies the form of op ( B ) to be used in
	the matrix multiplication as follows:
	'N' or 'n', op ( B ) = B.
	'T' or 't', op ( B ) = B'.
	'C' or 'c', op ( B ) = B'.

	Input, int M, the number of rows of the  matrix op ( A ) and of the
	matrix C.  0 <= M.

	Input, int N, the number  of columns of the matrix op ( B ) and the
	number of columns of the matrix C.  0 <= N.

	Input, int K, the number of columns of the matrix op ( A ) and the
	number of rows of the matrix op ( B ).  0 <= K.

	Input, float ALPHA, the scalar multiplier
	for op ( A ) * op ( B ).

	Input, float A(LDA,KA), where:
	if TRANSA is 'N' or 'n', KA is equal to K, and the leading M by K
	part of the array contains A;
	if TRANSA is not 'N' or 'n', then KA is equal to M, and the leading
	K by M part of the array must contain the matrix A.

	Input, int LDA, the first dimension of A as declared in the calling
	routine.  When TRANSA = 'N' or 'n' then LDA must be at least max ( 1, M ),
	otherwise LDA must be at least max ( 1, K ).

	Input, float B(LDB,KB), where:
	if TRANSB is 'N' or 'n', kB is N, and the leading K by N
	part of the array contains B;
	if TRANSB is not 'N' or 'n', then KB is equal to K, and the leading
	N by K part of the array must contain the matrix B.

	Input, int LDB, the first dimension of B as declared in the calling
	routine.  When TRANSB = 'N' or 'n' then LDB must be at least max ( 1, K ),
	otherwise LDB must be at least max ( 1, N ).

	Input, float BETA, the scalar multiplier for C.

	Input/output, float C(LDC,N).
	Before entry, the leading M by N part of this array must contain the
	matrix C, except when BETA is 0.0, in which case C need not be set
	on entry.
	On exit, the array C is overwritten by the M by N matrix
	alpha * op ( A ) * op ( B ) + beta * C.

	Input, int LDC, the first dimension of C as declared in the calling
	routine.  max ( 1, M ) <= LDC.
	*/
{
	int i;
	int info;
	int j;
	int l;
	int ncola;
	int nrowa;
	int nrowb;
	int nota;
	int notb;
	float temp;
	/*
	Set NOTA and NOTB as true if A and B respectively are not
	transposed and set NROWA, NCOLA and NROWB as the number of rows
	and columns of A and the number of rows of B respectively.
	*/
	nota = ((transa == 'N') || (transa == 'n'));

	if (nota)
	{
		nrowa = m;
		ncola = k;
	}
	else
	{
		nrowa = k;
		ncola = m;
	}

	notb = ((transb == 'N') || (transb == 'n'));

	if (notb)
	{
		nrowb = k;
	}
	else
	{
		nrowb = n;
	}
	/*
	Test the input parameters.
	*/
	info = 0;

	if (!(transa == 'N' || transa == 'n' ||
		transa == 'C' || transa == 'c' ||
		transa == 'T' || transa == 't'))
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input TRANSA had illegal value.\n");
		exit(1);
	}

	if (!(transb == 'N' || transb == 'n' ||
		transb == 'C' || transb == 'c' ||
		transb == 'T' || transb == 't'))
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input TRANSB had illegal value.\n");
		exit(1);
	}

	if (m < 0)
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input M had illegal value.\n");
		exit(1);
	}

	if (n < 0)
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input N had illegal value.\n");
		exit(1);
	}

	if (k  < 0)
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input K had illegal value.\n");
		exit(1);
	}

	if (lda < i4_max(1, nrowa))
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input LDA had illegal value.\n");
		exit(1);
	}

	if (ldb < i4_max(1, nrowb))
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input LDB had illegal value.\n");
		exit(1);
	}

	if (ldc < i4_max(1, m))
	{
		fprintf(stderr, "\n");
		fprintf(stderr, "DGEMM - Fatal error!\n");
		fprintf(stderr, "  Input LDC had illegal value.\n");
		exit(1);
	}
	/*
	Quick return if possible.
	*/
	if (m == 0)
	{
		return;
	}

	if (n == 0)
	{
		return;
	}

	if ((alpha == 0.0 || k == 0) && (beta == 1.0))
	{
		return;
	}
	/*
	And if alpha is 0.0.
	*/
	if (alpha == 0.0)
	{
		if (beta == 0.0)
		{
			for (j = 0; j < n; j++)
			{
				for (i = 0; i < m; i++)
				{
					c[i + j*ldc] = 0.0;
				}
			}
		}
		else
		{
			for (j = 0; j < n; j++)
			{
				for (i = 0; i < m; i++)
				{
					c[i + j*ldc] = beta * c[i + j*ldc];
				}
			}
		}
		return;
	}
	/*
	Start the operations.
	*/
	if (notb)
	{
		/*
		Form  C := alpha*A*B + beta*C.
		*/
		if (nota)
		{
			for (j = 0; j < n; j++)
			{
				if (beta == 0.0)
				{
					for (i = 0; i < m; i++)
					{
						c[i + j*ldc] = 0.0;
					}
				}
				else if (beta != 1.0)
				{
					for (i = 0; i < m; i++)
					{
						c[i + j*ldc] = beta * c[i + j*ldc];
					}
				}

				for (l = 0; l < k; l++)
				{
					if (b[l + j*ldb] != 0.0)
					{
						temp = alpha * b[l + j*ldb];
						for (i = 0; i < m; i++)
						{
							c[i + j*ldc] = c[i + j*ldc] + temp * a[i + l*lda];
						}
					}
				}

			}
		}
		/*
		Form  C := alpha*A'*B + beta*C
		*/
		else
		{
			for (j = 0; j < n; j++)
			{
				for (i = 0; i < m; i++)
				{
					temp = 0.0;
					for (l = 0; l < k; l++)
					{
						temp = temp + a[l + i*lda] * b[l + j*ldb];
					}

					if (beta == 0.0)
					{
						c[i + j*ldc] = alpha * temp;
					}
					else
					{
						c[i + j*ldc] = alpha * temp + beta * c[i + j*ldc];
					}
				}
			}
		}
	}
	/*
	Form  C := alpha*A*B' + beta*C
	*/
	else
	{
		if (nota)
		{
			for (j = 0; j < n; j++)
			{
				if (beta == 0.0)
				{
					for (i = 0; i < m; i++)
					{
						c[i + j*ldc] = 0.0;
					}
				}
				else if (beta != 1.0)
				{
					for (i = 0; i < m; i++)
					{
						c[i + j*ldc] = beta * c[i + j*ldc];
					}
				}

				for (l = 0; l < k; l++)
				{
					if (b[j + l*ldb] != 0.0)
					{
						temp = alpha * b[j + l*ldb];
						for (i = 0; i < m; i++)
						{
							c[i + j*ldc] = c[i + j*ldc] + temp * a[i + l*lda];
						}
					}
				}
			}
		}
		/*
		Form  C := alpha*A'*B' + beta*C
		*/
		else
		{
			for (j = 0; j < n; j++)
			{
				for (i = 0; i < m; i++)
				{
					temp = 0.0;
					for (l = 0; l < k; l++)
					{
						temp = temp + a[l + i*lda] * b[j + l*ldb];
					}
					if (beta == 0.0)
					{
						c[i + j*ldc] = alpha * temp;
					}
					else
					{
						c[i + j*ldc] = alpha * temp + beta * c[i + j*ldc];
					}
				}
			}
		}
	}

	return;
}

#include <iostream>

void CPU_sgemm(float* C, float* A, float* B, int HA, int WA, int WB)
{
	int rowA = HA, colA = WA, rowB = WA, colB = WB, rowC = HA;
	float alpha = 1, beta = 0;

	//  C := alpha*op( A )*op( B ) + beta*C
	sgemm('n', 'n', rowA, colB, colA, alpha, A, rowA, B, rowB, beta, C, rowC);

}

void CPU_dgemm(double* C, double* A, double* B, int HA, int WA, int WB)
{
	int rowA = HA, colA = WA, rowB = WA, colB = WB, rowC = HA;
	double alpha = 1, beta = 0;

	//  C := alpha*op( A )*op( B ) + beta*C
	dgemm('n', 'n', rowA, colB, colA, alpha, A, rowA, B, rowB, beta, C, rowC);

}


