#ifndef MATH_MATH_BASE_HPP
#define MATH_MATH_BASE_HPP

#include "../core/error.hpp"

namespace nnlib
{

/// Math utility class with naive implementations.
/// May use a subclass to accelerate (i.e. use BLAS).
template <typename T>
class MathBase
{
public:
	// MARK: Vector/scalar operations
	
	/// x[i] = alpha
	static void vFill(T *x, size_t n, size_t s, T alpha)
	{
		for(size_t i = 0; i < n; ++i)
			x[i * s] = alpha;
	}
	
	/// x[i] *= alpha
	static void vScale(T *x, size_t n, size_t s, T alpha)
	{
		for(size_t i = 0; i < n; ++i)
			x[i * s] *= alpha;
	}
	
	// MARK: Matrix/scalar operations
	
	/// A[i][j] = alpha
	static void mFill(T *A, size_t r, size_t c, size_t ld, T alpha)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				A[i * ld + j] = alpha;
	}
	
	/// A[i][j] *= alpha
	static void mScale(T *A, size_t r, size_t c, size_t ld, T alpha)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				A[i * ld + j] *= alpha;
	}
	
	// MARK: Vector/Vector operations
	
	/// y = alpha * x + beta * y
	static void vAdd_v(const T *x, size_t n, size_t sx, T *y, size_t sy, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < n; ++i)
			y[i * sy] = alpha * x[i * sx] + beta * y[i * sy];
	}
	
	// MARK: Matrix/Vector operations
	
	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(const T *x, size_t r, size_t sx, const T *y, size_t c, size_t sy, T *A, size_t lda, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				A[i * lda + j] = alpha * x[i * sx] * y[j * sy] + beta * A[i * lda + j];
	}
	
	
	/// y = alpha * A * x^T + beta * y
	static void vAdd_mv(const T *A, size_t ra, size_t ca, size_t lda, const T *x, size_t sx, T *y, size_t sy, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < ra; ++i)
		{
			T &v = y[i * sy];
			v *= beta;
			for(size_t j = 0; j < ca; ++j)
				v += alpha * A[i * lda + j] * x[j * sx];
		}
	}
	
	/// y = alpha * A^T * x^T + beta * y
	static void vAdd_mtv(const T *A, size_t ra, size_t ca, size_t lda, const T *x, size_t sx, T *y, size_t sy, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < ca; ++i)
		{
			T &v = y[i * sy];
			v *= beta;
			for(size_t j = 0; j < ra; ++j)
				v += alpha * A[j * lda + i] * x[j * sx];
		}
	}
	
	// MARK: Matrix/Matrix operations
	
	/// B = alpha * A + beta * B
	static void mAdd_m(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				B[i * ldb + j] = alpha * A[i * lda + j] + beta * B[i * ldb + j];
	}
	
	/// B = alpha * A^T + beta * B
	static void mAdd_mt(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				B[i * ldb + j] = alpha * A[j * lda + i] + beta * B[i * ldb + j];
	}
	
	/// C = alpha * A * B + beta * C
	static void mAdd_mm(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < M; ++i)
		{
			for(size_t j = 0; j < N; ++j)
			{
				T &v = C[i * ldc + j];
				v *= beta;
				for(size_t k = 0; k < K; ++k)
				{
					v += alpha * A[i * lda + k] * B[k * ldb + j];
				}
			}
		}
	}
	
	/// C = alpha * A^T * B + beta * C
	static void mAdd_mtm(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < M; ++i)
		{
			for(size_t j = 0; j < N; ++j)
			{
				T &v = C[i * ldc + j];
				v *= beta;
				for(size_t k = 0; k < K; ++k)
				{
					v += alpha * A[k * lda + i] * B[k * ldb + j];
				}
			}
		}
	}
	
	/// C = alpha * A * B^T + beta * C
	static void mAdd_mmt(size_t M, size_t N, size_t K, const T *A, size_t lda, const T *B, size_t ldb, T *C, size_t ldc, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < M; ++i)
		{
			for(size_t j = 0; j < N; ++j)
			{
				T &v = C[i * ldc + j];
				v *= beta;
				for(size_t k = 0; k < K; ++k)
				{
					v += alpha * A[i * lda + k] * B[j * ldb + k];
				}
			}
		}
	}
};

}

#endif
