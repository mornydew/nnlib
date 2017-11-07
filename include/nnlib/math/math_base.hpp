#ifndef MATH_MATH_BASE_HPP
#define MATH_MATH_BASE_HPP

#include "../core/error.hpp"
#include "../core/tensor.hpp"

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
	static void vFill(Tensor<T> &x, T alpha)
	{
		NNAssertEquals(x.dims(), 1, "Expected a vector!");
		vFill(x.ptr(), x.size(), x.stride(0), alpha);
	}
	
	/// x[i] = alpha
	static void vFill(T *x, size_t n, size_t s, T alpha)
	{
		for(size_t i = 0; i < n; ++i)
			x[i * s] = alpha;
	}
	
	/// x[i] *= alpha
	static void vScale(Tensor<T> &x, T alpha)
	{
		NNAssertEquals(x.dims(), 1, "Expected a vector!");
		vScale(x.ptr(), x.size(), x.stride(0), alpha);
	}
	
	/// x[i] *= alpha
	static void vScale(T *x, size_t n, size_t s, T alpha)
	{
		for(size_t i = 0; i < n; ++i)
			x[i * s] *= alpha;
	}
	
	// MARK: Matrix/scalar operations
	
	/// A[i][j] = alpha
	static void mFill(Tensor<T> &A, T alpha)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		mFill(A.ptr(), A.size(0), A.size(1), A.stride(0), alpha);
	}
	
	/// A[i][j] = alpha
	static void mFill(T *A, size_t r, size_t c, size_t ld, T alpha)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				A[i * ld + j] = alpha;
	}
	
	/// A[i][j] *= alpha
	static void mScale(Tensor<T> &A, T alpha)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		mScale(A.ptr(), A.size(0), A.size(1), A.stride(0), alpha);
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
	static void vAdd_v(const Tensor<T> &x, Tensor<T> &y, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(x.dims(), 1, "Expected a vector!");
		NNAssertEquals(y.dims(), 1, "Expected a vector!");
		NNAssertEquals(x.size(), y.size(), "Incompatible operands!");
		vAdd_v(x.ptr(), x.size(), x.stride(0), y.ptr(), y.stride(0), alpha, beta);
	}
	
	/// y = alpha * x + beta * y
	static void vAdd_v(const T *x, size_t n, size_t sx, T *y, size_t sy, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < n; ++i)
			y[i * sy] = alpha * x[i * sx] + beta * y[i * sy];
	}
	
	// MARK: Matrix/Vector operations
	
	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &A, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(x.dims(), 1, "Expected a vector!");
		NNAssertEquals(y.dims(), 1, "Expected a vector!");
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(A.size(0), x.size(), "Incompatible operands!");
		NNAssertEquals(A.size(1), y.size(), "Incompatible operands!");
		mAdd_vv(x.ptr(), x.size(), x.stride(0), y.ptr(), y.size(), y.stride(0), A.ptr(), A.stride(0), alpha, beta);
	}
	
	/// A = alpha * x <*> y + beta * A, <*> = outer product
	static void mAdd_vv(const T *x, size_t r, size_t sx, const T *y, size_t c, size_t sy, T *A, size_t lda, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				A[i * lda + j] = alpha * x[i * sx] * y[j * sy] + beta * A[i * lda + j];
	}
	
	/// y = alpha * A * x^T + beta * y
	static void vAdd_mv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(x.dims(), 1, "Expected a vector!");
		NNAssertEquals(y.dims(), 1, "Expected a vector!");
		NNAssertEquals(A.size(1), x.size(0), "Incompatible operands!");
		NNAssertEquals(A.size(0), y.size(0), "Incompatible operands!");
		NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
		vAdd_mv(A.ptr(), A.size(0), A.size(1), A.stride(0), x.ptr(), x.stride(0), y.ptr(), y.stride(0), alpha, beta);
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
	static void vAdd_mtv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(x.dims(), 1, "Expected a vector!");
		NNAssertEquals(y.dims(), 1, "Expected a vector!");
		NNAssertEquals(A.size(0), x.size(0), "Incompatible operands!");
		NNAssertEquals(A.size(1), y.size(0), "Incompatible operands!");
		NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
		vAdd_mtv(A.ptr(), A.size(0), A.size(1), A.stride(0), x.ptr(), x.stride(0), y.ptr(), y.stride(0), alpha, beta);
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
	static void mAdd_m(const Tensor<T> &A, Tensor<T> &B, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(A.shape(), B.shape(), "Incompatible operands!");
		mAdd_m(A.ptr(), A.size(0), A.size(1), A.stride(0), ptr(), stride(0), alpha, beta);
	}
	
	/// B = alpha * A + beta * B
	static void mAdd_m(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				B[i * ldb + j] = alpha * A[i * lda + j] + beta * B[i * ldb + j];
	}
	
	/// B = alpha * A^T + beta * B
	static void mAdd_mt(const Tensor<T> &A, Tensor<T> &B, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(A.size(0), B.size(1), "Incompatible operands!");
		NNAssertEquals(A.size(1), B.size(0), "Incompatible operands!");
		mAdd_mt(A.ptr(), A.size(0), A.size(1), A.stride(0), ptr(), stride(0), alpha, beta);
	}
	
	/// B = alpha * A^T + beta * B
	static void mAdd_mt(const T *A, size_t r, size_t c, size_t lda, T *B, size_t ldb, T alpha = 1, T beta = 1)
	{
		for(size_t i = 0; i < r; ++i)
			for(size_t j = 0; j < c; ++j)
				B[i * ldb + j] = alpha * A[j * lda + i] + beta * B[i * ldb + j];
	}
	
	/// C = alpha * A * B + beta * C
	static void mAdd_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(B.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(C.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(A.size(0), C.size(0), "Incompatible operands!");
		NNAssertEquals(B.size(1), C.size(1), "Incompatible operands!");
		NNAssertEquals(A.size(1), B.size(0), "Incompatible operands!");
		mAdd_mm(A.size(0), B.size(1), A.size(1), A.ptr(), A.stride(0), B.ptr(), B.stride(0), C.ptr(), C.stride(0), alpha, beta);
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
	static void mAdd_mtm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(B.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(C.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(A.size(1), C.size(0), "Incompatible operands!");
		NNAssertEquals(B.size(1), C.size(1), "Incompatible operands!");
		NNAssertEquals(A.size(0), B.size(0), "Incompatible operands!");
		mAdd_mtm(A.size(1), B.size(1), A.size(0), A.ptr(), A.stride(0), B.ptr(), B.stride(0), C.ptr(), C.stride(0), alpha, beta);
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
	static void mAdd_mmt(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, T alpha = 1, T beta = 1)
	{
		NNAssertEquals(A.dims(), 2, "Expected a matrix!");
		NNAssertEquals(B.dims(), 2, "Expected a matrix!");
		NNAssertEquals(C.dims(), 2, "Expected a matrix!");
		NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(B.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(C.stride(1), 1, "Expected a contiguous matrix!");
		NNAssertEquals(A.size(0), C.size(0), "Incompatible operands!");
		NNAssertEquals(B.size(0), C.size(1), "Incompatible operands!");
		NNAssertEquals(A.size(1), B.size(1), "Incompatible operands!");
		mAdd_mmt(A.size(0), B.size(0), A.size(1), A.ptr(), A.stride(0), B.ptr(), B.stride(0), C.ptr(), C.stride(0), alpha, beta);
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
