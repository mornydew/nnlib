#ifndef MATH_TENSOR_MATH_HPP
#define MATH_TENSOR_MATH_HPP

#include "math.hpp"
#include "../core/tensor.hpp"

namespace nnlib
{

template <typename T, typename U>
Tensor<T> &operator*=(Tensor<T> &x, U alpha)
{
	switch(x.dims())
	{
	case 1:
		Math<T>::vScale(x.ptr(), x.size(), x.stride(0), alpha);
		break;
	case 2:
		Math<T>::mScale(x.ptr(), x.size(0), x.size(1), x.stride(0), alpha);
		break;
	default:
		for(auto &v : x)
			v *= alpha;
	}
	return x;
}

template <typename T, typename U>
Tensor<T> operator*(const Tensor<T> &x, U alpha)
{
	Tensor<T> copy = x.copy();
	return copy *= alpha;
}

template <typename T, typename U>
Tensor<T> &operator/=(Tensor<T> &x, U alpha)
{
	return x *= (1.0 / alpha);
}

template <typename T, typename U>
Tensor<T> operator/(const Tensor<T> &x, U alpha)
{
	return x * (1.0 / alpha);
}

template <typename T, typename U = T>
Tensor<T> &add(Tensor<T> &y, const Tensor<T> &x, U alpha = 1)
{
	NNAssertEquals(y.shape(), x.shape(), "Incompatible operands!");
	
	switch(y.dims())
	{
	case 1:
		Math<T>::vAdd_v(x.ptr(), x.size(), x.stride(0), y.ptr(), y.stride(0), alpha);
		break;
	case 2:
		Math<T>::mAdd_m(x.ptr(), x.size(0), x.size(1), x.stride(0), y.ptr(), y.stride(0), alpha);
		break;
	default:
		auto i = x.begin();
		for(auto &j : y)
			j += alpha * *i, ++i;
	}
	
	return y;
}

template <typename T>
Tensor<T> &operator+=(Tensor<T> &y, const Tensor<T> &x)
{
	return add(y, x);
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &y, const Tensor<T> &x)
{
	Tensor<T> copy = y.copy();
	return add(copy, x);
}

template <typename T>
Tensor<T> &operator-=(Tensor<T> &y, const Tensor<T> &x)
{
	return add(y, x, -1);
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &y, const Tensor<T> &x)
{
	Tensor<T> copy = y.copy();
	return add(copy, x, -1);
}

/// y = alpha * x + beta * y
template <typename T, typename U, typename V = double, typename W = double>
void vAdd_v(const Tensor<T> &x, U &&y, V alpha = 1, W beta = 1)
{
	NNAssertEquals(x.dims(), 1, "Expected a vector!");
	NNAssertEquals(y.dims(), 1, "Expected a vector!");
	NNAssertEquals(x.size(), y.size(), "Incompatible operands!");
	Math<T>::vAdd_v(x.ptr(), x.size(), x.stride(0), y.ptr(), y.stride(0), alpha, beta);
}

// MARK: Matrix/Vector operations

/// A = alpha * x <*> y + beta * A, <*> = outer product
template <typename T, typename U = T, typename V = T>
void mAdd_vv(const Tensor<T> &x, const Tensor<T> &y, Tensor<T> &A, U alpha = 1, V beta = 1)
{
	NNAssertEquals(x.dims(), 1, "Expected a vector!");
	NNAssertEquals(y.dims(), 1, "Expected a vector!");
	NNAssertEquals(A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(A.size(0), x.size(), "Incompatible operands!");
	NNAssertEquals(A.size(1), y.size(), "Incompatible operands!");
	Math<T>::mAdd_vv(x.ptr(), x.size(), x.stride(0), y.ptr(), y.size(), y.stride(0), A.ptr(), A.stride(0), alpha, beta);
}

/// y = alpha * A * x^T + beta * y
template <typename T, typename U = T, typename V = T>
void vAdd_mv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, U alpha = 1, V beta = 1)
{
	NNAssertEquals(A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(x.dims(), 1, "Expected a vector!");
	NNAssertEquals(y.dims(), 1, "Expected a vector!");
	NNAssertEquals(A.size(1), x.size(0), "Incompatible operands!");
	NNAssertEquals(A.size(0), y.size(0), "Incompatible operands!");
	NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
	Math<T>::vAdd_mv(A.ptr(), A.size(0), A.size(1), A.stride(0), x.ptr(), x.stride(0), y.ptr(), y.stride(0), alpha, beta);
}

/// y = alpha * A^T * x^T + beta * y
template <typename T, typename U = T, typename V = T>
void vAdd_mtv(const Tensor<T> &A, const Tensor<T> &x, Tensor<T> &y, U alpha = 1, V beta = 1)
{
	NNAssertEquals(A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(x.dims(), 1, "Expected a vector!");
	NNAssertEquals(y.dims(), 1, "Expected a vector!");
	NNAssertEquals(A.size(0), x.size(0), "Incompatible operands!");
	NNAssertEquals(A.size(1), y.size(0), "Incompatible operands!");
	NNAssertEquals(A.stride(1), 1, "Expected a contiguous matrix!");
	Math<T>::vAdd_mtv(A.ptr(), A.size(0), A.size(1), A.stride(0), x.ptr(), x.stride(0), y.ptr(), y.stride(0), alpha, beta);
}

// MARK: Matrix/Matrix operations

/// B = alpha * A + beta * B
template <typename T, typename U = T, typename V = T>
void mAdd_m(const Tensor<T> &A, Tensor<T> &B, U alpha = 1, V beta = 1)
{
	NNAssertEquals(A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(B.dims(), 2, "Expected a matrix!");
	NNAssertEquals(A.shape(), B.shape(), "Incompatible operands!");
	Math<T>::mAdd_m(A.ptr(), A.size(0), A.size(1), A.stride(0), B.ptr(), B.stride(0), alpha, beta);
}

/// B = alpha * A^T + beta * B
template <typename T, typename U = T, typename V = T>
void mAdd_mt(const Tensor<T> &A, Tensor<T> &B, U alpha = 1, V beta = 1)
{
	NNAssertEquals(A.dims(), 2, "Expected a matrix!");
	NNAssertEquals(B.dims(), 2, "Expected a matrix!");
	NNAssertEquals(A.size(0), B.size(1), "Incompatible operands!");
	NNAssertEquals(A.size(1), B.size(0), "Incompatible operands!");
	Math<T>::mAdd_mt(A.ptr(), A.size(0), A.size(1), A.stride(0), B.ptr(), B.stride(0), alpha, beta);
}

/// C = alpha * A * B + beta * C
template <typename T, typename U = T, typename V = T>
void mAdd_mm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, U alpha = 1, V beta = 1)
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
	Math<T>::mAdd_mm(A.size(0), B.size(1), A.size(1), A.ptr(), A.stride(0), B.ptr(), B.stride(0), C.ptr(), C.stride(0), alpha, beta);
}

/// C = alpha * A^T * B + beta * C
template <typename T, typename U = T, typename V = T>
void mAdd_mtm(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, U alpha = 1, V beta = 1)
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
	Math<T>::mAdd_mtm(A.size(1), B.size(1), A.size(0), A.ptr(), A.stride(0), B.ptr(), B.stride(0), C.ptr(), C.stride(0), alpha, beta);
}

/// C = alpha * A * B^T + beta * C
template <typename T, typename U = T, typename V = T>
void mAdd_mmt(const Tensor<T> &A, const Tensor<T> &B, Tensor<T> &C, U alpha = 1, V beta = 1)
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
	Math<T>::mAdd_mmt(A.size(0), B.size(0), A.size(1), A.ptr(), A.stride(0), B.ptr(), B.stride(0), C.ptr(), C.stride(0), alpha, beta);
}

}

#endif
