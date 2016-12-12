#ifndef MATRIX_H
#define MATRIX_H

#include "tensor.h"
#include "vector.h"

namespace nnlib
{

template <typename T>
class Matrix : public Tensor<T>
{
using Tensor<T>::m_ptr;
using Tensor<T>::m_size;
public:
	/// Element iterator
	class Iterator : public std::iterator<std::forward_iterator_tag, T>
	{
	public:
		Iterator(T *ptr, size_t cols, size_t ld)	: m_ptr(ptr), m_cols(cols), m_ld(ld), m_col(0) {}
		Iterator &operator++()						{ if(++m_col % m_cols == 0) { m_col = 0; m_ptr += m_ld; } return *this; }
		T &operator*()								{ return m_ptr[m_col]; }
		bool operator==(const Iterator &i)			{ return m_ptr == i.m_ptr && m_cols == i.m_cols && m_ld == i.m_ld && m_col == i.m_col; }
		bool operator!=(const Iterator &i)			{ return m_ptr != i.m_ptr || m_cols != i.m_cols || m_ld != i.m_ld || m_col != i.m_col; }
	private:
		T *m_ptr;
		size_t m_cols, m_ld, m_col;
	};
	
	// MARK: Constructors
	
	/// Create a matrix of size rows * cols.
	Matrix(size_t rows = 0, size_t cols = 0) : Tensor<T>(rows * cols), m_rows(rows), m_cols(cols), m_ld(cols)
	{
		fill(T());
	}
	
	/// Create a shallow copy of another matrix.
	Matrix(const Matrix &m) : Tensor<T>(m), m_rows(m.rows), m_cols(m.cols), m_ld(m.ld) {}
	
	// MARK: Element Manipulation
	
	void fill(const T &val)
	{
		std::fill(begin(), end(), val);
	}
	
	// MARK: Row and Column Access
	
	/// Get a vector looking at the ith row in the matrix.
	Vector<T> operator[](size_t i)
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	ConstVector<T> operator[](size_t i) const
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return ConstVector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	Vector<T> operator()(size_t i)
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return Vector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the ith row in the matrix.
	ConstVector<T> operator()(size_t i) const
	{
		NNAssert(i < m_rows, "Invalid Matrix row index!");
		return ConstVector<T>(*this, i * m_ld, m_cols, 1);
	}
	
	/// Get a vector looking at the jth column in the matrix.
	Vector<T> column(size_t j)
	{
		NNAssert(j < m_cols, "Invalid Matrix column index!");
		return Vector<T>(*this, j, m_rows, m_ld);
	}
	
	/// Get a vector looking at the jth column in the matrix.
	ConstVector<T> column(size_t j) const
	{
		NNAssert(j < m_cols, "Invalid Matrix column index!");
		return ConstVector<T>(*this, j, m_rows, m_ld);
	}
	
	// MARK: Element Access
	
	T &operator()(size_t i, size_t j)
	{
		NNAssert(i < m_rows && j < m_cols, "Invalid Matrix indices!");
		return m_ptr[i * m_ld + j];
	}
	
	const T &operator()(size_t i, size_t j) const
	{
		NNAssert(i < m_rows && j < m_cols, "Invalid Matrix indices!");
		return m_ptr[i * m_ld + j];
	}
	
	// MARK: Iterators
	
	Iterator begin()
	{
		return Iterator(m_ptr, m_cols, m_ld);
	}
	
	Iterator end()
	{
		return Iterator(m_ptr + m_ld * m_rows, m_cols, m_ld);
	}
private:
	size_t m_rows, m_cols;	///< the dimensions of this Matrix
	size_t m_ld;			///< the length of the leading dimension (i.e. row stride)
};

}

#endif