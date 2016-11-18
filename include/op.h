#ifndef OP_H
#define OP_H

#include "tensor.h"

namespace nnlib
{

template <typename T>
class Operation
{
public:
	virtual ~Operation() {}
	virtual void assign(T &dest) const = 0;
};

template <typename T, typename U = T>
class UnaryOperation : public Operation<T>
{
public:
	UnaryOperation(const U &target) : m_target(target) {}
protected:
	const U &m_target;
};

template <typename T>
class OperationNeg : public UnaryOperation<T>
{
using UnaryOperation<T>::UnaryOperation;
using UnaryOperation<T>::m_target;
public:
	virtual void assign(T &dest) const override
	{
		dest = m_target;
		dest *= -1;
	}
};

/// Unary negation operator overload.
template <typename T>
OperationNeg<T> operator-(const T &target)
{
	return OperationNeg<T>(target);
}

template <typename T, typename U = T, typename V = T>
class BinaryOperation : public Operation<T>
{
public:
	BinaryOperation(const U &lhs, const V &rhs) : m_lhs(lhs), m_rhs(rhs) {}
protected:
	const U &m_lhs;
	const V &m_rhs;
};

template <typename T>
class OperationAdd : public BinaryOperation<T>
{
using BinaryOperation<T>::BinaryOperation;
using BinaryOperation<T>::m_lhs;
using BinaryOperation<T>::m_rhs;
public:
	virtual void assign(T &dest) const override
	{
		dest = m_lhs;
		dest += m_rhs;
	}
};

/// Addition operator overload.
template <typename T>
OperationAdd<T> operator+(const T &lhs, const T &rhs)
{
	return OperationAdd<T>(lhs, rhs);
}

template <typename T>
class OperationSub : public BinaryOperation<T>
{
using BinaryOperation<T>::BinaryOperation;
using BinaryOperation<T>::m_lhs;
using BinaryOperation<T>::m_rhs;
public:
	virtual void assign(T &dest) const override
	{
		dest = m_lhs;
		dest -= m_rhs;
	}
};

/// Subtraction operator overload.
template <typename T>
OperationSub<T> operator-(const T &lhs, const T &rhs)
{
	return OperationSub<T>(lhs, rhs);
}

}

#endif
