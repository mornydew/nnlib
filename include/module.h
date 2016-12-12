#ifndef MODULE_H
#define MODULE_H

namespace nnlib
{

template <typename T>
class Module
{
public:
	virtual void forward(const Matrix<T> &inputs) = 0;
	virtual void backward(const Matrix<T> &inputs, const Matrix<T> &blame) = 0;
	
	virtual Vector<Matrix<T> *> parameters()
	{
		return Vector<Matrix<T> *>();
	}
	
	virtual Vector<Matrix<T> *> blame()
	{
		return Vector<Matrix<T> *>();
	}
};

}

#endif