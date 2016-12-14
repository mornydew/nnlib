#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include "container.h"

namespace nnlib
{

template <typename T>
class Sequential : public Container<T>
{
using Container<T>::m_components;
public:
	virtual void batch(size_t size) override
	{
		for(Module<T> *layer : m_components)
			layer->batch(size);
	}
	
	virtual Matrix<T> &forward(const Matrix<T> &inputs) override
	{
		const Matrix<T> *inps = &inputs;
		for(Module<T> *layer : m_components)
			inps = &layer->forward(*inps);
		return *const_cast<Matrix<T> *>(inps);
	}
	
	virtual Matrix<T> &backward(const Matrix<T> &inputs, const Matrix<T> &blame) override
	{
		const Matrix<T> *blam = &blame;
		for(size_t i = m_components.size() - 1; i > 0; --i)
			blam = &m_components[i]->backward(m_components[i - 1]->output(), *blam);
		return m_components[0]->backward(inputs, *blam);
	}
	
	virtual Matrix<T> &output() override
	{
		return m_components[m_components.size() - 1]->output();
	}
	
	virtual Matrix<T> &inputBlame() override
	{
		return m_components[0]->inputBlame();
	}
	
	virtual Vector<Tensor<T> *> parameters() override
	{
		Vector<Tensor<T> *> params;
		for(Module<T> *layer : m_components)
			for(Tensor<T> *t : layer->parameters())
				params.push_back(t);
		return params;
	}
	
	virtual Vector<Tensor<T> *> blame() override
	{
		Vector<Tensor<T> *> blam;
		for(Module<T> *layer : m_components)
			for(Tensor<T> *t : layer->blame())
				blam.push_back(t);
		return blam;
	}
};

}

#endif
