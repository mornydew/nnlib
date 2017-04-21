#ifndef RMSPROP_H
#define RMSPROP_H

#include "optimizer.h"

namespace nnlib
{

template <template <typename> class M, template <typename> class C, typename T = double>
class RMSProp : public Optimizer<M, C, T>
{
using Optimizer<M, C, T>::m_model;
using Optimizer<M, C, T>::m_critic;
public:
	RMSProp(M<T> &model, C<T> &critic) :
		Optimizer<M, C, T>(model, critic),
		m_learningRate(0.001),
		m_gamma(0.9)
	{
		m_parameters = Tensor<T>::flatten(model.parameters());
		m_grads = Tensor<T>::flatten(model.grad());
		m_meanSquare.resize(m_grads.size()).fill(0.0);
	}
	
	RMSProp &learningRate(T learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	T learningRate() const
	{
		return m_learningRate;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual void step(const Tensor<T> &input, const Tensor<T> &target) override
	{
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		for(size_t i = 0, end = m_grads.size(); i != end; ++i)
		{
			// update mean square
			m_meanSquare(i) = m_gamma * m_meanSquare(i) + (1 - m_gamma) * m_grads(i) * m_grads(i);
			
			// update parameters
			m_parameters(i) -= m_learningRate * m_grads(i) / (sqrt(m_meanSquare(i)) + 1e-8);
		}
	}
	
private:
	Tensor<T> m_parameters;
	Tensor<T> m_grads;
	Tensor<T> m_meanSquare;
	T m_learningRate;
	T m_gamma;
};

}

#endif
