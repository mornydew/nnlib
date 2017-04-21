#ifndef ADAM_H
#define ADAM_H

#include "optimizer.h"

namespace nnlib
{

template <template <typename> class M, template <typename> class C, typename T = double>
class Adam : public Optimizer<M, C, T>
{
using Optimizer<M, C, T>::m_model;
using Optimizer<M, C, T>::m_critic;
public:
	Adam(M<T> &model, C<T> &critic) :
		Optimizer<M, C, T>(model, critic),
		m_learningRate(0.001),
		m_beta1(0.9),
		m_beta2(0.999),
		m_normalize1(1),
		m_normalize2(1),
		m_steps(0)
	{
		m_parameters = Tensor<T>::flatten(model.parameters());
		m_grads = Tensor<T>::flatten(model.grad());
		m_velocity.resize(m_grads.size()).fill(0.0);
		m_meanSquare.resize(m_grads.size()).fill(0.0);
	}
	
	void reset()
	{
		m_steps = 0;
		m_normalize1 = 1;
		m_normalize2 = 1;
	}
	
	Adam &learningRate(T learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	T learningRate() const
	{
		return m_learningRate;
	}
	
	Adam &beta1(T beta1)
	{
		m_beta1 = beta1;
		return *this;
	}
	
	T beta1() const
	{
		return m_beta1;
	}
	
	Adam &beta2(T beta2)
	{
		m_beta2 = beta2;
		return *this;
	}
	
	T beta2() const
	{
		return m_beta2;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual void step(const Tensor<T> &input, const Tensor<T> &target) override
	{
		m_normalize1 *= m_beta1;
		m_normalize2 *= m_beta2;
		++m_steps;
		
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		// update momentum and mean square
		auto m = m_velocity.begin(), n = m_meanSquare.begin();
		for(T &g : m_grads)
		{
			*m *= m_beta1;
			*m += (1 - m_beta1) * g;
			*n *= m_beta2;
			*n += (1 - m_beta2) * g * g;
		}
		
		// update parameters
		T lr = m_learningRate / (1 - m_normalize1) * sqrt(1 - m_normalize2);
		m = m_velocity.begin(), n = m_meanSquare.begin();
		for(T &p : m_parameters)
		{
			p -= lr * *m / (sqrt(*n) + 1e-8);
			++m;
			++n;
		}
	}
	
private:
	Tensor<T> m_parameters;
	Tensor<T> m_grads;
	Tensor<T> m_velocity;
	Tensor<T> m_meanSquare;
	T m_learningRate;
	T m_beta1;
	T m_beta2;
	T m_normalize1;
	T m_normalize2;
	size_t m_steps;
};

}

#endif
