#ifndef OPT_SGD_HPP
#define OPT_SGD_HPP

#include "optimizer.hpp"
#include "../math/tensor_math.hpp"

namespace nnlib
{

template <typename T = double>
class SGD : public Optimizer<T>
{
using Optimizer<T>::m_model;
using Optimizer<T>::m_critic;
public:
	SGD(Module<T> &model, Critic<T> &critic) :
		Optimizer<T>(model, critic),
		m_parameters(model.params()),
		m_grads(model.grad()),
		m_velocity(m_grads.size(0)),
		m_learningRate(0.001),
		m_momentum(0)
	{
		m_velocity.fill(0.0);
	}
	
	SGD &learningRate(T learningRate)
	{
		m_learningRate = learningRate;
		return *this;
	}
	
	T learningRate() const
	{
		return m_learningRate;
	}
	
	SGD &momentum(T momentum)
	{
		m_momentum = momentum;
		return *this;
	}
	
	T momentum() const
	{
		return m_momentum;
	}
	
	// MARK: Critic methods
	
	/// Perform a single step of training given an input and a target.
	virtual SGD &step(const Tensor<T> &input, const Tensor<T> &target) override
	{
		// calculate gradient
		m_grads.fill(0);
		m_model.backward(input, m_critic.backward(m_model.forward(input), target));
		
		if(m_momentum)
		{
			// apply momentum
			m_velocity *= m_momentum;
			vAdd_v(m_grads, m_velocity);
			
			// Nesterov step
			vAdd_v(m_velocity, m_grads, m_momentum);
		}
		
		// update parameters
		vAdd_v(m_grads, m_parameters, -m_learningRate);
		
		return *this;
	}
	
private:
	Tensor<T> &m_parameters;
	Tensor<T> &m_grads;
	Tensor<T> m_velocity;
	T m_learningRate;
	T m_momentum;
};

}

#endif
