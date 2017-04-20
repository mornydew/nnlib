#ifndef SSE_H
#define SSE_H

#include "critic.h"

namespace nnlib
{

/// Sum squared error critic.
template <typename T = double>
class SSE : public Critic<double>
{
public:
	template <typename M>
	SSE(const M &model) :
		m_output(model.outputs(), true),
		m_inGrad(model.outputs(), true)
	{}
	
	SSE(const Storage<size_t> &shape) :
		m_output(shape, true),
		m_inGrad(shape, true)
	{}
	
	/// Loss = sum_i( 1/2 * (target(i) - input(i))^2 )
	virtual Tensor<T> &forward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_output.shape(), "Incompatible operands to SSE!");
		auto inp = input.begin(), tar = target.begin();
		T diff;
		for(auto out = m_output.begin(), end = m_output.end(); out != end; ++inp, ++tar, ++out)
		{
			diff = *tar - *inp;
			*out = 0.5 * diff * diff;
		}
		return m_output;
	}
	
	/// Grad(Loss) = sum_i( target(i) - input(i) )
	virtual Tensor<T> &backward(const Tensor<T> &input, const Tensor<T> &target) override
	{
		NNAssert(input.shape() == target.shape() && input.shape() == m_output.shape(), "Incompatible operands to SSE!");
		auto inp = input.begin(), tar = target.begin();
		for(auto grad = m_inGrad.begin(), end = m_inGrad.end(); grad != end; ++inp, ++tar, ++grad)
		{
			*grad = *tar - *inp;
		}
		return m_inGrad;
	}
	
	/// Output buffer (the loss).
	virtual Tensor<T> &output() override
	{
		return m_output;
	}
	
	/// Input gradient buffer.
	virtual Tensor<T> &inGrad() override
	{
		return m_inGrad;
	}

private:
	Tensor<T> m_output;	///< The loss.
	Tensor<T> m_inGrad;	///< The gradient of the loss w.r.t. the input.
};

}

#endif