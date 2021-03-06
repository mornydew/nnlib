#ifndef TEST_DROPOUT_H
#define TEST_DROPOUT_H

#include "nnlib/nn/dropout.hpp"
#include "test_module.hpp"
using namespace nnlib;

/// \brief Test the dropout module.
///
/// Dropout depends on a stochastic process, so testing it is less straightforward.
/// We test by running it several times and making sure the statistics are approximately right.
void TestDropout()
{
	RandomEngine::seed(0);
	
	Tensor<> ones = Tensor<>(4, 25).ones();
	double p = 0.75, sum1 = 0, sum2 = 0;
	size_t c = 100;
	
	Dropout<> module(p);
	NNAssertEquals(module.dropProbability(), p, "Dropout::Dropout failed!");
	
	for(size_t i = 0; i < c; ++i)
	{
		sum1 += module.forward(ones).sum() / c;
		sum2 += module.backward(ones, ones).sum() / c;
	}
	
	NNAssertAlmostEquals(sum1, c * (1 - p), 2, "Dropout::forward failed! (Note: this may be due to the random process).");
	NNAssertEquals(sum1, sum2, "Dropout::backward failed!");
	
	module.dropProbability(p = 0.5).training(false);
	sum1 = sum2 = 0;
	
	for(size_t i = 0; i < c; ++i)
	{
		sum1 += module.forward(ones).sum() / c;
		sum2 += module.backward(ones, ones).sum() / c;
	}
	
	NNAssertAlmostEquals(sum1, c * (1 - p), 2, "Dropout::forward (inference) failed! (Note: this may be due to the random process).");
	NNAssertEquals(sum1, sum2, "Dropout::backward (inference) failed!");
	
	module.state().fill(0);
	NNAssertAlmostEquals(module.output().sum(), 0, 1e-12, "Dropout::state failed!");
	
	TestModule("Dropout", module, ones);
}

#endif
