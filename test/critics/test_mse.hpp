#ifndef TEST_MSE
#define TEST_MSE

#include "nnlib/critics/mse.hpp"
using namespace nnlib;

void TestMSE()
{
	Storage<size_t> shape = { 5, 1 };
	Tensor<> inp = Tensor<>({  1,  2,  3,  4,  5 }).resize(shape);
	Tensor<> tgt = Tensor<>({  2,  4,  6,  8,  0 }).resize(shape);
	Tensor<> sqd = Tensor<>({  1,  4,  9, 16, 25 }).resize(shape);
	Tensor<> dif = Tensor<>({ -2, -4, -6, -8, 10 }).resize(shape);
	MSE<> critic(false);
	
	double mse = critic.forward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.sum()) < 1e-12, "MSE<>::forward with no average failed!");
	
	critic.average(true);
	mse = critic.forward(inp, tgt);
	NNHardAssert(fabs(mse - sqd.mean()) < 1e-12, "MSE<>::forward with average failed!");
	
	critic.average(false);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().add(dif, -1).square().sum() < 1e-12, "MSE<>::backward with no average failed!");
	
	critic.average(true);
	critic.backward(inp, tgt);
	NNHardAssert(critic.inGrad().add(dif.scale(1.0 / dif.size()), -1).square().sum() < 1e-12, "MSE<>::backward with average failed!");
}

#endif
