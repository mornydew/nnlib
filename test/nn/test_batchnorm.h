#ifndef TEST_BATCHNORM_H
#define TEST_BATCHNORM_H

#include "nnlib/nn/batchnorm.h"
using namespace nnlib;

void TestBatchNorm()
{
	Tensor<> inp = Tensor<>({
		 3,  6,  9,
		-1,  5,  4,
		12,  5, 11
	}).resize(3, 3);
	
	Tensor<> grad = Tensor<>({
		 2,  3,  4,
		-2,  0,  4,
		10,  2,  4
	}).resize(3, 3);
	
	Tensor<> inGrad = Tensor<>({
		 0.03596,  0.00000,  0,
		-0.02489, -2.12132,  0,
		-0.01106,  2.12132,  0
	}).resize(3, 3);
	
	BatchNorm<> bn(3, 3);
	
	bn.forward(inp);
	for(size_t i = 0; i < 3; ++i)
	{
		NNHardAssert(fabs(bn.output().select(1, i).mean()) < 1e-9, "BatchNorm::forward failed! Non-zero mean!");
		NNHardAssert(fabs(bn.output().select(1, i).variance() - 1) < 1e-9, "BatchNorm::forward failed! Non-unit variance!");
	}
	
	bn.backward(inp, grad);
	NNHardAssert(
		bn.grad().add(Tensor<>({ 14.9606, 2.82843, 0, 10, 5, 12 }), -1).square().sum() < 1e-9,
		"BatchNorm::backward failed! Wrong parameter gradient!"
	);
	
	NNHardAssert(
		bn.inGrad().add(inGrad, -1).square().sum() < 1e-9,
		"BatchNorm::backward failed! Wrong input gradient!"
	);
	
	BatchNorm<> *deserialized = nullptr;
	Archive::fromString((Archive::toString() << bn).str()) >> deserialized;
	NNHardAssert(
		deserialized != nullptr && bn.inputs() == deserialized->inputs() && bn.outputs() == deserialized->outputs(),
		"BatchNorm::save and/or BatchNorm::load failed!"
	);
	
	delete deserialized;
}

#endif
