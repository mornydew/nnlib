#ifndef TEST_RELU_H
#define TEST_RELU_H

#include "nnlib/nn/relu.hpp"
#include "test_map.hpp"
using namespace nnlib;

void TestReLU()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = inp.copy();
	out(0, 0) *= 0.5;
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = grd.copy();
	ing(0, 0) *= 0.5;
	
	ReLU<> map(0.5);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().copy().add(out, -1).square().sum() < 1e-9, "ReLU::forward failed!");
	NNAssert(map.inGrad().copy().add(ing, -1).square().sum() < 1e-9, "ReLU::backward failed!");
	
	TestMap("ReLU", map, inp);
}

#endif
