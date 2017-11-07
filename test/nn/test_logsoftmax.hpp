#ifndef TEST_LOGSOFTMAX_H
#define TEST_LOGSOFTMAX_H

#include "nnlib/nn/logsoftmax.hpp"
#include "test_module.hpp"
using namespace nnlib;

void TestLogSoftMax()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -4, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ -4.56173148054, -2.26173148054, -0.12173148053 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 2.01044395977, -3.89583003975, 1.88538607998 }).resize(1, 3);
	
	// Begin test
	
	LogSoftMax<> map;
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().add(out, -1).square().sum() < 1e-9, "LogSoftMax::forward failed!");
	NNAssert(map.inGrad().add(ing, -1).square().sum() < 1e-9, "LogSoftMax::backward failed!");
	
	TestModule("LogSoftMax", map, inp);
}

#endif
