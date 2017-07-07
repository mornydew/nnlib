#ifndef TEST_TANH_H
#define TEST_TANH_H

#include "nnlib/nn/tanh.h"
#include "test_module.h"
using namespace nnlib;

void TestTanH()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({ -1.3, 1.0, 3.14 }).resize(1, 3);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({ 2, -3, 1 }).resize(1, 3);
	
	// Output, fixed given input
	Tensor<> out = Tensor<>({ -0.86172315931, 0.76159415595, 0.99626020494 }).resize(1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({ 0.5148663934, -1.25992302484, 0.00746560404 }).resize(1, 3);
	
	// Begin test
	
	TanH<> map(3, 1);
	map.forward(inp);
	map.backward(inp, grd);
	
	NNAssert(map.output().addM(out, -1).square().sum() < 1e-9, "TanH::forward failed!");
	NNAssert(map.inGrad().addM(ing, -1).square().sum() < 1e-9, "TanH::backward failed!");
	
	map.inputs({ 3, 4 });
	NNAssert(map.inputs() == map.outputs(), "TanH::inputs failed to resize outputs!");
	
	map.outputs({ 12, 3 });
	NNAssert(map.inputs() == map.outputs(), "TanH::outputs failed to resize inputs!");
	
	bool ok = true;
	try
	{
		map.resize({ 3, 4 }, { 4, 3 });
		ok = false;
	}
	catch(const Error &e) {}
	NNAssert(ok, "TanH::resize allowed unequal inputs and outputs!");
	
	TestSerializationOfModule(map);
	TestModule(map);
}

#endif
