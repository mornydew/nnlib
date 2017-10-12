#ifndef TEST_SEQUENCER_H
#define TEST_SEQUENCER_H

#include "nnlib/nn/sequencer.hpp"
#include "nnlib/nn/lstm.hpp"
#include "test_module.hpp"
using namespace nnlib;

void TestSequencer()
{
	// Input, arbitrary
	Tensor<> inp = Tensor<>({
		-5, 10,
		15, -20,
		3, 4
	}).resize(3, 1, 2);
	
	// Output gradient, arbitrary
	Tensor<> grd = Tensor<>({
		1, 2, 3,
		3, -4, 5,
		-3, 2, -7
	}).resize(3, 1, 3);
	
	// LSTM layer with specific weights and bias, arbitrary
	LSTM<> *lstm = new LSTM<>(2, 3);
	lstm->params().copy({
		// inpGateX: 9
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
		
		// inpGateY: 12
		1, 0, 1, 2, 1, 0, 1, 2, 3, 0.1, -0.1, 0.5,
		
		// inpGateH: 12
		2, 1, 2, 1, 2, 1, 2, 1, 2, 0.2, 0.2, 0.3,
		
		// fgtGateX: 9
		0.5, 0.4, 0.5, 0.3, 0.2, -0.1, 0.7, 0.8, -0.7,
		
		// fgtGateY: 12
		9, 8, 7, 6, 5, 6, 7, 8, 9, 8, 7, 7,
		
		// fgtGateH: 12
		1, 2, 1, 4, 1, 5, 1, 6, 1, 7, 1, 8,
		
		// inpModX: 9
		8, 6, 7, 5, 3, 0, 9, 1, 2,
		
		// inpModY: 12
		5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
		
		// outGateX: 9
		0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0,
		
		// outGateY: 12
		1, 2, 3, 2, 1, 2, 3, 2, 1, 2, 3, 2,
		
		// outGateH: 12
		0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, -1.1, -0.4, -0.125
	});
	
	// Output, fixed given input and weights
	Tensor<> out = Tensor<>({
		0.69628, 0.73962, -0.69862,
		0.75505, 0.76076, -0.75973,
		0.96590, 0.96347, -0.00642
	}).resize(3, 1, 3);
	
	// Reversed output, fixed given input and weights
	Tensor<> rOut = Tensor<>({
		0.99347, 0.98614, 0.70684,
		0.95272, 0.90196, 0.95457,
		0.73827, 0.74884, 0.75822
	}).resize(3, 1, 3);
	
	// Input gradient, fixed given input and output gradient
	Tensor<> ing = Tensor<>({
		-0.03112, -0.00071,
		0.04170, 0.02960,
		-0.02195, -0.04333
	}).resize(3, 1, 2);
	
	// Reversed input gradient, fixed given input and output gradient
	Tensor<> rIng = Tensor<>({
		0.00002, 0.00001,
		0.02605, 0.03943,
		-0.00803, -0.01839
	}).resize(3, 1, 2);
	
	// Parameter gradient, fixed given the input and output gradient
	Tensor<> prg = Tensor<>({
		0.47546, -0.17090, -0.40531, -0.59076, 0.22103, -0.01223, 0.05094, -0.01696, -0.07484, 0.02503,
		-0.00834, -0.06154, 0.02662, -0.00891, -0.06239, -0.02511, 0.00836, 0.06190, 0.05094, -0.01696,
		-0.07484, 0.03558, -0.01189, -0.08465, 0.03574, -0.01198, -0.08228, -0.03595, 0.01205, 0.08198,
		0.05094, -0.01696, -0.07484, 0, -0.00025, 0, 0, 0.00034, 0, 0, -0.00002, 0, 0, -0.00001, 0, 0,
		-0.00001, 0, 0, 0.00001, 0, 0, -0.00002, 0, 0, -0.00002, 0, 0, -0.00002, 0, 0, 0.00002, 0, 0,
		-0.00002, 0, 0, 0.00008, 0, 0, -0.00016, 0, 0, -0.00002, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -0.00002, 0,
		0.58074, -0.27490, 0.91818, -0.57242, 0.50183, -1.84492, 0.12095, 0.03566, -0.18707, 0.04098,
		-0.00334, -0.00060, 0.04355, -0.00355, -0.00063, -0.04112, 0.00335, 0.00060, 0.12095, 0.03566,
		-0.18707, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
	});
	
	// Test forward and backward using the parameters and targets above
	
	Sequencer<> module(lstm);
	NNAssertEquals(lstm, &module.module(), "Sequencer::Sequencer failed!");
	
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssertLessThan(module.output().add(out, -1).square().sum(), 1e-9, "Sequencer::forward failed!");
	NNAssertLessThan(module.inGrad().add(ing, -1).square().sum(), 1e-9, "Sequencer::backward failed; wrong inGrad!");
	NNAssertLessThan(module.grad().addV(prg, -1).square().sum(), 1e-9, "Sequencer::backward failed; wrong grad!");
	
	module.forget();
	module.reverse(true);
	module.forward(inp);
	module.backward(inp, grd);
	
	NNAssertLessThan(module.output().add(rOut, -1).square().sum(), 1e-9, "Sequencer::forward (reversed) failed!");
	NNAssertLessThan(module.inGrad().add(rIng, -1).square().sum(), 1e-9, "Sequencer::backward (reversed) failed; wrong inGrad!");
	
	{
		BatchNorm<> *b = new BatchNorm<>(10);
		Sequencer<> s(b);
		s.training(false);
		NNAssert(!b->isTraining(), "Sequencer::training failed!");
	}
	
	TestModule("Sequencer", module, inp);
}

#endif