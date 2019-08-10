#pragma once
#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string>

extern cudaError_t cudaStatus;

#define CUDA_CALL(STATEMENT)\
	cudaStatus = STATEMENT;\
    if (cudaStatus != cudaSuccess) {\
        fprintf(stderr, __FILE__);\
        fprintf(stderr, "Line %d failed!\n", __LINE__);\
        fprintf(stderr, cudaGetErrorName(cudaStatus));\
        throw std::string(cudaGetErrorName(cudaStatus));\
    }

#define CUDA_GET_ERROR()\
	cudaStatus = cudaGetLastError();\
    if (cudaStatus != cudaSuccess) {\
        fprintf(stderr, __FILE__);\
        fprintf(stderr, "Line %d failed!\n", __LINE__);\
        fprintf(stderr, cudaGetErrorName(cudaStatus));\
        throw std::string(cudaGetErrorName(cudaStatus));\
    }

struct NeuralLayer
{
	friend class NeuralNet;

	NeuralLayer();
	NeuralLayer(double* inputPtr, const int inputs, const int neurons);
	NeuralLayer(NeuralLayer&  other) = delete;
	NeuralLayer(NeuralLayer&&  other);
	~NeuralLayer();
	double* d_inputs = nullptr;
	double* d_outputs = nullptr;	
	
	const int numberOfInputs;
	const int numberOfNeurons;
	double* d_weightMatrix;
	double* d_dydx;
	double* d_dydw;

	void generateRandomWeights();
};

struct HiddenLayer : NeuralLayer
{
	HiddenLayer();
	HiddenLayer(double* inputPtr, const int inputs, const int neurons);
	void calcDyDx();
	void calcDyDw();
	void processInput();
};

struct OutputLayer : NeuralLayer
{
	OutputLayer();
	OutputLayer(double* inputPtr, const int inputs, const int neurons);
	void calcDlnyDx();
	void calcDlnyDw();
	void processInput();
};

