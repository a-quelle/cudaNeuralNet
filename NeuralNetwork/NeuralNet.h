#pragma once
#include "NeuralLayer.h"
#include <algorithm>
#include <string>
//#include <mutex>

struct NeuralNet
{
	NeuralNet(const unsigned int inputs, const unsigned int outputs, const unsigned int layers, const unsigned int neurons, const unsigned int batchSize);

	void createLayers();
	
	double* processInput(double* input);
	void saveWeights(const std::string& fileName);
	void readWeights(const std::string& fileName);

	OutputLayer outputLayer;
	std::vector<HiddenLayer> layers;
	
	const unsigned int numberOfLayers;
	const unsigned int numberOfInputs;
	const unsigned int numberOfOutputs;
	const unsigned int neuronsPerLayer;
	const unsigned int batchSize;
	//std::mutex lock;
};

