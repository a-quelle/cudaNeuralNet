#pragma once
#include "NeuralLayer.h"
#include <algorithm>
#include <string>
//#include <mutex>

struct NeuralNet
{
	NeuralNet(const int inputs, const int outputs, const int layers, const int neurons);

	void createLayers();
	
	double* processInput(double* input);
	void saveWeights(const std::string& fileName);
	void readWeights(const std::string& fileName);

	OutputLayer outputLayer;
	std::vector<HiddenLayer> layers;
	
	const int numberOfLayers;
	const int numberOfInputs;
	const int numberOfOutputs;
	const int neuronsPerLayer;
	//std::mutex lock;
};

