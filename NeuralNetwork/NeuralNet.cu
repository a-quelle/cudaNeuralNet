#include "NeuralNet.h"
#include <fstream>

using namespace std;

NeuralNet::NeuralNet(const unsigned int inputs, const unsigned int outputs, const unsigned int nLayers, const unsigned int neurons, const unsigned int batchSize) :
	numberOfLayers(nLayers), numberOfInputs(inputs), numberOfOutputs(outputs), neuronsPerLayer(neurons), outputLayer(nullptr, neurons, outputs, batchSize), batchSize(batchSize)
{	
	layers.reserve(numberOfLayers);
	createLayers();
	outputLayer.d_inputs = layers.back().d_outputs;
}

void NeuralNet::createLayers()
{
	layers.emplace_back(nullptr, numberOfInputs, neuronsPerLayer, batchSize);
	for (int i = 1; i < numberOfLayers; i++) {		
		layers.emplace_back(layers.back().d_outputs, neuronsPerLayer, neuronsPerLayer, batchSize);
	}
}

double* NeuralNet::processInput(double* input)
{
	layers[0].d_inputs = input;
	for (int i = 0; i < numberOfLayers; i++)
	{
		layers[i].processInput();
	}
	outputLayer.processInput();
	return outputLayer.d_outputs;
}

void NeuralNet::saveWeights(const std::string& fileName)
{
	vector<double> weights;
	for_each(begin(layers), end(layers),
	[&](HiddenLayer& layer){
		double* weightMatrix = new double[layer.numberOfNeurons*(layer.numberOfInputs + 1)];
		CUDA_CALL(cudaMemcpy(weightMatrix, layer.d_weightMatrix, layer.numberOfNeurons*(layer.numberOfInputs + 1)*sizeof(double), cudaMemcpyDeviceToHost));
		for (int neuron = 0; neuron < layer.numberOfNeurons; neuron++)
		{
			for (int weight = 0; weight < layer.numberOfInputs + 1; weight++)
			{
				weights.push_back(weightMatrix[neuron*(layer.numberOfInputs + 1) + weight]);
			}
		}
		delete[] weightMatrix;
	});	
	double* weightMatrix = new double[outputLayer.numberOfNeurons*(outputLayer.numberOfInputs + 1)];
	CUDA_CALL(cudaMemcpy(weightMatrix, outputLayer.d_weightMatrix, outputLayer.numberOfNeurons*(outputLayer.numberOfInputs + 1)*sizeof(double), cudaMemcpyDeviceToHost));
	for (int neuron = 0; neuron < outputLayer.numberOfNeurons; neuron++)
	{
		for (int weight = 0; weight < outputLayer.numberOfInputs + 1; weight++)
		{
			weights.push_back(weightMatrix[neuron*(outputLayer.numberOfInputs + 1) + weight]);
		}
	}
	delete[] weightMatrix;
	std::ofstream f(fileName);
	for_each( weights.begin(), weights.end(),
		[&](double x){ f << x << "\n";});
}

void NeuralNet::readWeights(const std::string& fileName)
{
	string str;
	vector<double> input;
	ifstream f(fileName);
	if(f.is_open()){
		while(getline(f, str))
			input.push_back(stod(str));
		for_each(layers.begin(), layers.end(),
		[&](HiddenLayer& layer){
			double* weightMatrix = new double[layer.numberOfNeurons*(layer.numberOfInputs + 1)];
			for (int neuron = 0; neuron < layer.numberOfNeurons; neuron++)
			{
				for (int weight = 0; weight < layer.numberOfInputs + 1; weight++)
				{
					weightMatrix[neuron*(layer.numberOfInputs + 1) + weight] = input[0];
					input.erase(input.begin());
				}
			}
			CUDA_CALL(cudaMemcpy(layer.d_weightMatrix, weightMatrix,layer.numberOfNeurons*(layer.numberOfInputs + 1)*sizeof(double), cudaMemcpyHostToDevice));
			delete[] weightMatrix;
		});
		double* weightMatrix = new double[outputLayer.numberOfNeurons*(outputLayer.numberOfInputs + 1)];
		for (int neuron = 0; neuron < outputLayer.numberOfNeurons; neuron++)
		{
			for (int weight = 0; weight < outputLayer.numberOfInputs + 1; weight++)
			{
				weightMatrix[neuron*(outputLayer.numberOfInputs + 1) + weight] = input[0];
				input.erase(input.begin());
			}
		}
		CUDA_CALL(cudaMemcpy(outputLayer.d_weightMatrix, weightMatrix,outputLayer.numberOfNeurons*(outputLayer.numberOfInputs + 1)*sizeof(double), cudaMemcpyHostToDevice));
		delete[] weightMatrix;
	} else{
		cout << "Could not read weights from file." << endl;
	}
}


