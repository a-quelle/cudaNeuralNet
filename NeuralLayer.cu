#include "NeuralLayer.h"
#include <random>

cudaError_t cudaStatus;

using namespace std;
random_device rd;  //Will be used to obtain a seed for the random number engine
mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
uniform_real_distribution<> dis(0.0, 1.0);

NeuralLayer::NeuralLayer() :
	numberOfInputs(0), numberOfNeurons(0), batchSize(0)
{}

NeuralLayer::NeuralLayer(double* inputPtr, const unsigned int nInputs, const unsigned int nNeurons, const unsigned int batchSize) : 
	numberOfInputs(nInputs), numberOfNeurons(nNeurons), batchSize(batchSize)
{
	d_inputs = inputPtr;

	CUDA_CALL(cudaMalloc(&d_outputs, batchSize*(nNeurons+1) * sizeof(double)));
	double* outputs = new double[batchSize*(nNeurons+1)]();
	for(int i = 0; i < batchSize*(nNeurons+1); i += nNeurons + 1)
	{
		outputs[i] = 1;
	}
	CUDA_CALL(cudaMemcpy(d_outputs, outputs, batchSize*(nNeurons+1) * sizeof(double), cudaMemcpyHostToDevice));
	delete[] outputs;

	CUDA_CALL(cudaMalloc(&d_weightMatrix, (nNeurons*(nInputs + 1)) * sizeof(double)));
	CUDA_CALL(cudaMemset(d_weightMatrix, 0, nNeurons*(nInputs + 1) * sizeof(double)));
	CUDA_CALL(cudaMalloc(&d_dydx, batchSize*(nNeurons*nInputs) * sizeof(double)));
	CUDA_CALL(cudaMemset(d_dydx, 0, batchSize*(nNeurons*nInputs) * sizeof(double)));
	CUDA_CALL(cudaMalloc(&d_dydw, batchSize*(nNeurons*(nInputs + 1)*nNeurons) * sizeof(double)));
	CUDA_CALL(cudaMemset(d_dydw, 0, batchSize*(nNeurons*(nInputs + 1)*nNeurons) * sizeof(double)));
	generateRandomWeights();
}

NeuralLayer::NeuralLayer(NeuralLayer&& other): numberOfInputs(other.numberOfInputs), numberOfNeurons(other.numberOfNeurons), batchSize(other.batchSize),
	d_weightMatrix(other.d_weightMatrix), d_dydx(other.d_dydx), d_dydw(other.d_dydw), d_inputs(other.d_inputs), d_outputs(other.d_outputs)
{
	other.d_weightMatrix = nullptr;
	other.d_dydx = nullptr;
	other.d_dydw = nullptr;
	other.d_outputs = nullptr;		
}

NeuralLayer::~NeuralLayer(void)
{
	cudaFree(d_weightMatrix);
	cudaFree(d_dydx);
	cudaFree(d_dydw);
	cudaFree(d_outputs);
}



void NeuralLayer::generateRandomWeights()
{
	double* weightMatrix = new double[numberOfNeurons*(numberOfInputs + 1)];
	for ( int i = 0; i < numberOfNeurons; i++)
	{
		for (int j = 0; j < numberOfInputs + 1; j++) {
			weightMatrix[i*(numberOfInputs+1) + j] = dis(gen) - dis(gen);
		}
	}
	CUDA_CALL(cudaMemcpy(d_weightMatrix, weightMatrix, numberOfNeurons*(numberOfInputs + 1)*sizeof(double), cudaMemcpyHostToDevice));
	delete[] weightMatrix;
}

__global__ void HidCalcDyDx(double* d_dydx, double* d_outputs, double* d_weightMatrix, const unsigned int numberOfNeurons, const unsigned int numberOfInputs, const unsigned int batchSize)
{
    int input = threadIdx.x;
	int output =  threadIdx.y;
	int batch = blockIdx.z*blockDim.z + threadIdx.z;
	if (output < numberOfNeurons && input < numberOfInputs && batch < batchSize)
	{
		d_dydx[batch*numberOfInputs*numberOfNeurons + output * numberOfInputs + input] = 
			d_outputs[batch*(numberOfNeurons+1) + output + 1] * d_weightMatrix[output * (numberOfInputs + 1) + input + 1] * (1 - d_outputs[batch*(numberOfNeurons+1) + output + 1]);
	}
}

void HiddenLayer::calcDyDx()
{
	HidCalcDyDx<<<{1,1, (batchSize+1)/2},{numberOfInputs,numberOfNeurons, 2}>>>(d_dydx, d_outputs, d_weightMatrix, numberOfNeurons, numberOfInputs, batchSize);
	CUDA_GET_ERROR();
}

__global__ void HidCalcDyDw(double* d_dydw, double* d_outputs, double* d_inputs, const unsigned int numberOfNeurons, const unsigned int numberOfInputs, const unsigned int batchSize)
{
	int weightInput = threadIdx.x;
	int batch = blockIdx.y * blockDim.y + threadIdx.y;
	if(batch < batchSize && weightInput < numberOfInputs+1)
	{
		for (int output = 0; output < numberOfNeurons; output++)
		{
					d_dydw[output*numberOfNeurons*(numberOfInputs+1)+output*(numberOfInputs+1)+weightInput] = 
						d_outputs[output+1] * d_inputs[weightInput] * (1 - d_outputs[output+1]);
		}
	}
}

void HiddenLayer::calcDyDw()
{
	HidCalcDyDw<<<{1, (batchSize+31)/32}, {numberOfInputs+1, 32}>>>(d_dydw, d_outputs, d_inputs, numberOfNeurons, numberOfInputs, batchSize);
	CUDA_GET_ERROR();
}

HiddenLayer::HiddenLayer(){};
HiddenLayer::HiddenLayer(double* inputPtr, const unsigned int inputs, const unsigned int neurons, const unsigned int batchSize) : NeuralLayer(inputPtr, inputs, neurons, batchSize){};

__global__ void HidProcessInput(double* d_outputs, double* d_weightMatrix, double* d_inputs, const unsigned int numberOfNeurons, const unsigned int numberOfInputs, const unsigned int batchSize)
{
	int i = threadIdx.x;
	int batch = blockIdx.y*blockDim.y + threadIdx.y;
	if(i < numberOfNeurons && batch < batchSize)
	{
		
		d_outputs[i + 1] = 0;
		for(int j = 0; j < numberOfInputs+1; j++)
		{
			d_outputs[batch*numberOfNeurons + i + 1] += d_weightMatrix[i*(numberOfInputs+1) + j] * d_inputs[batch*(numberOfInputs+1) + j];
		}
		d_outputs[batch*numberOfNeurons + i+1] = 1 / (1+ exp(-d_outputs[batch*numberOfNeurons + i+1]));
	}
}

void HiddenLayer::processInput()
{
	HidProcessInput<<<{1, (batchSize+31)/32},{numberOfNeurons, 32}>>>(d_outputs, d_weightMatrix, d_inputs, numberOfNeurons, numberOfInputs, batchSize);
	CUDA_GET_ERROR();

	calcDyDx();
	calcDyDw();
}

__global__ void OutCalcDlnyDx(double* d_dydx, double* d_outputs, double* d_weightMatrix, const unsigned int numberOfNeurons, const unsigned int numberOfInputs, const unsigned int batchSize)
{
    int input = threadIdx.x;
	int output =  threadIdx.y;
	int batch = blockIdx.z*blockDim.z + threadIdx.z;
	if(output < numberOfNeurons && input < numberOfInputs && batch < batchSize)
	{
		d_dydx[batch*numberOfInputs*numberOfNeurons + output*numberOfInputs + input] = 0;
		for(int neuron = 0; neuron < numberOfNeurons; ++neuron)
		{
			if (neuron == output)
			{
				d_dydx[batch*numberOfInputs*numberOfNeurons + output*numberOfInputs + input] += 
					d_weightMatrix[neuron*(numberOfInputs+1) + input + 1] * (1 - d_outputs[batch*(numberOfNeurons+1) + neuron + 1]);
			}
			else
			{
				d_dydx[batch*numberOfInputs*numberOfNeurons + output*numberOfInputs + input] -= 
					d_weightMatrix[neuron*(numberOfInputs+1) + input + 1] * (d_outputs[batch*(numberOfNeurons+1) + neuron + 1]);
			}			
		}
	}
}

void OutputLayer::calcDlnyDx()
{	
	OutCalcDlnyDx<<<{1,1, (batchSize+7)/8},{numberOfInputs,numberOfNeurons, 8}>>>(d_dydx, d_outputs, d_weightMatrix, numberOfNeurons, numberOfInputs, batchSize);
	CUDA_GET_ERROR();
}

__global__ void OutCalcDlnyDw(double* d_dydw, double* d_outputs, double* d_inputs, const unsigned int numberOfNeurons, const unsigned int numberOfInputs, const unsigned int batchSize)
{
    int weightInput = threadIdx.x;
    int batch = blockIdx.y * blockDim.y + threadIdx.y;
	if(batch < batchSize && weightInput < numberOfInputs+1)
	{
		for (int output = 0; output < numberOfNeurons; output++)
		{
			for (int weightOutput = 0; weightOutput < numberOfNeurons; weightOutput++)
			{
				d_dydw[output*numberOfInputs*(numberOfInputs+1)+weightOutput*(numberOfInputs+1)+ weightInput] = 0;
				if (weightOutput == output)
				{
					d_dydw[output*numberOfInputs*(numberOfInputs+1)+weightOutput*(numberOfInputs+1)+ weightInput] += d_inputs[weightInput] * (1 - d_outputs[weightOutput + 1]);
				}
				else
				{
					d_dydw[output*numberOfInputs*(numberOfInputs+1)+weightOutput*(numberOfInputs+1)+ weightInput] -= d_inputs[weightInput] * (d_outputs[weightOutput + 1]);
				}
			}
		}
	}
}

void OutputLayer::calcDlnyDw()
{
	OutCalcDlnyDw<<<{1, (batchSize+31)/32}, {numberOfInputs+1, 32}>>>(d_dydw, d_outputs, d_inputs, numberOfNeurons, numberOfInputs, batchSize);
	CUDA_CALL(cudaGetLastError());
}

__global__ void OutProcessInput(double* d_outputs, double* d_weightMatrix, double* d_inputs, const unsigned int numberOfNeurons, const unsigned int numberOfInputs, const unsigned int batchSize)
{
	int i = threadIdx.x;
	int batch = blockIdx.y*blockDim.y + threadIdx.y;
	double invSum = 0;
	if(i < numberOfNeurons && batch < batchSize)
	{
		
		d_outputs[i + 1] = 0;
		for(int j = 0; j < numberOfInputs+1; j++)
		{
			d_outputs[batch*numberOfNeurons + i + 1] += d_weightMatrix[i*(numberOfInputs+1) + j] * d_inputs[batch*(numberOfInputs+1) + j];
		}
		d_outputs[batch*numberOfNeurons + i + 1] = exp(d_outputs[batch*numberOfNeurons + i + 1]);
	}
	__syncthreads();
	
	if(i < numberOfNeurons && batch < batchSize)
	{
		double sum = 0;			
		for(int j = 0; j < numberOfNeurons; j++)
		sum += d_outputs[batch*numberOfNeurons + j + 1];
		invSum = 1 / sum;	

		d_outputs[batch*numberOfNeurons + i + 1] = invSum * d_outputs[batch*numberOfNeurons + i + 1];	
	}
}

void OutputLayer::processInput()
{
	OutProcessInput<<<{1, (batchSize+31)/32},{numberOfNeurons, 32}>>>(d_outputs, d_weightMatrix, d_inputs, numberOfNeurons, numberOfInputs, batchSize);

	calcDlnyDw();
	calcDlnyDx();
}

OutputLayer::OutputLayer(){};
OutputLayer::OutputLayer(double* inputPtr, const unsigned int numberOfInputs, const unsigned int numberOfNeurons, const unsigned int batchSize) : NeuralLayer(inputPtr, numberOfInputs, numberOfNeurons, batchSize)
{
}