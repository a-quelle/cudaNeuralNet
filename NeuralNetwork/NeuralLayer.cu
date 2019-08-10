#include "NeuralLayer.h"
#include <random>

cudaError_t cudaStatus;

using namespace std;
random_device rd;  //Will be used to obtain a seed for the random number engine
mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
uniform_real_distribution<> dis(0.0, 1.0);

NeuralLayer::NeuralLayer() :
	numberOfInputs(0), numberOfNeurons(0)
{}

NeuralLayer::NeuralLayer(double* inputPtr, const int nInputs, const int nNeurons) : 
	numberOfInputs(nInputs), numberOfNeurons(nNeurons)
{
	d_inputs = inputPtr;

	CUDA_CALL(cudaMalloc(&d_outputs, (nNeurons+1) * sizeof(double)));
	double* outputs = new double[nNeurons+1]();
	outputs[0] = 1;
	CUDA_CALL(cudaMemcpy(d_outputs, outputs, (nNeurons+1) * sizeof(double), cudaMemcpyHostToDevice));
	delete[] outputs;

	CUDA_CALL(cudaMalloc(&d_weightMatrix, (nNeurons*(nInputs + 1)) * sizeof(double)));
	CUDA_CALL(cudaMemset(d_weightMatrix, 0, nNeurons*(nInputs + 1) * sizeof(double)));
	CUDA_CALL(cudaMalloc(&d_dydx, (nNeurons*nInputs) * sizeof(double)));
	CUDA_CALL(cudaMemset(d_dydx, 0, (nNeurons*nInputs) * sizeof(double)));
	CUDA_CALL(cudaMalloc(&d_dydw, (nNeurons*(nInputs + 1)*nNeurons) * sizeof(double)));
	CUDA_CALL(cudaMemset(d_dydw, 0, (nNeurons*(nInputs + 1)*nNeurons) * sizeof(double)));
	generateRandomWeights();
}

NeuralLayer::NeuralLayer(NeuralLayer&& other): numberOfInputs(other.numberOfInputs), numberOfNeurons(other.numberOfNeurons), d_weightMatrix(other.d_weightMatrix),
	d_dydx(other.d_dydx), d_dydw(other.d_dydw), d_inputs(other.d_inputs), d_outputs(other.d_outputs)
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

__global__ void HidCalcDyDx(double* d_dydx, double* d_outputs, double* d_weightMatrix, int numberOfNeurons, int numberOfInputs)
{
    int input = threadIdx.x;
	int output =  threadIdx.y;
	if (output < numberOfNeurons && input < numberOfInputs)
		d_dydx[output * numberOfInputs + input] = d_outputs[output + 1] * d_weightMatrix[output * (numberOfInputs + 1) + input + 1] * (1 - d_outputs[output + 1]);
}

void HiddenLayer::calcDyDx()
{
	HidCalcDyDx<<<{1,1},{32,32}>>>(d_dydx, d_outputs, d_weightMatrix, numberOfNeurons, numberOfInputs);
	CUDA_GET_ERROR();
}

__global__ void HidCalcDyDw(double* d_dydw, double* d_outputs, double* d_inputs, int numberOfNeurons, int numberOfInputs)
{
	int weightInput = blockIdx.x * blockDim.x + threadIdx.x;
	int weightOutput = blockIdx.y * blockDim.y + threadIdx.y;
	int output = blockIdx.z * blockDim.z + threadIdx.z;
	if(output < numberOfNeurons && weightOutput < numberOfNeurons && weightInput < (numberOfInputs+1))
	{
		if(weightOutput == output)
		{
			d_dydw[output*numberOfNeurons*(numberOfInputs+1)+weightOutput*(numberOfInputs+1)+weightInput] = 
				d_outputs[output+1] * d_inputs[weightInput] * (1 - d_outputs[weightOutput+1]);
		}
	}
}

void HiddenLayer::calcDyDw()
{
	HidCalcDyDw<<<{3,3,3},{8, 8, 8}>>>(d_dydw, d_outputs, d_weightMatrix, numberOfNeurons, numberOfInputs);
	CUDA_GET_ERROR();
}

HiddenLayer::HiddenLayer(){};
HiddenLayer::HiddenLayer(double* inputPtr, const int inputs, const int neurons) : NeuralLayer(inputPtr, inputs, neurons){};

__global__ void HidProcessInput(double* d_outputs, double* d_weightMatrix, double* d_inputs, int numberOfNeurons, int numberOfInputs)
{
	int i = threadIdx.x;
	if(i < numberOfNeurons)
	{
		
		d_outputs[i + 1] = 0;
		for(int j = 0; j < numberOfInputs+1; j++)
		{
			d_outputs[i + 1] += d_weightMatrix[i*(numberOfInputs+1) + j] * d_inputs[j];
		}
		d_outputs[i+1] = 1 / (1+ exp(-d_outputs[i+1]));
	}
}

void HiddenLayer::processInput()
{
	HidProcessInput<<<1,32>>>(d_outputs, d_weightMatrix, d_inputs, numberOfNeurons, numberOfInputs);
	CUDA_GET_ERROR();

	calcDyDx();
	calcDyDw();
}

__global__ void OutCalcDlnyDx(double* d_dydx, double* d_outputs, double* d_weightMatrix, int numberOfNeurons, int numberOfInputs)
{
    int input = threadIdx.x;
	int output =  threadIdx.y;
	if(output < numberOfNeurons && input < numberOfInputs)
	{
		d_dydx[output*numberOfInputs + input] = 0;
		for(int neuron = 0; neuron < numberOfNeurons; ++neuron)
		{
			if (neuron == output)
			{
				d_dydx[output*numberOfInputs + input] += d_weightMatrix[neuron*(numberOfInputs+1) + input + 1] * (1 - d_outputs[neuron + 1]);
			}
			else
			{
				d_dydx[output*numberOfInputs + input] -= d_weightMatrix[neuron*(numberOfInputs+1) + input + 1] * (d_outputs[neuron + 1]);
			}			
		}
	}
}

void OutputLayer::calcDlnyDx()
{	
	OutCalcDlnyDx<<<{1,1},{32,32}>>>(d_dydx, d_outputs, d_weightMatrix, numberOfNeurons, numberOfInputs);
	CUDA_GET_ERROR();
}

__global__ void OutCalcDlnyDw(double* d_dydw, double* d_outputs, double* d_inputs, int numberOfNeurons, int numberOfInputs)
{
    int weightInput = blockIdx.x * blockDim.x + threadIdx.x;
	int input =  blockIdx.y * blockDim.y + threadIdx.y;
	int output = blockIdx.z * blockDim.z + threadIdx.z;
	if(output < numberOfNeurons && input < numberOfInputs && weightInput < numberOfInputs+1)
	{
		d_dydw[output*numberOfInputs*(numberOfInputs+1)+input*(numberOfInputs+1)+ weightInput] = 0;
		if (input == output)
		{
			d_dydw[output*numberOfInputs*(numberOfInputs+1)+input*(numberOfInputs+1)+ weightInput] += d_inputs[weightInput] * (1 - d_outputs[input + 1]);
		}
		else
		{
			d_dydw[output*numberOfInputs*(numberOfInputs+1)+input*(numberOfInputs+1)+ weightInput] -= d_inputs[weightInput] * (d_outputs[input + 1]);
		}
	}
}

void OutputLayer::calcDlnyDw()
{
	OutCalcDlnyDw<<<{4, 4, 4},{8, 8, 8}>>>(d_dydw, d_outputs, d_inputs, numberOfNeurons, numberOfInputs);
	CUDA_CALL(cudaGetLastError());
}

__global__ void OutProcessInput(double* d_outputs, double* d_weightMatrix, double* d_inputs, int numberOfNeurons, int numberOfInputs)
{
	int i = threadIdx.x;
	double invSum = 0;
	if(i < numberOfNeurons)
	{
		
		d_outputs[i + 1] = 0;
		for(int j = 0; j < numberOfInputs+1; j++)
		{
			d_outputs[i + 1] += d_weightMatrix[i*(numberOfInputs+1) + j] * d_inputs[j];
		}
		d_outputs[i + 1] = exp(d_outputs[i + 1]);
	}
	__syncthreads();
	double sum = 0;			
	for(int j = 0; j < numberOfNeurons; j++)
		sum += d_outputs[j + 1];
		invSum = 1 / sum;	

	if(i < numberOfNeurons)
	{
		d_outputs[i + 1] = invSum * d_outputs[i + 1];	
	}
}

void OutputLayer::processInput()
{
	OutProcessInput<<<1,32>>>(d_outputs, d_weightMatrix, d_inputs, numberOfNeurons, numberOfInputs);

	calcDlnyDw();
	calcDlnyDx();
}

OutputLayer::OutputLayer(){};
OutputLayer::OutputLayer(double* inputPtr, const int numberOfInputs, const int numberOfNeurons) : NeuralLayer(inputPtr, numberOfInputs, numberOfNeurons)
{
}