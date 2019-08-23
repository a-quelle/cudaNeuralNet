#include "Regression.h"
#include <ctime>

int getNumberOfWeights (NeuralNet& neuralNet);
void updateNetworkWeights (NeuralNet& neuralNet);
void calcGrad (NeuralNet& neuralNet);
__global__ void gradFromOutputLayer (double* d_dydw, double* singleGradient, unsigned int* direction, const unsigned int size, const unsigned int numOutputs, const unsigned int batchSize, const unsigned int weightsSize);
__global__ void gradFromSecondHidden (double* d_dydx, double* d_dydw, double* singleGradient, unsigned int* direction, const unsigned int neuronsPerLayer, const unsigned int numOutputs, const unsigned int batchSize, const unsigned int weightsSize);
__global__ void gradFromFirstHidden (double* d_dydxOut, double* d_dydxHid, double* d_dydw, double* singleGradient, unsigned int* direction, const unsigned int iSize, const unsigned int jSize, const unsigned int numOutputs, const unsigned int batchSize, const unsigned int weightsSize);

using namespace std;

Datum::Datum (double lDist, double fDist, double rDist) :
  fDist (fDist), lDist (lDist), rDist (rDist)
{}

bool running = true;
static double scale = 1 / 10000;
Datum* dataVector = new Datum[dataSize];
Direction* resultsVector = new Direction[dataSize];
unsigned int weightsSize = 0;
static double* d_weights = nullptr;
static double* d_gradient = nullptr;
static double* d_input = nullptr;
static unsigned int* d_direction = nullptr;

__global__ void incrementWeights (double* d_weights, double* d_gradient)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  d_weights[i] += d_gradient[i];
}

void gradientDescentLoop (NeuralNet& neuralNet, int batches)
{
  weightsSize = getNumberOfWeights (neuralNet);
  CUDA_CALL (cudaMalloc (&d_weights, weightsSize * sizeof (double)));
  CUDA_CALL (cudaMalloc (&d_gradient, weightsSize * sizeof (double)));
  CUDA_CALL (cudaMalloc (&d_input, dataSize * 4 * sizeof (double)));
  CUDA_CALL (cudaMalloc (&d_direction, dataSize * sizeof (unsigned int)));

  CUDA_CALL (cudaMemcpy (d_input, dataVector, dataSize * 4 * sizeof (double), cudaMemcpyHostToDevice));
  CUDA_CALL (cudaMemcpy (d_direction, resultsVector, dataSize * sizeof (unsigned int), cudaMemcpyHostToDevice));
  getNetworkWeights (neuralNet);
  while (running)
  {
    cout << "Continuing, not converged yet..." << endl;
    for (int batch = 1; batch <= batches; batch++)
    {
      std::clock_t start;
      double duration;

      start = std::clock ();

      calcGrad (neuralNet);
      incrementWeights << <(weightsSize + 31) / 32, 32 >> > (d_weights, d_gradient);
      CUDA_GET_ERROR ();
      updateNetworkWeights (neuralNet);
      cudaThreadSynchronize();

      duration = (std::clock () - start) / (double)CLOCKS_PER_SEC;
      cout << "This batch took " << duration << endl;
    }
    cout << "Saving Weights..." << endl;
    neuralNet.saveWeights ("weights.dat");
  }
  cout << "Regression has stopped." << endl;
  cudaFree (d_weights);
  cudaFree (d_gradient);
  cudaFree (d_input);
  cudaFree (d_direction);
}

void generateData ()
{
  int counter = 0;
  for (int lDist = 1; lDist <= 150; lDist += 4)
  {
    for (int rDist = 1; rDist <= 150; rDist += 4)
    {
      for (int fDist = 1; fDist <= 150; fDist += 4)
      {
        if (fDist > 100)
        {
          dataVector[counter] = Datum (lDist, fDist, rDist);
          resultsVector[counter] = Direction::forward;
          ++counter;
        }
        if (fDist < 50 && rDist > lDist)
        {
          dataVector[counter] = Datum (lDist, fDist, rDist);
          resultsVector[counter] = Direction::right;
          ++counter;
        }
        if (fDist < 50 && lDist > rDist)
        {
          dataVector[counter] = Datum (lDist, fDist, rDist);
          resultsVector[counter] = Direction::right;
          ++counter;
        }
      }
    }
  }
}

void cleanData ()
{
  delete[] dataVector;
  delete[] resultsVector;
}

int getNumberOfWeights (NeuralNet& neuralNet)
{
  int numberOfWeights = 0;
  for_each (begin (neuralNet.layers), end (neuralNet.layers),
    [&](HiddenLayer& layer) {
    numberOfWeights += layer.numberOfNeurons * (layer.numberOfInputs + 1);
  });
  OutputLayer& outputLayer = neuralNet.outputLayer;
  numberOfWeights += outputLayer.numberOfNeurons * (outputLayer.numberOfInputs + 1);
  return numberOfWeights;
}

void getNetworkWeights (NeuralNet& neuralNet)
{
  CUDA_CALL (cudaMemset (d_weights, 0, weightsSize * sizeof (double)));
  double* d_weightsCopy = d_weights;
  for_each (begin (neuralNet.layers), end (neuralNet.layers),
    [&](HiddenLayer& layer) {
    cudaMemcpy (d_weightsCopy, layer.d_weightMatrix, layer.numberOfNeurons * (layer.numberOfInputs + 1) * sizeof (double), cudaMemcpyDeviceToDevice);
    d_weightsCopy += layer.numberOfNeurons * (layer.numberOfInputs + 1);
  });
  cudaMemcpy (d_weightsCopy, neuralNet.outputLayer.d_weightMatrix, neuralNet.outputLayer.numberOfNeurons * (neuralNet.outputLayer.numberOfInputs + 1) * sizeof (double), cudaMemcpyDeviceToDevice);
}

void updateNetworkWeights (NeuralNet& neuralNet)
{
  double* d_weightsPtr = d_weights;
  for_each (begin (neuralNet.layers), end (neuralNet.layers),
    [&d_weightsPtr](HiddenLayer& layer) {
    CUDA_CALL (cudaMemcpy (layer.d_weightMatrix, d_weightsPtr, layer.numberOfNeurons * (layer.numberOfInputs + 1) * sizeof (double), cudaMemcpyDeviceToDevice));
    d_weightsPtr += layer.numberOfNeurons * (layer.numberOfInputs + 1);
  });
  CUDA_CALL (cudaMemcpy (neuralNet.outputLayer.d_weightMatrix, d_weightsPtr, neuralNet.outputLayer.numberOfNeurons * (neuralNet.outputLayer.numberOfInputs + 1) * sizeof (double), cudaMemcpyDeviceToDevice));
}

//Only works for exactly 2 hidden layers.
void calcGrad (NeuralNet& neuralNet)
{
  neuralNet.processInput ((double*)d_input);

  const unsigned int iSize = neuralNet.neuronsPerLayer;
  const unsigned int jSize = neuralNet.numberOfInputs + 1;
  const unsigned int numOutputs = neuralNet.numberOfOutputs;
  const unsigned int yDim = 32;

  gradFromFirstHidden <<<{jSize, (dataSize + yDim - 1) / yDim}, { iSize, yDim }, yDim*iSize*sizeof(double)>>> (neuralNet.outputLayer.d_dydx, neuralNet.layers[1].d_dydx, neuralNet.layers[0].d_dydw,
    d_gradient, d_direction, iSize, jSize, numOutputs, dataSize, weightsSize);
  CUDA_GET_ERROR ();

  double* layerTwoGrad = d_gradient + neuralNet.neuronsPerLayer * (neuralNet.numberOfInputs + 1);
  gradFromSecondHidden <<< {jSize, (dataSize + yDim - 1) / yDim}, { iSize, yDim }, iSize*yDim*sizeof(double) >>> (neuralNet.outputLayer.d_dydx, neuralNet.layers[1].d_dydw, 
    layerTwoGrad, d_direction, neuralNet.neuronsPerLayer, numOutputs, dataSize, weightsSize);
  CUDA_GET_ERROR ();

  double* outGrad = layerTwoGrad + neuralNet.neuronsPerLayer * (neuralNet.neuronsPerLayer + 1);
  unsigned int size = numOutputs * (neuralNet.neuronsPerLayer + 1);
  gradFromOutputLayer <<< {(size + 31) / 32, (dataSize + 7) / 8}, { 32, 8 } >>> (neuralNet.outputLayer.d_dydw, outGrad, d_direction, size, numOutputs, dataSize, weightsSize);
  CUDA_GET_ERROR ();
}

__global__ void gradFromOutputLayer (double* d_dydw, double* grad, unsigned int* direction, const unsigned int size, const unsigned int numOutputs, const unsigned int batchSize, const unsigned int weightsSize)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < size && batch < batchSize)
  {
    atomicAdd(grad + i, d_dydw[batch * numOutputs * size + direction[batch] * size + i]);
  }
}

__global__ void gradFromSecondHidden (double* d_dydx, double* d_dydw, double* grad, unsigned int* direction, const unsigned int neuronsPerLayer,
  const unsigned int numOfOutputs, const unsigned int batchSize, const unsigned int weightsSize)
{
  extern __shared__ double s[];
  const int iSize = neuronsPerLayer; //blockDim.x
  const int jSize = neuronsPerLayer + 1;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int batch = blockIdx.y * blockDim.y + threadIdx.y;
  double gradVal = 0;
  s[threadIdx.y*iSize + threadIdx.x] = d_dydx[batch*numOfOutputs*iSize + direction[batch]*iSize + threadIdx.x];
  __syncthreads();
  if (j < jSize * iSize && batch < batchSize)
  {
    #pragma unroll
    for (int k = 0; k < iSize; ++k)
    {
      gradVal += s[threadIdx.y*iSize + k] //d_dydx[batch * numOfOutputs * iSize + direction[batch] * iSize + k]
      * d_dydw[batch * iSize * iSize * jSize + k * iSize * jSize + j];
    }
    atomicAdd(grad+j, gradVal);
  }
}

__global__ void gradFromFirstHidden (double* d_dydxOut, double* d_dydxHid, double* d_dydw, double* grad, unsigned int* direction, const unsigned int iSize, const unsigned int jSize,
  const unsigned int numOutputs, const unsigned int batchSize, const unsigned int weightsSize)
{
  const int j = blockIdx.x * blockDim.y + threadIdx.x;
  const int batch = blockIdx.y * blockDim.y + threadIdx.y;
  extern __shared__ double outyx[];

  if(threadIdx.x < iSize && batch < batchSize)
  {
    #pragma unroll    
    for (int k = 0; k < iSize; k++)
    {
      outyx[threadIdx.y*iSize + threadIdx.x] += d_dydxHid[batch*iSize*jSize + k*iSize + threadIdx.x] *  d_dydxOut[batch * numOutputs * iSize + direction[batch] * iSize + k];
    }
  }

  __syncthreads();
  double gradVal = 0;
  if (j < iSize * jSize && batch < batchSize)
  {
    #pragma unroll
    for(int l = 0; l < iSize; ++l)
    {
      gradVal += outyx[threadIdx.y*iSize + l] * d_dydw[batch * iSize * jSize * iSize + l * iSize * jSize + j];
    }
    atomicAdd(grad + j, gradVal / 1000);
  }  

    // for (int k = 0; k < iSize; k++)
    // {
    //   for (int l = 0; l < iSize; l++)
    //   {
    //     singleGradient[batch * weightsSize + j] +=
    //       d_dydxOut[batch * numOutputs * iSize + direction[batch] * iSize + k] * d_dydxHid[batch * iSize * iSize + k * iSize + l] * d_dydw[batch * iSize * jSize * iSize + l * iSize * jSize + j];
    //   }
    // }       
}
