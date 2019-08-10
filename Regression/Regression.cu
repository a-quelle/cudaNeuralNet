#include "Regression.h"
#include <ctime>

int getNumberOfWeights (NeuralNet& neuralNet);
void updateNetworkWeights (NeuralNet& neuralNet);
void calcScaledTotalGrad (NeuralNet& neuralNet);
void calcGradFromDatum (double* d_gradient, const Datum& input, NeuralNet& neuralNet);
__global__ void gradFromOutputLayer (const double* d_dydw, double* singleGradient, int size);
__global__ void gradFromSecondHidden (double* d_dydx, double* d_dydw, double* singleGradient, int neuronsPerLayer);
__global__ void gradFromFirstHidden (double* d_dydxOut, double* d_dydxHid, double* d_dydw, double* singleGradient, int iSize, int jSize);

using namespace std;

Datum::Datum (double lDist, double fDist, double rDist, Direction direction) :
  fDist (fDist), lDist (lDist), rDist (rDist), direction (direction)
{}

bool running = true;
static double scale = 1 / 10000;
vector<Datum> dataVector;
int weightsSize = 0;
static double* d_weights = nullptr;
static double* d_gradient = nullptr;
static double* d_singleGradient = nullptr;
static double* d_input = nullptr;

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
  CUDA_CALL (cudaMalloc (&d_singleGradient, weightsSize * sizeof (double)));
  CUDA_CALL (cudaMalloc (&d_input, 4 * sizeof (double)));
  getNetworkWeights (neuralNet);
  while (running)
  {
    cout << "Continuing, not converged yet..." << endl;
    for (int batch = 1; batch <= batches; batch++)
    {
      std::clock_t start;
      double duration;

      start = std::clock ();

      calcScaledTotalGrad (neuralNet);
      incrementWeights << <(weightsSize + 31) / 32, 32 >> > (d_weights, d_gradient);
      CUDA_GET_ERROR ();
      updateNetworkWeights (neuralNet);

      duration = (std::clock () - start) / (double)CLOCKS_PER_SEC;
      cout << "This batch took " << duration << endl;
    }
    cout << "Saving Weights..." << endl;
    neuralNet.saveWeights ("weights.dat");
  }
  cout << "Regression has stopped." << endl;
  cudaFree (d_weights);
  cudaFree (d_gradient);
  cudaFree (d_singleGradient);
  cudaFree (d_input);
}


void setDataVector (std::vector<Datum> data)
{
  dataVector = data;
}

void generateData ()
{
  dataVector.reserve (50 * 50 * 25);

  for (int lDist = 1; lDist <= 150; lDist += 4)
  {
    for (int rDist = 1; rDist <= 150; rDist += 4)
    {
      for (int fDist = 1; fDist <= 150; fDist += 4)
      {
        if (fDist > 100)
          dataVector.push_back (Datum (lDist, fDist, rDist, Direction::forward));
        if (fDist < 50 && rDist > lDist)
          dataVector.push_back (Datum (lDist, fDist, rDist, Direction::right));
        if (fDist < 50 && lDist > rDist)
          dataVector.push_back (Datum (lDist, fDist, rDist, Direction::left));
      }
    }
  }
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

__global__ void normaliseGradient (double* d_gradient, int weightsSize, int scale)
{
  int i = threadIdx.x;
  double normalisation = 0;
  for (int j = 0; j < weightsSize; ++j)
  {
    int x = d_gradient[j];
    normalisation += x * x;
  }
  normalisation = 1 / sqrt (normalisation);
  if (i < weightsSize)
  {
    d_gradient[i] *= scale;
  }
}

void calcScaledTotalGrad (NeuralNet& neuralNet)
{
  CUDA_CALL (cudaMemset (d_gradient, 0, weightsSize * sizeof (double)));
  for_each (begin (dataVector), end (dataVector),
    [&](Datum& datum) {
    calcGradFromDatum (d_gradient, datum, neuralNet);
  });
  normaliseGradient << <(weightsSize + 31) / 32, 32 >> > (d_gradient, weightsSize, scale);
  CUDA_GET_ERROR();
}

//Only works for exactly 2 hidden layers.
void calcGradFromDatum (double* d_gradient, const Datum& input, NeuralNet& neuralNet)
{
  vector<double> layerGrad;
  double toProcess[4] = { 1,  0.01 * input.lDist, 0.01 * input.fDist, 0.01 * input.rDist }; //leading 1 for affine inputs
  CUDA_CALL (cudaMemcpy(d_input, toProcess, 4 * sizeof(double), cudaMemcpyHostToDevice));
  neuralNet.processInput (d_input);

  double* singGradCpy = d_gradient;

  gradFromFirstHidden << <1, 128 >> > (&neuralNet.outputLayer.d_dydx[input.direction * neuralNet.neuronsPerLayer], neuralNet.layers[1].d_dydx, neuralNet.layers[0].d_dydw,
    singGradCpy, neuralNet.neuronsPerLayer, neuralNet.numberOfInputs + 1);
  CUDA_GET_ERROR();
  singGradCpy += neuralNet.neuronsPerLayer * (neuralNet.numberOfInputs + 1);
  gradFromSecondHidden << <1, 512 >> > (&neuralNet.outputLayer.d_dydx[input.direction * neuralNet.neuronsPerLayer], neuralNet.layers[1].d_dydw, singGradCpy, 
    neuralNet.neuronsPerLayer);
  CUDA_GET_ERROR();
  singGradCpy += neuralNet.neuronsPerLayer * (neuralNet.neuronsPerLayer + 1);
  gradFromOutputLayer << <1, 64 >> > (&neuralNet.outputLayer.d_dydw[input.direction * neuralNet.numberOfOutputs * (neuralNet.neuronsPerLayer + 1)],
    singGradCpy, neuralNet.numberOfOutputs * (neuralNet.neuronsPerLayer + 1));
  CUDA_GET_ERROR();
}

__global__ void gradFromOutputLayer (const double* d_dydw, double* singleGradient, int size)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < size)
  {
    singleGradient[i] = d_dydw[i];
  }
}

__global__ void gradFromSecondHidden (double* d_dydx, double* d_dydw, double* singleGradient, int neuronsPerLayer)
{
  const int iSize = neuronsPerLayer;
  const int jSize = neuronsPerLayer + 1;
  int j = threadIdx.x;
  if (j < jSize * iSize)
  {
    for (int k = 0; k < iSize; ++k)
      singleGradient[j] += d_dydx[k]
      * d_dydw[k * iSize * jSize + j];
  }



}

__global__ void gradFromFirstHidden (double* d_dydxOut, double* d_dydxHid, double* d_dydw, double* singleGradient, int iSize, int jSize)
{
  int j = threadIdx.x;
  if (j < iSize * jSize)
  {
    for (int k = 0; k < iSize; k++)
    {
      for (int l = 0; l < iSize; l++)
      {
        singleGradient[j] +=
          d_dydxOut[k] * d_dydxHid[k * iSize + l] * d_dydw[l * iSize * jSize + j];
      }
    }
  }
}
