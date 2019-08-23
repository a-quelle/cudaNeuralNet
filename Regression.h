#include "..\NeuralNetwork\NeuralNet.h"

enum Direction { left, forward, right };

struct Datum
{
    double one = 1, lDist, fDist, rDist;
    Datum(double lDist, double fDist, double rDist);
    Datum() = default;
};

void gradientDescentLoop(NeuralNet& neuralNet, int batches);
void getNetworkWeights(NeuralNet& neuranNet);
void generateData();
void cleanData();