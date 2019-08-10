#include "..\NeuralNetwork\NeuralNet.h"

enum Direction { left, forward, right };

struct Datum
{
    double fDist, lDist, rDist;
    Direction direction;    
    Datum(double lDist, double fDist, double rDist, Direction direction);
};

void gradientDescentLoop(NeuralNet& neuralNet, int batches);
void getNetworkWeights(NeuralNet& neuranNet);
void setDataVector(std::vector<Datum> data);
void generateData();