#include "Regression\Regression.h"
//#include "Plotter\Plotter.h"
#include <thread>

//Current best: 0.105 secs per epoch
using namespace std;

bool quit; 
extern bool running;

int main(int argc, char* argv[])
{
    quit = false;
    NeuralNet neuralNet(3, 3, 2, 20, dataSize);
    neuralNet.readWeights("weights.dat");    
    generateData();
    gradientDescentLoop(neuralNet, 10000);
    cleanData();
    cout << "Exiting program" << endl;
    return 0;
}