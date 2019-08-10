#include "Regression\Regression.h"
//#include "Plotter\Plotter.h"
#include <thread>


using namespace std;

bool quit; 
extern bool running;

int main(int argc, char* argv[])
{quit = false;
    NeuralNet neuralNet(3, 3, 2, 20);
    neuralNet.readWeights("weights.dat");    
    generateData();
    gradientDescentLoop(neuralNet, 100);
    cout << "Exiting program" << endl;
    return 0;
}