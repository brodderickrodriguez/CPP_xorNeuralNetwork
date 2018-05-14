//
//  Neuron.h
//  NeuralNetwork_1
//
//  Created by bcr@brodderick.com on 5/13/18.
//  Copyright © 2018 BCR. All rights reserved.
//

#ifndef Neuron_h
#define Neuron_h
#include <vector>
#include <cmath>

struct Connection
{
    double weight, deltaWeight;
}; // Connection

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    double getOutputVal(void) const { return m_outputVal; }
    void setOutputVal(double val) { m_outputVal = val; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double transferFunction(double x) { return tanh(x); }
    static double transferFunctionDerivative(double x) { return 1 - x * x; }
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    const double eta = 0.01;    // modulate output val
    const double alpha = 0.01;  // modulate delta weight
    double m_outputVal;
    double m_gradient;
    unsigned m_myIndex;
    std::vector<Connection> m_outputWeights;
}; // Neuron



#endif /* Neuron_h */
