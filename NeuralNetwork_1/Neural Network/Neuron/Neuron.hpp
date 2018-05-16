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

struct Connection {
    double weight, deltaWeight;
}; // Connection

class Neuron;
typedef std::vector<Neuron> Layer;

class Neuron {
public:
    Neuron(unsigned numOutputs, unsigned myIndex);
    void setOutputVal(double val) { m_outputVal = val; }
    double getOutputVal(void) const { return m_outputVal; }
    void feedForward(const Layer &prevLayer);
    void calcOutputGradients(double targetVals);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);
private:
    static double eta; // [0.0...1.0] overall net training rate
    static double alpha; // [0.0...n] multiplier of last weight change [momentum]
    static double transferFunction(double x);
    static double transferFunctionDerivative(double x);
    // randomWeight: 0 - 1
    static double randomWeight(void) { return rand() / double(RAND_MAX); }
    double sumDOW(const Layer &nextLayer) const;
    double m_outputVal;
    std::vector<Connection> m_outputWeights;
    unsigned m_myIndex;
    double m_gradient;
}; // Neuron

#endif /* Neuron_h */
