//
//  main.cpp
//  NeuralNetwork_1
//
//  Created by bcr@brodderick.com on 2/26/18.
//  Copyright © 2018 BCR. All rights reserved.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <stdio.h>

using namespace std;

struct Connection
{
    double weight, deltaWeight;
}; // Connection

class Neuron;
typedef vector<Neuron> Layer;

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
    vector<Connection> m_outputWeights;
}; // Neuron

void Neuron::updateInputWeights(Layer &prevLayer)
{
    for (unsigned n = 0; n < prevLayer.size() - 1; ++n)
    {
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;
        double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient
                                + alpha * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight += newDeltaWeight;
    } // for n
} // Neuron::updateInputWeights

double Neuron::sumDOW(const Layer &nextLayer) const
{
    double sum = 0.0;
    for (unsigned n = 0; n < nextLayer.size() - 1; ++n)
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    return sum;
} // Neuron::sumDOW

void Neuron::calcHiddenGradients(const Layer &nextLayer)
{
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
} // Neuron::calcHiddenGradients

void Neuron::calcOutputGradients(double targetVal)
{
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
} // Neuron::calcOutputGradients

void Neuron::feedForward(const Layer &prevLayer)
{
    double sum = 0.0;
    for (unsigned n = 0; n < prevLayer.size(); ++n)
    {
        sum += prevLayer[n].getOutputVal() *
        prevLayer[n].m_outputWeights[m_myIndex].weight;
    } // for n
    m_outputVal = Neuron::transferFunction(sum);
} // Neuron::feedForward

Neuron::Neuron(unsigned numOutputs, unsigned myIndex)
{
    for (unsigned c = 0; c < numOutputs; ++c)
    {
        m_outputWeights.push_back(Connection());
        m_outputWeights.back().weight = randomWeight();
    } // for c
    m_myIndex = myIndex;
} // Neuron::Neuron


class Network
{
public:
    Network(const vector<unsigned> &topology);
    void feedForward(const vector<double> &inputVals);
    void backPropogate(const vector<double> &targetVals);
    vector<double> getResults();
    double m_error;
    double m_recentAverageError;
   // double m_recentAverageSmoothingFactor;
private:
    vector<Layer> m_layers;
}; // Network

vector<double> Network::getResults()
{
    vector<double> result;
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
        result.push_back(m_layers.back()[n].getOutputVal());
    return result;
} // Network::getResults

void Network::backPropogate(const vector<double> &targetVals)
{
    Layer &outputLayer = m_layers.back();
    m_error = 0.0;
    
    for (unsigned n = 0; n < outputLayer.size() - 1; ++n)
    {
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        m_error += delta * delta;
        outputLayer[n].calcOutputGradients(targetVals[n]);
    } // for n

    m_error /= outputLayer.size() - 1;
    m_error = sqrt(m_error);
    m_recentAverageError = m_recentAverageError *  m_error;

    for (unsigned long layerNum = m_layers.size() - 2; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum],
              &nextLayer = m_layers[layerNum + 1];

        for (unsigned n = 0; n < layer.size(); ++n)
            layer[n].calcHiddenGradients(nextLayer);
    } // for layerNum

    for (unsigned long layerNum = m_layers.size() - 1; layerNum > 0; --layerNum)
    {
        Layer &layer = m_layers[layerNum],
              &prevLayer = m_layers[layerNum - 1];

        for (unsigned n = 0; n < layer.size(); ++n)
            layer[n].updateInputWeights(prevLayer);
    } // for layerNum
} // Network::backPropogate

void Network::feedForward(const vector<double> &inputVals)
{
    for (unsigned i = 0; i < inputVals.size(); ++i)
        m_layers[0][i].setOutputVal(inputVals[i]);

    for (unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum)
    {
        Layer &prevLayer = m_layers[layerNum - 1];
        for (unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n)
            m_layers[layerNum][n].feedForward(prevLayer);
    } // for layerNum
} // Network::feedforward

Network::Network(const vector<unsigned> &topology)
{
    unsigned long numLayers = topology.size();
    for (unsigned layerNum = 0; layerNum < numLayers; ++layerNum)
    {
        m_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum] + 1;

        for (unsigned j = 0; j <= topology[layerNum]; ++j)
            m_layers.back().push_back(Neuron(numOutputs, j));
    } // for i
    m_layers.back().back().setOutputVal(1.0);
} // Network::Network


class TestValues {
public:
    TestValues(vector<double> inputs, vector<double> outputs)
    {
        m_inputs = inputs;
        m_outputs = outputs;
    } // TestValues
    vector<double> m_inputs, m_outputs;
};

string writeVector(vector<double> &vec)
{
    string output = "";
    for (int i = 0; i < vec.size(); i++)
        output += to_string(vec[i]) + " ";
    return  output;
} // writeVector

int main(int argc, const char * argv[])
{
    cout << "\n\t\t~BCR Neural Network~\n\n";
    
    Network n({2, 3, 2, 3, 2, 1});
    
    vector<TestValues> tests = {TestValues({1, 0}, {1}),
                                TestValues({0, 1}, {1}),
                                TestValues({1, 1}, {0}),
                                TestValues({0, 0}, {0})};

    unsigned long int i = 0;
    while (true)
    {
        int index = int(rand() % tests.size());
        TestValues tv = tests[index];
        n.feedForward(tv.m_inputs);
        n.backPropogate(tv.m_outputs);
        vector<double> results = n.getResults();
        
        cout << "run #" << ++i << endl;
        cout << "\tinputs: " << writeVector(tv.m_inputs) << endl;
        cout << "\texp out: " << writeVector(tv.m_outputs) << endl;
        cout << "\toutput: " << writeVector(results) << endl << endl;
    } // while true
    
    return 0;
} // int main

