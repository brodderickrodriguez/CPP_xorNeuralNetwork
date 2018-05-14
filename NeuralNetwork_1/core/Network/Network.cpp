//
//  Network.cpp
//  NeuralNetwork_1
//
//  Created by bcr@brodderick.com on 5/13/18.
//  Copyright © 2018 BCR. All rights reserved.
//

#include "Network.hpp"

std::vector<double> Network::getResults()
{
    std::vector<double> result;
    for (unsigned n = 0; n < m_layers.back().size() - 1; ++n)
        result.push_back(m_layers.back()[n].getOutputVal());
    return result;
} // Network::getResults

void Network::backPropogate(const std::vector<double> &targetVals)
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

void Network::feedForward(const std::vector<double> &inputVals)
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

Network::Network(const std::vector<unsigned> &topology)
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


