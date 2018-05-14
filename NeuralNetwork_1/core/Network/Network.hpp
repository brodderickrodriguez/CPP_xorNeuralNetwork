//
//  Network.hpp
//  NeuralNetwork_1
//
//  Created by bcr@brodderick.com on 5/13/18.
//  Copyright © 2018 BCR. All rights reserved.
//

#ifndef Network_hpp
#define Network_hpp
#include "Neuron.hpp"

class Network
{
public:
    Network(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backPropogate(const std::vector<double> &targetVals);
    std::vector<double> getResults();
    double m_error;
    double m_recentAverageError;
private:
    std::vector<Layer> m_layers;
}; // Network

#endif /* Network_hpp */
