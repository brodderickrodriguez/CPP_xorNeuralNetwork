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

class Network {
public:
    Network(const std::vector<unsigned> &topology);
    void feedForward(const std::vector<double> &inputVals);
    void backPropogate(const std::vector<double> &targetVals);
    void getResults(std::vector<double> &resultVals) const;
    double getRecentAverageError(void) const { return m_recentAverageError; }
    double getNumberOfRuns() { return m_runs; }
    void setNumberOfRuns(unsigned long r) { m_runs = r; }
    
private:
    std::vector<Layer> m_layers;
    double m_error;
    double m_recentAverageError;
    unsigned long m_runs = 0;
}; // Network

#endif /* Network_hpp */
