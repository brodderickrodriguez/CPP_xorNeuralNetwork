//
//  Trainer.hpp
//  NeuralNetwork_1
//
//  Created by bcr@brodderick.com on 5/13/18.
//  Copyright © 2018 BCR. All rights reserved.
//

#ifndef Trainer_hpp
#define Trainer_hpp
#include <vector>

class Trainer {
public:
    Trainer(std::vector<double> inputs, std::vector<double> outputs);
    std::vector<double> m_inputs, m_outputs;
};

#endif /* Trainer_hpp */
