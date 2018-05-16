//
//  main.cpp
//  NeuralNetwork_1
//
//  Created by bcr@brodderick.com on 2/26/18.
//  Copyright © 2018 BCR. All rights reserved.
//

#include <iostream>
#include <vector>


#include "Network.hpp"
#include "Trainer.hpp"

std::string writeVector(std::vector<double> &vec) {
    std::string output = "";
    for (int i = 0; i < vec.size(); i++)
        output += std::to_string(vec[i]) + " ";
    return  output;
} // writeVector'


int main(int argc, const char * argv[]) {
    std::cout << "\n\t\t~BCR Neural Network~\n\n";
    
    Network n({2, 1});
    
    std::vector<Trainer> tests = {Trainer({1, 0}, {1}),
        Trainer({0, 1}, {1}),
        Trainer({1, 1}, {0}),
        Trainer({0, 0}, {0})};
    
    unsigned long int i = 0;
    while (i < 1000000) {
        int index = int(rand() % tests.size());
        Trainer tv = tests[index];
        n.feedForward(tv.m_inputs);
        n.backPropogate(tv.m_outputs);
        
        std::vector<double> results;
        n.getResults(results);
        
        std::vector<double> correctness;
        for (int i = 0; i < tv.m_outputs.size(); i++)
            correctness.push_back(1 - abs(tv.m_outputs[i] - results[i]));
        
        
        std::cout << "run #" << ++i << std::endl;
        std::cout << "\tinputs: " << writeVector(tv.m_inputs) << std::endl;
        std::cout << "\texp out: " << writeVector(tv.m_outputs) << std::endl;
        std::cout << "\tact out: " << writeVector(results) << std::endl;
        std::cout << "\tcorrectness: " << writeVector(correctness) << std::endl << std::endl;

    } // while true
    
    return 0;
} // int main
 
 
 
 

