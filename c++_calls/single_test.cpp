#include "model_fns.h"

#include <iostream>
#include <string>
#include <vector>

/*
This script will run a pretrained Keras model on a single data point saves in a .dat file.
The data will consists of the top line being the number of elements to be fed into the network
and the second line consistsing of space separated variables
 */

int main()
{
    std::cout << "This is simple example with Keras neural network model loading into C++.\n"
              << "Keras model will be used in C++ for prediction only on one point.\n"
              << "Note: this example will take in already processed data.\n";

    std::string datafile { "./tests/data/single_test_sample.dat" };
    std::string dumpednn { "./tests/single_test_dumped.nnet" };

    std::vector<double> sample { nn::read_input_from_file(datafile) };
    std::cout << "First two sample elements = " << sample[0] << ", " << sample[1] << '\n';

    // load model
    nn::KerasModel kerasModel;
    kerasModel.load_weights(dumpednn);

    // infer on model
    std::vector<double> result { kerasModel.compute_output(sample) };

    std::cout << "Standardised C++ Model output is: " << result[0] << std::endl;

    double NN_output_de = -0.21683742;
    
    std::cout << "Standardised NN  Model output is: " << NN_output << std::endl;

    std::cout << "This NN number needs updating each time a new python model is trained" << std::endl; 
    
    return 0;
}
