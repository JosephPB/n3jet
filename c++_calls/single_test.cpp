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

    //destandardise model result
    std::string metadata_file { "./tests/data/single_test_dataset_metadata.dat" };
    std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);

    std::cout << "Using y_mean " << metadata[8] << std::endl;
    
    double output { nn::destandardise(result[0], metadata[8], metadata[9]) };

    std::string python_infer { "./tests/data/single_test_infer.dat" };
    
    std::ifstream fin(python_infer.c_str());
    double python_outputs[2];
    fin >> python_outputs[0];
    fin >> python_outputs[1];

    std::cout << "               " << " Python NN " << "  C++ NN  " << std::endl;
    
    std::cout << "Standardised   " << python_outputs[0] << "   " << result[0] << std::endl;
    std::cout << "Destandardised " << python_outputs[1] << "  " << output << std::endl;
    
    return 0;
}
