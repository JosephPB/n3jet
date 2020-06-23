#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "model_fns.h"

int main()
{
    std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
    std::cout.precision(16);

    std::cout
        << '\n'
        << "n3jet: simple example of calling a pretrained neural network for inference in a C++ interface" << '\n'
        << '\n';

    const int d { 4 };
    const int legs { 5 };
    const int pspoints { 2 };

    //raw momenta input

    double momenta[pspoints][legs][d] {
        { { 500., 0., 0., 500. },
            { 500., 0., 0., -500. },
            { 253.58419798, -239.58965912, 66.81985738, -49.36443422 },
            { 373.92489886, 7.43568582, -321.18384469, 191.32558238 },
            { 372.49090317, 232.1539733, 254.36398731, -141.96114816 } },
        { { 500., 0., 0., 500. },
            { 500., 0., 0., -500. },
            { 253.58419798, -239.58965912, 66.81985738, -49.36443422 },
            { 373.92489886, 7.43568582, -321.18384469, 191.32558238 },
            { 372.49090317, 232.1539733, 254.36398731, -141.96114816 } }
    };

    std::string metadata_file { "./tests/data/single_test_dataset_metadata.dat" };
    std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);

    std::string dumpednn { "./tests/single_test_dumped.nnet" };

    nn::KerasModel kerasModel;
    kerasModel.load_weights(dumpednn);

    std::string python_infer { "./tests/data/single_test_infer.dat" };
    
    std::ifstream fin(python_infer.c_str());
    double python_outputs[2];
    fin >> python_outputs[0];
    fin >> python_outputs[1];
  
    for (int i { 0 }; i < pspoints; ++i) {
        std::cout << "==================== Test point " << i + 1 << " ====================" << '\n';

        std::vector<double> mom(legs * d); // standard four-momenta, flattened

        // flatten momenta
        for (int p { 0 }; p < legs; ++p) {
            for (int mu { 0 }; mu < d; ++mu) {
                // standardise input
#ifdef DEBUG
                std::cout << momenta[i][p][mu] << " ";
#endif
                mom[p * d + mu] = nn::standardise(momenta[i][p][mu], metadata[p], metadata[4+p]);
            }
#ifdef DEBUG
            std::cout << '\n';
#endif
        }
#ifdef DEBUG
        std::cout << '\n';
#endif

        std::vector<double> result { kerasModel.compute_output(mom) };

        // destandardise output
        double output { nn::destandardise(result[0], metadata[8], metadata[9]) };

        std::cout << "Loop( 0) = " << output << '\n';

	std::cout << "               " << " Python NN " << "  C++ NN  " << std::endl;
    
	std::cout << "Standardised   " << python_outputs[0] << "   " << result[0] << std::endl;
	std::cout << "Destandardised " << python_outputs[1] << "  " << output << std::endl;
	
    }
}
