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

    double x_means[4] = { 4.00000000e+02, 4.54747351e-15, 7.95807864e-15, -9.09494702e-15 };
    double x_stds[4] = { 121.24600323, 188.36617697, 119.5484733, 353.45005193 };
    double y_mean = 1.8749375283902703e-07;
    double y_std = 3.8690694630335114e-07;

    std::string dumpednn { "./tests/single_test_dumped.nnet" };

    nn::KerasModel kerasModel(dumpednn);

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
                mom[p * d + mu] = nn::standardise(momenta[i][p][mu], x_means[p], x_stds[p]);
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
        double output { nn::destandardise(result[0], y_mean, y_std) };

        std::cout << "Loop( 0) = " << output << '\n';
    }
}
