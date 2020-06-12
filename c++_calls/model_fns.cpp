#include "model_fns.h"

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// error for missing implementation of activation function
// you can add your activation implementation in compute_output if required
void nn::missing_activation_impl(const std::string& activation)
{
    std::cout << "Activation " << activation << " not defined!" << '\n';
    std::cout << "Please add its implementation before use." << '\n';
    exit(1);
}

std::vector<double> nn::read_input_from_file(const std::string& fname)
{
#ifdef DEBUG
    std::cout << "Reading input..." << '\n';
#endif
    std::ifstream fin(fname.c_str());

    if (!fin.good()) {
        std::cerr << "Error: no input file!\n";
        std::exit(EXIT_FAILURE);
    }

    int n_features;
    fin >> n_features;
#ifdef DEBUG
    std::cout << "n_features: " << n_features << '\n';
#endif
    std::vector<double> input_data(n_features);
    for (int i { 0 }; i < n_features; ++i) {
        fin >> input_data[i];
#ifdef DEBUG
        std::cout << "...reading: " << input_data[i] << '\n';
#endif
    }
    return input_data;
}

// Save s and t vals from file
std::vector<std::vector<double>> nn::read_multi_input_from_file(const std::string& fname)
{
    std::cout << "Reading input..." << '\n';
    std::ifstream fin(fname.c_str());

    if (!fin.good()) {
        std::cerr << "Error: no input file!\n";
        std::exit(EXIT_FAILURE);
    }

    int n_features;
    int N_samples;
    fin >> N_samples;
    fin >> n_features;
    std::cout << "N_samples: " << N_samples << '\n';
    std::cout << "n_features: " << n_features << '\n';
    std::vector<std::vector<double>> input_data(N_samples, std::vector<double>(n_features));
    for (int i { 0 }; i < N_samples; ++i) {
        for (int j { 0 }; j < n_features; ++j) {
            fin >> input_data[i][j];
        }
    }
    return input_data;
}

double nn::standardise(double value, double mean, double stnd)
{
    double new_value;
    new_value = (value - mean) / stnd;
    return new_value;
}

double nn::destandardise(double value, double mean, double stnd)
{
    double new_value;
    new_value = value * stnd + mean;
    return new_value;
}
/*
double standardise_array(double array[][4], int legs, double means[4], double stds[4]){
       double new_array[legs*4] = {};
       for (int i { 0 }; i < legs; ++i){
	 for (int j { 0 }; j < 4; ++j){
	   double new_val = standardise(array[i][j], means[j], stds[j]);
	   new_array[i+j] = new_val;
	 }
       }
       return new_array;
}
*/

// KerasModel constructor
nn::KerasModel::KerasModel(std::string& input_fname)
{
    load_weights(input_fname);
}

// KerasModel destructor
nn::KerasModel::~KerasModel()
{
    for (unsigned int i { 0 }; i < layers.size(); ++i) {
        delete layers[i]; // deallocate memory
    }
}

// load weights for all layers
void nn::KerasModel::load_weights(std::string& input_fname)
{
#ifdef DEBUG
    std::cout << "###################################" << '\n';
    std::cout << "Reading weights from file " << input_fname << '\n';
#endif
    std::ifstream fin(input_fname.c_str(), std::ifstream::in);

    if (!fin.good()) {
        std::cerr << "Error: no nnet file!\n";
        std::exit(EXIT_FAILURE);
    }

    std::string tmp_str = "";
    std::string layer_type = "";
    int layer_id = 0;
    if (fin.is_open()) {
        // get layers count in layers_count var
        fin >> tmp_str >> layers_count;
#ifdef DEBUG
        std::cout << "Getting layers and count: " << tmp_str << layers_count << '\n';
#endif

        // Now iterate over  each layer
#ifdef DEBUG
        std::cout << "Iterating over layers..." << '\n';
#endif
        for (unsigned int layer_index = 0; layer_index < layers_count; ++layer_index) {
            fin >> tmp_str >> layer_id >> layer_type;
#ifdef DEBUG
            std::cout << tmp_str << layer_id << layer_type << '\n';
#endif
            // pointer to layer
            Layer* layer = 0L;
            if (layer_type == "Dense") {
                layer = new LayerDense();
            } else if (layer_type == "Activation") {
                layer = new LayerActivation();
            }
            // if none of above case is true, means layer not-defined
            if (layer == 0L) {
#ifdef DEBUG
                std::cout << "Layer is empty, maybe layer " << layer_type << " is not defined? Cannot define network." << '\n';
#endif
                return;
            }
            layer->load_weights(fin);
            layers.push_back(layer);
#ifdef DEBUG
            std::cout << "Layer pushed back!" << '\n';
#endif
        }
    }
#ifdef DEBUG
    std::cout << "Closing file " << input_fname << '\n';
#endif
    fin.close();
}

std::vector<double> nn::KerasModel::compute_output(std::vector<double> test_input)
{
#ifdef DEBUG
    std::cout << "###################################" << '\n';
    std::cout << "KerasModel compute output" << '\n';
    std::cout << "for test input " << test_input[0] << ", " << test_input[1] << '\n';
    std::cout << "Layer count: " << layers_count << '\n';
#endif
    std::vector<double> response;
    for (unsigned int i { 0 }; i < layers_count; ++i) {
#ifdef DEBUG
        std::cout << "Processing layer to compute output " << layers[i]->layer_name << '\n';
#endif
        response = layers[i]->compute_output(test_input);
        test_input = response;
#ifdef DEBUG
        std::cout << "Response size " << response.size() << '\n';
#endif
    }
    return response;
}

// load weights and bias from input file for Dense layer
void nn::LayerDense::load_weights(std::ifstream& fin)
{
#ifdef DEBUG
    std::cout << "Loading weights for Dense layer" << '\n';
#endif
    fin >> input_node_count >> output_weights;
#ifdef DEBUG
    std::cout << "Input node count " << input_node_count << " with output weights " << output_weights << '\n';
#endif
    double tmp_double;
    // read weights for all the input nodes
#ifdef DEBUG
    std::cout << "Now read weights of all input modes..." << '\n';
#endif
    char tmp_char = ' ';
    for (unsigned int i { 0 }; i < input_node_count; ++i) {
        fin >> tmp_char; // for '['
#ifdef DEBUG
        std::cout << "Input node " << i << '\n';
#endif
        std::vector<double> tmp_weights;
        for (unsigned int j { 0 }; j < output_weights; ++j) {
            fin >> tmp_double;
#ifdef DEBUG
            std::cout << tmp_double << '\n';
#endif
            tmp_weights.push_back(tmp_double);
        }
        fin >> tmp_char; // for ']'
        layer_weights.push_back(tmp_weights);
    }
    // read and save bias values
#ifdef DEBUG
    std::cout << "Saving biases..." << '\n';
#endif
    fin >> tmp_char; // for '['
    for (unsigned int output_node_index = 0; output_node_index < output_weights; output_node_index++) {
        fin >> tmp_double;
#ifdef DEBUG
        std::cout << tmp_double << '\n';
#endif
        bias.push_back(tmp_double);
    }
    fin >> tmp_char; // for ']'
}

std::vector<double> nn::LayerDense::compute_output(std::vector<double> test_input)
{
#ifdef DEBUG
    std::cout << "Inside dense layer compute output" << '\n';
    std::cout << "weights: input size " << layer_weights.size() << '\n';
    std::cout << "weights: neurons size " << layer_weights[0].size() << '\n';
    std::cout << "bias size " << bias.size() << '\n';
#endif
    std::vector<double> out(output_weights);
    double weighted_term = 0;
    for (size_t i { 0 }; i < output_weights; ++i) {
        weighted_term = 0;
        for (size_t j { 0 }; j < input_node_count; ++j) {
            weighted_term += (test_input[j] * layer_weights[j][i]);
        }
        out[i] = weighted_term + bias[i];
#ifdef DEBUG
        std::cout << "...out[i]: " << out[i] << '\n';
#endif
    }
    return out;
}

std::vector<double> nn::LayerActivation::compute_output(std::vector<double> test_input)
{
    if (activation_type == "linear") {
        return test_input;
    } else if (activation_type == "relu") {
        for (unsigned int i { 0 }; i < test_input.size(); ++i) {
            if (test_input[i] < 0) {
                test_input[i] = 0;
            }
        }
    } else if (activation_type == "softmax") {
        double sum = 0.0;
        for (unsigned int k { 0 }; k < test_input.size(); ++k) {
            test_input[k] = std::exp(test_input[k]);
            sum += test_input[k];
        }

        for (unsigned int k { 0 }; k < test_input.size(); ++k) {
            test_input[k] /= sum;
        }
    } else if (activation_type == "sigmoid") {
        double denominator = 0.0;
        for (unsigned int k { 0 }; k < test_input.size(); ++k) {
            denominator = 1 + std::exp(-(test_input[k]));
            test_input[k] = 1 / denominator;
        }
    } else if (activation_type == "softplus") {
        for (unsigned int k { 0 }; k < test_input.size(); ++k) {
            // log1p = natural logarithm (to base e) of 1 plus the given number (ln(1+x))
            test_input[k] = std::log1p(std::exp(test_input[k]));
        }
    }
    /*
	else if(activation_type == "softsign") {
		for (unsigned int k { 0 }; k < test_input.size(); ++k) {
			test_input[k] = test_input[k]/(1+abs(test_input[k]));
		}
	}
	*/
    else if (activation_type == "tanh") {
        for (unsigned int k { 0 }; k < test_input.size(); ++k) {
            test_input[k] = std::tanh(test_input[k]);
        }
    } else {
        missing_activation_impl(activation_type);
    }
    return test_input;
}

void nn::LayerActivation::load_weights(std::ifstream& fin)
{
#ifdef DEBUG
    std::cout << "Loading weights for Activation layer" << '\n';
#endif
    fin >> activation_type;
}
