#include "model_fns.h"

#include <iostream>
#include <fstream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

// error for missing implementation of activation function
// you can add your activation implementation in compute_output if required
void missing_activation_impl(const string &activation) {
	cout << "Activation " << activation << " not defined!" << endl;
	cout << "Please add its implementation before use." << endl;
	exit(1);
}

vector<double> read_input_from_file(const string &fname) {
	/* cout << "Reading input..." << endl; */
	ifstream fin(fname.c_str());
	int n_features;
	fin >> n_features;
	/* cout << "n_features: " << n_features << endl; */
	vector<double> input_data(n_features);
	for (unsigned i = 0; i < n_features; i++) {
		fin >> input_data[i];
		/* cout << "...reading: " << input_data[i] << endl; */
	}
	return input_data;
}

vector<double> read_metadata_from_file(const string &fname){
       ifstream fin(fname.c_str());
       int n_x_mean = 4;
       int n_x_std = 4;
       int n_y_mean = 1;
       int n_y_std = 1;

       vector<double> metadata(10);

       for (int i = 0; i < n_x_mean; i++){
	 fin >> metadata[i]; 
       }
       for (int i = 0; i < n_x_std; i++){
	 fin >> metadata[n_x_mean+i];
       }
       for (int i = 0; i < n_y_mean; i++){
	 fin >> metadata[n_x_mean+n_x_std+i]; 
       }
       for (int i = 0; i < n_y_std; i++){
	 fin >> metadata[n_x_mean+n_x_std+n_y_mean+i];
       }

       return metadata;
}

vector<vector<double> > read_multi_input_from_file(const string &fname) {
	cout << "Reading input..." << endl;
	ifstream fin(fname.c_str());
	int n_features;
	int N_samples;
	fin >> N_samples;
	fin >> n_features;
	cout << "N_samples: " << N_samples << endl;
	cout << "n_features: " << n_features << endl;
	vector<vector<double> > input_data(N_samples,vector<double>(n_features));
	for (unsigned i = 0; i < N_samples; i++){
		for (unsigned j = 0; j < n_features; j++) {
			fin >> input_data[i][j];
		}
	}
	return input_data;
}


double standardise(double value, double mean, double std){
       double new_value;
       new_value = (value-mean)/std;
       return new_value;
}

double destandardise(double value, double mean, double std){
       double new_value;
       new_value = new_value*std + mean;
       return new_value;
}

/*
double standardise_array(double array[][4], int legs, double means[4], double stds[4]){
       double new_array[legs*4] = {};
       for (int i = 0; i < legs; i++){
	 for (int j = 0; j < 4; j++){
	   double new_val = standardise(array[i][j], means[j], stds[j]);
	   new_array[i+j] = new_val;
	 }
       }
       return new_array;
}
*/


// KerasModel constructor
KerasModel::KerasModel(string &input_fname) {
	load_weights(input_fname);
}

// KerasModel destructor
KerasModel::~KerasModel() {
	for (unsigned int i = 0; i < layers.size(); i++) {
		delete layers[i];	// deallocate memory
	}
}

// load weights for all layers
void KerasModel::load_weights(string &input_fname) {
	/* cout << "###################################" << endl; */
	/* cout << "Reading weights from file " << input_fname << endl; */
	ifstream fin(input_fname.c_str(),ifstream::in);
	string tmp_str = "";
	string layer_type = "";
	int layer_id = 0;
	if(fin.is_open()) {
		// get layers count in layers_count var
		fin >> tmp_str >> layers_count;
		/* cout << "Getting layers and count: " << tmp_str << layers_count << endl; */

		// Now iterate over  each layer
		/* cout << "Iterating over layers..."<< endl; */
		for (unsigned int layer_index = 0; layer_index < layers_count; ++layer_index) {
			fin >> tmp_str >> layer_id >> layer_type;
			/* cout << tmp_str << layer_id << layer_type << endl; */
			// pointer to layer
			Layer *layer = 0L;
			if (layer_type == "Dense") {
				layer = new LayerDense();
			}
			else if(layer_type == "Activation") {
				layer = new LayerActivation();
			}
			// if none of above case is true, means layer not-defined
			if(layer == 0L) {
		      	/* cout << "Layer is empty, maybe layer " << layer_type << " is not defined? Cannot define network." << endl; */
			     return;
			}
			layer->load_weights(fin);
			layers.push_back(layer);
			/* cout << "Layer pushed back!" << endl; */
		}
	}
	/* cout << "Closing file " << input_fname << endl; */
	fin.close();
}

vector<double> KerasModel::compute_output(vector<double> test_input) {
	/* cout << "###################################" << endl; */
	/* cout << "KreasModel compute output" << endl; */
	/* cout << "for test input " << test_input[0] << ", " << test_input[1] << endl; */
	/* cout << "Layer count: " << layers_count << endl; */
	vector<double> response;
	for (unsigned int i = 0; i < layers_count; i++) {
		/* cout << "Processing layer to compute output " << layers[i]->layer_name << endl; */
		response = layers[i]->compute_output(test_input);
		test_input = response;
		/* cout << "Response size " << response.size() << endl; */
	}
	return response;
}

// load weights and bias from input file for Dense layer
void LayerDense::load_weights(ifstream &fin) {
	/* cout << "Loading weights for Dense layer" << endl; */
	fin >> input_node_count >> output_weights;
	/* cout << "Input node count " << input_node_count << " with output weights " << output_weights << endl; */
	double tmp_double;
	// read weights for all the input nodes
	/* cout << "Now read weights of all input modes..." << endl; */
	char tmp_char = ' ';
	for (unsigned int i = 0; i < input_node_count; i++) {
		fin >> tmp_char;	// for '['
		/* cout << "Input node " << i << endl; */
		vector<double> tmp_weights;
		for (unsigned int j = 0; j < output_weights; j++) {
			fin >> tmp_double;
			/* cout << tmp_double << endl; */
			tmp_weights.push_back(tmp_double);
		}
		fin >> tmp_char;	// for ']'
		layer_weights.push_back(tmp_weights);
	}
	// read and save bias values
	/* cout << "Saving biases..." << endl; */
	fin >> tmp_char;	// for '['
	for (unsigned int output_node_index = 0; output_node_index < output_weights; output_node_index++) {
		fin >> tmp_double;
		/* cout << tmp_double << endl; */
		bias.push_back(tmp_double);
	}
	fin >> tmp_char;	// for ']'
}

vector<double> LayerDense::compute_output(vector<double> test_input) {
	/* cout << "Inside dense layer compute output" << '\n'; */
    /* cout << "weights: input size " << layer_weights.size() << endl; */
    /* cout << "weights: neurons size " << layer_weights[0].size() << endl; */
    /* cout << "bias size " << bias.size() << endl; */
	vector<double> out(output_weights);
	double weighted_term = 0;
	for (size_t i = 0; i < output_weights; i++) {
		weighted_term = 0;
		for (size_t j = 0; j < input_node_count; j++) {
			weighted_term += (test_input[j] * layer_weights[j][i]);
		}
		out[i] = weighted_term + bias[i];
		/* cout << "...out[i]: " << out[i] << endl; */
	}
	return out;
}


vector<double> LayerActivation::compute_output(vector<double> test_input) {
	if (activation_type == "linear") {
		return test_input;
	}
	else if(activation_type == "relu") {
		for (unsigned int i = 0; i < test_input.size(); i++) {
			if(test_input[i] < 0) {
				test_input[i] = 0;
			}
		}
	}
	else if(activation_type == "softmax") {
		double sum = 0.0;
        for(unsigned int k = 0; k < test_input.size(); ++k) {
			test_input[k] = exp(test_input[k]);
			sum += test_input[k];
        }

        for(unsigned int k = 0; k < test_input.size(); ++k) {
			test_input[k] /= sum;
        }
	}
	else if (activation_type == "sigmoid") {
		double denominator = 0.0;
		for(unsigned int k = 0; k < test_input.size(); ++k) {
			denominator = 1 + exp(-(test_input[k]));
          	test_input[k] = 1/denominator;
        }
	}
	else if(activation_type == "softplus") {
		for (unsigned int k = 0; k < test_input.size(); ++k) {
			// log1p = natural logarithm (to base e) of 1 plus the given number (ln(1+x))
			test_input[k] = log1p(exp(test_input[k]));
		}
	}
	/*
	else if(activation_type == "softsign") {
		for (unsigned int k = 0; k < test_input.size(); ++k) {
			test_input[k] = test_input[k]/(1+abs(test_input[k]));
		}
	}
	*/
	else if(activation_type == "tanh") {
      		for(unsigned int k = 0; k < test_input.size(); ++k) {
        		test_input[k] = tanh(test_input[k]);
      		}
    	}
	else {
      missing_activation_impl(activation_type);
    }
	return test_input;
}

void LayerActivation::load_weights(ifstream &fin) {
	// cout << "Loading weights for Activation layer" << endl;
	fin >> activation_type;
}

