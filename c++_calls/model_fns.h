#ifndef PREDICT__H
#define PREDICT__H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

void missing_activation_impl(const string &activation);
vector<double> read_input_from_file(const string &f_name);
double read_response_from_file(const string &f_name);
vector<vector<double> > read_multi_input_from_file(const string &f_name);
vector<double> read_multi_response_from_file(const string &f_name);
vector<vector<double> > read_scalers_from_file(const string &f_name);
//double untransform(double value, double mean, double scale);
//double transform(double value, double mean, double scale);

// layer class - base class for other layr classes
class Layer {
public:
	unsigned int layer_id;
	string layer_name;

	//  constructor sets parameter string to member variable  i.e. -> layer_name
	Layer(string name) : layer_name(name) {};
	virtual ~Layer() { };

	// virtual methods are expected to be redefined in derived class
	// virtual methods for derived classes can to be accessed
	// using pointer/reference to the base class
	virtual void load_weights(ifstream &input_fname) { };
	virtual vector<double> compute_output(vector<double> test_input) { };

	string get_layer_name() { return layer_name; }	// returns layer name
};

class LayerDense: public Layer {
public:
	unsigned int input_node_count;
	unsigned int output_weights;
	vector<vector<double> > layer_weights;
	vector<double> bias;

	LayerDense() : Layer("Dense") {};
	void load_weights(ifstream &fin);
	vector<double> compute_output(vector<double> test_input);

};

class LayerActivation: public Layer {
public:
	string activation_type;

	LayerActivation() : Layer("Activation") { };
	void load_weights(ifstream &fin);
	vector<double> compute_output(vector<double> test_input);
};

// keras model class
class KerasModel {
public:
	unsigned int input_node_count();
	unsigned int output_node_count();

	KerasModel(string &input_fname);	// constructor declaration
	~KerasModel();		// destructor declaration
	vector<double> compute_output(vector<double> test_input);

private:
	void load_weights(string &input_fname);
	unsigned int layers_count;
	vector<Layer *> layers;	// container with layers
};

//double one_point_NN(KerasModel& object, vector<vector<double> > scaler_properties, vector<double> data);

#endif
