#ifndef PREDICT__H
#define PREDICT__H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace nn {

void missing_activation_impl(const std::string& activation);
std::vector<double> read_input_from_file(const std::string& f_name);
std::vector<std::vector<double>> read_multi_input_from_file(const std::string& f_name);
std::vector<double> read_metadata_from_file(const std::string& fname);
double standardise(double value, double mean, double stnd);
double destandardise(double value, double mean, double stnd);
//double standardise_array(double array[][4], int legs, double means[4], double stds[4]);
//double untransform(double value, double mean, double scale);
//double transform(double value, double mean, double scale);

// layer class - base class for other layr classes
class Layer {
public:
    unsigned int layer_id;
    std::string layer_name;

    //  constructor sets parameter std::string to member variable  i.e. -> layer_name
    Layer(std::string name)
        : layer_name(name) {};
    virtual ~Layer() {};

    // virtual methods are expected to be redefined in derived class
    // virtual methods for derived classes can to be accessed
    // using pointer/reference to the base class
    virtual void load_weights(std::ifstream& input_fname) = 0;
    virtual std::vector<double> compute_output(std::vector<double> test_input) = 0;

    std::string get_layer_name() { return layer_name; } // returns layer name
};

class LayerDense : public Layer {
public:
    unsigned int input_node_count;
    unsigned int output_weights;
    std::vector<std::vector<double>> layer_weights;
    std::vector<double> bias;

    LayerDense()
        : Layer("Dense") {};
    void load_weights(std::ifstream& fin);
    std::vector<double> compute_output(std::vector<double> test_input);
};

class LayerActivation : public Layer {
public:
    std::string activation_type;

    LayerActivation()
        : Layer("Activation") {};
    void load_weights(std::ifstream& fin);
    std::vector<double> compute_output(std::vector<double> test_input);
};

// keras model class
class KerasModel {
public:
    unsigned int input_node_count();
    unsigned int output_node_count();

    KerasModel() {}; // constructor declaration
    ~KerasModel();   // destructor declaration
    void load_weights(std::string& input_fname);
    std::vector<double> compute_output(std::vector<double> test_input);

private:
    // void load_weights(std::string& input_fname);
    unsigned int layers_count;
    std::vector<Layer*> layers; // container with layers
};

//double one_point_NN(KerasModel& object, std::vector<std::vector<double> > scaler_properties, std::vector<double> data);

} // namespace nn

#endif // PREDICT__H
