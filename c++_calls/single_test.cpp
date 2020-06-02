#include "model_fns.h"

#include <iostream>

using namespace std;

/*
This script will run a pretrained Keras model on a single data point saves in a .dat file.
The data will consists of the top line being the number of elements to be fed into the network
and the second line consistsing of space separated variables
 */


int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\n"
           << "Keras model will be used in C++ for prediction only on one point." << endl;

  string datafile = "./tests/data/single_test_sample.dat";
  string dumpednn = "./tests/single_test_dumped.nnet";
  
  vector<double> sample = read_input_from_file(datafile);
  cout << "First two sample elements = " << sample[0] << ", " << sample[1] << endl;

  // load model
  KerasModel kerasModel(dumpednn);

  // infer on model
  vector<double> result = kerasModel.compute_output(sample);

  cout << result[0] << endl;
  
  return 0;
}
