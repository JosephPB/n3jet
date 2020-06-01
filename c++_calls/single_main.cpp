#include "model_fns.h"

#include <iostream>

using namespace std;


int main() {
  cout << "This is simple example with Keras neural network model loading into C++.\n"
           << "Keras model will be used in C++ for prediction only." << endl;

  string datafile = "./tests/single_test_sample.dat";
  string dumpednn = "./tests/single_test_dumped.nnet";
  
  //DataChunk *sample = new DataChunk2D();
  //sample->read_from_file("./tests/single_test_sample.dat");
  //std::cout << sample->get_3d().size() << std::endl;
  vector<double> sample = read_input_from_file(datafile);
  cout << sample[0] << sample[1] << endl;
  //KerasModel m("./tests/single_test_dumped.nnet", true);
  //m.compute_output(sample);
  //delete sample;
  KerasModel kerasModel(dumpednn);
  vector<double> result = kerasModel.compute_output(sample);

  cout << result[0] << endl;
  
  return 0;
}
