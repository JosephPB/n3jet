#include <cmath>
#include <cstdlib>
#include <iostream>

#include "model_fns.h"

using namespace std;

//using std::cout;
//using std::endl;
//using std::ios_base;

int main()
{
  cout.setf(ios_base::scientific, ios_base::floatfield);
  cout.precision(16);

  cout << endl;
  cout << "n3jet: simple example of calling a pretrained neural network for inference in a C++ interface" << endl;
  cout << endl;

  const int legs = 5;
  const int pspoints = 2;

  //processed input - need to add scaling functions
  double network_input[2][12] = {
				 {-0.6891879533860993, -0.9852371318837948, 0.4329493931538896, -0.24218950063470177, 0.35079023542183335, 0.030576919705734094, -2.0810632662052093, 0.9386727101484141, 0.33839771796426593, 0.9546602121780606, 1.6481138730513194, -0.6964832095137122},
				 {-0.6891879533860993, -0.9852371318837948, 0.4329493931538896, -0.24218950063470177, 0.35079023542183335, 0.030576919705734094, -2.0810632662052093, 0.9386727101484141, 0.33839771796426593, 0.9546602121780606, 1.6481138730513194, -0.6964832095137122}
  };

  string dumpednn = "./tests/single_test_dumped.nnet";

  KerasModel kerasModel(dumpednn);
  
  for (int i = 0; i < pspoints; i++){
    // create iterator pointing to beginning and end of array
    vector<double> input_vec(begin(network_input[i]), end(network_input[i]));
    
    vector<double> result = kerasModel.compute_output(input_vec);

    double* out = &result[0];

    cout << "Loop( 0) = " << out[0] << endl;
    
  }
  
}

