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
  double network_input[2][20] = {
				 {0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, 1.414627037889425, 0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, -1.414627037889425, -1.207592812304212, -1.2719356679503657, 0.5589352631452073, -0.13966452671732135, -0.21505946958578623, 0.03947463359954325, -2.6866411241389296, 0.5413086837446464, -0.22688662799086254, 1.2324610343508227, 2.1277058609937223, -0.40164415702732503},
				 {0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, 1.414627037889425, 0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, -1.414627037889425, -1.207592812304212, -1.2719356679503657, 0.5589352631452073, -0.13966452671732135, -0.21505946958578623, 0.03947463359954325, -2.6866411241389296, 0.5413086837446464, -0.22688662799086254, 1.2324610343508227, 2.1277058609937223, -0.40164415702732503},
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

