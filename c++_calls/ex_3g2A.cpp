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

  /*
  
  //processed input - need to add scaling functions
  double network_input[pspoints][legs*4] = {
				 {0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, 1.414627037889425, 0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, -1.414627037889425, -1.207592812304212, -1.2719356679503657, 0.5589352631452073, -0.13966452671732135, -0.21505946958578623, 0.03947463359954325, -2.6866411241389296, 0.5413086837446464, -0.22688662799086254, 1.2324610343508227, 2.1277058609937223, -0.40164415702732503},
				 {0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, 1.414627037889425, 0.8247694549404301, -2.4141666949495767e-17, -6.656779815797403e-17, -1.414627037889425, -1.207592812304212, -1.2719356679503657, 0.5589352631452073, -0.13966452671732135, -0.21505946958578623, 0.03947463359954325, -2.6866411241389296, 0.5413086837446464, -0.22688662799086254, 1.2324610343508227, 2.1277058609937223, -0.40164415702732503},
  };

  */

  //raw momenta input

  double Momenta[pspoints][legs][4] = {
				     {
				       {500.,   0.,   0., 500.},
				       {500.,   0.,   0., -500.},
				       {253.58419798, -239.58965912, 66.81985738, -49.36443422},
				       {373.92489886,    7.43568582, -321.18384469,  191.32558238},
				       {372.49090317,  232.1539733 ,  254.36398731, -141.96114816}
				     },
				     {
				       {500.,   0.,   0., 500.},
				       {500.,   0.,   0., -500.},
				       {253.58419798, -239.58965912, 66.81985738, -49.36443422},
				       {373.92489886,    7.43568582, -321.18384469,  191.32558238},
				       {372.49090317,  232.1539733 ,  254.36398731, -141.96114816}
				     }
  };

  double x_means[4] = {4.00000000e+02,  4.54747351e-15,  7.95807864e-15, -9.09494702e-15};
  double x_stds[4] = {121.24600323, 188.36617697, 119.5484733 , 353.45005193};
  double y_mean = 1.8749375283902703e-07;
  double y_std = 3.8690694630335114e-07;


  string dumpednn = "./tests/single_test_dumped.nnet";

  KerasModel kerasModel(dumpednn);

  int proc_len = legs*4;
  
  for (int i = 0; i < pspoints; i++){
    // create iterator pointing to beginning and end of array

    //double network_input[proc_len] = {};
    double mom[legs*4];

    // standardise
    for (int p = 0; p < legs; p++){
      for (int mu = 0; mu < 4; mu++){
	mom[p*4+mu] = (Momenta[i][p][mu]-x_means[p])/x_stds[p];
      }
    }
    
    
    //double network_input = standardise_array(Momenta[i], legs, x_means, x_stds);
    
    vector<double> input_vec(begin(mom), end(mom));
    
    vector<double> result = kerasModel.compute_output(input_vec);
    
    //double* out = &result[0];

    cout << "Loop( 0) = " << result[0] << endl;
    
  }
  
}

