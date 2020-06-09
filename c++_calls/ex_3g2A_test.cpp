#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "model_fns.h"

using namespace std;

int main()
{
  cout.setf(ios_base::scientific, ios_base::floatfield);
  cout.precision(16);

  cout << endl;
  cout << "n3jet: simple example of calling a pretrained neural network for inference in a C++ interface" << endl;
  cout << endl;

  const int legs = 5;
  const int pspoints = 2;

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

  /*
  double x_means[4] = {4.00000000e+02,  4.54747351e-15,  7.95807864e-15, -9.09494702e-15};
  double x_stds[4] = {121.24600323, 188.36617697, 119.5484733 , 353.45005193};
  double y_mean = 1.8749375283902703e-07;
  double y_std = 3.8690694630335114e-07;
  */

  string metadata_file = "./tests/data/single_test_dataset_metadata.dat";
  // metadata is in the format: {x_means,x_stds,y_mean,y_std}
  vector<double> metadata = read_metadata_from_file(metadata_file);
  cout << "Metadata test = " << metadata[9] << endl;
  
  string dumpednn = "./tests/single_test_dumped.nnet";
  //string dumpednn = "./models/diphoton/3g2A/RAMBO/events_100k_single_all_legs_all_save_0/model.nnet";

  KerasModel kerasModel(dumpednn);

  int proc_len = legs*4;
  
  for (int i = 0; i < pspoints; i++){
    cout << "==================== Test point " << i+1 << " ====================" << endl;
    double mom[legs*4];

    // flatten momenta
    for (int p = 0; p < legs; p++){
      for (int mu = 0; mu < 4; mu++){
	// standardise input
	cout << Momenta[i][p][mu] << " ";
	mom[p*4+mu] = standardise(Momenta[i][p][mu], metadata[p], metadata[4+p]);
      }
      cout << endl;
    }
    cout << endl;
    
    vector<double> input_vec(begin(mom), end(mom));
    
    vector<double> result = kerasModel.compute_output(input_vec);

    // destandardise output
    double output = destandardise(result[0], metadata[8], metadata[9]);

    cout << "Loop( 0) = " << output << endl;
    
  }
  
}
