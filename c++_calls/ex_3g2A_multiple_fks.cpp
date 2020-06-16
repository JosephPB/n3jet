#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "model_fns.h"

int main()
{
  std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
  std::cout.precision(16);

  std::cout << std::endl;
  std::cout << "n3jet: simple example of calling a pretrained neural network for inference in a C++ interface" << std::endl;
  std::cout << std::endl;

  const int legs = 5;
  const int pspoints = 2;
  const int pairs = 9;
  const int training_reruns = 20;

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
  
  std::string model_base = "./models/diphoton/3g2A/RAMBO/";
  std::string model_dirs[training_reruns] = {"events_100k_fks_all_legs_all_paires_all_save_0/",
					     "events_100k_fks_all_legs_all_paires_all_save_1/",
					     "events_100k_fks_all_legs_all_paires_all_save_2/",
					     "events_100k_fks_all_legs_all_paires_all_save_3/",
					     "events_100k_fks_all_legs_all_paires_all_save_4/",
					     "events_100k_fks_all_legs_all_paires_all_save_5/",
					     "events_100k_fks_all_legs_all_paires_all_save_6/",
					     "events_100k_fks_all_legs_all_paires_all_save_7/",
					     "events_100k_fks_all_legs_all_paires_all_save_8/",
					     "events_100k_fks_all_legs_all_paires_all_save_9/",
					     "events_100k_fks_all_legs_all_paires_all_save_10/",
					     "events_100k_fks_all_legs_all_paires_all_save_11/",
					     "events_100k_fks_all_legs_all_paires_all_save_12/",
					     "events_100k_fks_all_legs_all_paires_all_save_13/",
					     "events_100k_fks_all_legs_all_paires_all_save_14/",
					     "events_100k_fks_all_legs_all_paires_all_save_15/",
					     "events_100k_fks_all_legs_all_paires_all_save_16/",
					     "events_100k_fks_all_legs_all_paires_all_save_17/",
					     "events_100k_fks_all_legs_all_paires_all_save_18/",
					     "events_100k_fks_all_legs_all_paires_all_save_19/"
  };

  std::string pair_dirs[pairs] = {"pair_0.02_0",
				  "pair_0.02_1",
				  "pair_0.02_2",
				  "pair_0.02_3",
				  "pair_0.02_4",
				  "pair_0.02_5",
				  "pair_0.02_6",
				  "pair_0.02_7",
				  "pair_0.02_8"
				  
  };

  std::string cut_dirs = "cut_0.02";

  std::vector<std::vector<std::vector<double> > > metadatas(training_reruns, std::vector<std::vector<double> > (pairs+1, std::vector<double>(10)));
  std::string model_dir_models[training_reruns][pairs+1];

  for (int i = 0; i < training_reruns; i++){
    for (int j = 0; j < pairs; j++){
      std::string metadata_file = model_base + model_dirs[i] + pair_dirs[j] + "dataset_metadata.dat";
      std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
      for (int k = 0; k < 10 ; k++){
	metadatas[i][j][k] = metadata[k];
      };
      model_dir_models[i][j] = model_base + model_dirs[i] + pair_dirs[j] + "model.nnet";
    };
    std::string metadata_file = model_base + model_dirs[i] + cut_dirs + "dataset_metadata.dat";
    std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
    for (int k = 0; k < 10 ; k++){
      metadatas[i][pairs][k] = metadata[k];
    };
    model_dir_models[i][pairs] = model_base + model_dirs[i] + cut_dirs + "model.nnet";
  };

}
