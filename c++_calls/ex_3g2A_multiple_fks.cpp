#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "model_fns.h"

int pair_check(double p1[], double p2[], int delta, float s_com);

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
  const int delta = 0.02;
  const float s_com = 1000.;

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
  std::string model_dirs[training_reruns] = {"events_100k_fks_all_legs_all_pairs_all_save_0/",
					     "events_100k_fks_all_legs_all_pairs_all_save_1/",
					     "events_100k_fks_all_legs_all_pairs_all_save_2/",
					     "events_100k_fks_all_legs_all_pairs_all_save_3/",
					     "events_100k_fks_all_legs_all_pairs_all_save_4/",
					     "events_100k_fks_all_legs_all_pairs_all_save_5/",
					     "events_100k_fks_all_legs_all_pairs_all_save_6/",
					     "events_100k_fks_all_legs_all_pairs_all_save_7/",
					     "events_100k_fks_all_legs_all_pairs_all_save_8/",
					     "events_100k_fks_all_legs_all_pairs_all_save_9/",
					     "events_100k_fks_all_legs_all_pairs_all_save_10/",
					     "events_100k_fks_all_legs_all_pairs_all_save_11/",
					     "events_100k_fks_all_legs_all_pairs_all_save_12/",
					     "events_100k_fks_all_legs_all_pairs_all_save_13/",
					     "events_100k_fks_all_legs_all_pairs_all_save_14/",
					     "events_100k_fks_all_legs_all_pairs_all_save_15/",
					     "events_100k_fks_all_legs_all_pairs_all_save_16/",
					     "events_100k_fks_all_legs_all_pairs_all_save_17/",
					     "events_100k_fks_all_legs_all_pairs_all_save_18/",
					     "events_100k_fks_all_legs_all_pairs_all_save_19/"
  };

  std::string pair_dirs[pairs] = {"pair_0.02_0/",
				  "pair_0.02_1/",
				  "pair_0.02_2/",
				  "pair_0.02_3/",
				  "pair_0.02_4/",
				  "pair_0.02_5/",
				  "pair_0.02_6/",
				  "pair_0.02_7/",
				  "pair_0.02_8/"
				  
  };

  std::string cut_dirs = "cut_0.02/";

  std::vector<std::vector<std::vector<double> > > metadatas(training_reruns, std::vector<std::vector<double> > (pairs+1, std::vector<double>(10)));
  std::string model_dir_models[training_reruns][pairs+1];
  std::vector<std::vector<nn::KerasModel> > kerasModels(training_reruns, std::vector<nn::KerasModel>(training_reruns));

  for (int i = 0; i < training_reruns; i++){

    // Near networks
    for (int j = 0; j < pairs; j++){
      std::string metadata_file = model_base + model_dirs[i] + pair_dirs[j] + "dataset_metadata.dat";
      std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
      for (int k = 0; k < 10 ; k++){
	metadatas[i][j][k] = metadata[k];
      };
      model_dir_models[i][j] = model_base + model_dirs[i] + pair_dirs[j] + "model.nnet";
#ifdef DEBUG
      std::cout << "Loading from: " << model_dir_models[i][j] << std::endl;
#endif
      kerasModels[i][j].load_weights(model_dir_models[i][j]);
    };

    // Cut networks
    std::string metadata_file = model_base + model_dirs[i] + cut_dirs + "dataset_metadata.dat";
    std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
    for (int k = 0; k < 10 ; k++){
      metadatas[i][pairs][k] = metadata[k];
    };
    model_dir_models[i][pairs] = model_base + model_dirs[i] + cut_dirs + "model.nnet";
#ifdef DEBUG
    std::cout << "Loading from: " << model_dir_models[i][pairs] << std::endl;
#endif
    kerasModels[i][pairs].load_weights(model_dir_models[i][pairs]);
  }

  for (int i = 0; i < pspoints; i++){
    std::cout << "==================== Test point " << i+1 << " ====================" << std::endl;


    // standardise momenta
    double moms[training_reruns][pairs+1][legs*4];

    // flatten momenta
    for (int p = 0; p < legs; p++){
      for (int mu = 0; mu < 4; mu++){
	// standardise input
	std::cout << Momenta[i][p][mu] << " ";
	for (int k = 0; k < training_reruns; k++){
	  for (int j = 0; j < pairs; j++){
	    moms[k][j][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][j][p], metadatas[k][j][4+p]);
	  }
	  moms[k][pairs][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][pairs][p], metadatas[k][pairs][4+p]);
	}
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    //cut/near check
    int cut_near = 0;
    for (int j; j < legs-1; j++){
      for (int k = j+1; k < legs; k++){
	int check = pair_check(Momenta[i][j], Momenta[i][k], delta, s_com);
	  cut_near += check;
      }
    }

    std::cout << "Cut/near check is: " << cut_near <<std::endl;


    /*
    // inference
    for (int j = 0; j < training_reruns; j++){
	for (int k = 0; k < pairs; k++){
	}
    }

    */
  }
}

int pair_check(double p1[], double p2[], int delta, float s_com){
  double prod = p1[0]*p2[0]-(p1[1]*p2[1]+p1[2]*p2[2]+p1[3]*p2[3]);
  double distance = prod/s_com;

  if (distance <= delta){
    return 1;
  }
  else{
    return 0;
  }
}
