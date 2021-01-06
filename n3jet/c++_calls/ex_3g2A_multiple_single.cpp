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

  const int d = 4;
  const int legs = 5;
  const int pspoints = 2;
  const int training_reruns = 20;

  //raw momenta input

  double Momenta[pspoints][legs][d] = {
				     {
				       {500.,   0.,   0., 500.},
				       {500.,   0.,   0., -500.},
				       {  95.28866002,  -65.55901358,   63.31165884,  -27.81327599},
				       { 458.94891647,  105.2561955 , -226.16717764, -385.23194182},
				       { 445.76242351,  -39.69718192,  162.8555188 ,  413.04521781}
				     },
				     {
				       {500.,   0.,   0., 500.},
				       {500.,   0.,   0., -500.},
				       {469.28650056, -307.66538241, -354.34001509,   3.871147472},
				       {388.66911749,  246.28196866,  280.42564321, -108.4911675},
				       {142.04438195,   61.38341374,   73.91437188,  104.62002003}
				     }
  };

  double python_outputs[2] = {3.29613328631e-07, 3.3791138776e-08};
  
  std::string model_base = "./models/diphoton/3g2A/RAMBO/";
  std::string model_dirs[training_reruns] = {"events_100k_single_all_legs_all_save_0/",
					     "events_100k_single_all_legs_all_save_1/",
					     "events_100k_single_all_legs_all_save_2/",
					     "events_100k_single_all_legs_all_save_3/",
					     "events_100k_single_all_legs_all_save_4/",
					     "events_100k_single_all_legs_all_save_5/",
					     "events_100k_single_all_legs_all_save_6/",
					     "events_100k_single_all_legs_all_save_7/",
					     "events_100k_single_all_legs_all_save_8/",
					     "events_100k_single_all_legs_all_save_9/",
					     "events_100k_single_all_legs_all_save_10/",
					     "events_100k_single_all_legs_all_save_11/",
					     "events_100k_single_all_legs_all_save_12/",
					     "events_100k_single_all_legs_all_save_13/",
					     "events_100k_single_all_legs_all_save_14/",
					     "events_100k_single_all_legs_all_save_15/",
					     "events_100k_single_all_legs_all_save_16/",
					     "events_100k_single_all_legs_all_save_17/",
					     "events_100k_single_all_legs_all_save_18/",
					     "events_100k_single_all_legs_all_save_19/"
  };
  
  std::vector<std::vector<double> > metadatas(training_reruns, std::vector<double>(10));
  std::string model_dir_models[training_reruns];
  std::vector<nn::KerasModel> kerasModels(training_reruns);

  for (int i = 0; i < training_reruns; i++){
    std::string metadata_file = model_base + model_dirs[i] + "dataset_metadata.dat";
    std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
    for (int j = 0; j < 10 ; j++){
      metadatas[i][j] = metadata[j];
    };
    model_dir_models[i] = model_base + model_dirs[i] + "model.nnet";
    kerasModels[i].load_weights(model_dir_models[i]);
  };

  for (int i = 0; i < pspoints; i++){
    std::cout << "==================== Test point " << i+1 << " ====================" << std::endl;
    double moms[training_reruns][legs*d];

    // flatten momenta
    for (int p = 0; p < legs; p++){
      for (int mu = 0; mu < d; mu++){
	// standardise input
	std::cout << Momenta[i][p][mu] << " ";
	for (int k = 0; k < training_reruns; k++){
	  moms[k][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][mu], metadatas[k][4+mu]);
	}
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;

    double results_sum = 0;
    for (int l = 0; l < training_reruns; l++){
      std::vector<double> input_vec(std::begin(moms[l]), std::end(moms[l]));
      std::vector<double> result = kerasModels[l].compute_output(input_vec);
#ifdef DEBUG
      std::cout << "Using y_mean " << metadatas[l][8] << std::endl;
#endif
      double output = nn::destandardise(result[0], metadatas[l][8], metadatas[l][9]);
      results_sum += output;
    }

    double average_output = results_sum/training_reruns;

    std::cout << "Python Loop( 0) = " << python_outputs[i] << std::endl;
    std::cout << "C++    Loop( 0) = " << average_output << std::endl;
  };
}
