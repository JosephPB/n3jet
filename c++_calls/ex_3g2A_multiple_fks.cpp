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
  const double delta = 0.02;
  const double s_com = 500000.0;

  //raw momenta input

  double Momenta[pspoints][legs][4] = {
				     {
				       {500.,   0,    0.,  500.},
				       {500.,   0.,   0., -500.},
				       { 478.90769254,  179.71954662, -115.72193182,  428.55792728},
				       {  89.06015462,  -76.15029226,   43.54647889,  -15.38012701},
				       { 432.03215284, -103.56925437,   72.17545292, -413.17780026}
				     },
				     {
				       {500.,   0.,   0., 500.},
				       {500.,   0.,   0., -500.},
				       { 126.69820582,  120.01853098,  -31.03838326,   26.16498318},
				       { 397.08701945,  157.11405855, -345.48658337, -116.75741663},
				       { 476.21477472, -277.13258953,  376.52496662,   90.59243344}
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

  int python_cut_near[2] = {1, 0};
  double python_check[2] = {0.01885435257459624, 0.04757045055113991};
  double python_outputs[2] = {8.79497694672e-07, 1.62700327877e-08};

  std::vector<std::vector<std::vector<double> > > metadatas(training_reruns, std::vector<std::vector<double> > (pairs+1, std::vector<double>(10)));
  std::string model_dir_models[training_reruns][pairs+1];
  std::vector<std::vector<nn::KerasModel> > kerasModels(training_reruns, std::vector<nn::KerasModel>(pairs+1));

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
	    moms[k][j][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][j][mu], metadatas[k][j][4+mu]);
	  }
	  moms[k][pairs][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][pairs][mu], metadatas[k][pairs][4+mu]);
	}
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
#ifdef DEBUG
    std::cout << "Checking how near we are" << std::endl;
#endif
    //cut/near check
    int cut_near = 0;
    for (int j = 0; j < legs-1; j++){
      for (int k = j+1; k < legs; k++){
	double prod = Momenta[i][j][0]*Momenta[i][k][0]-(Momenta[i][j][1]*Momenta[i][k][1]+Momenta[i][j][2]*Momenta[i][k][2]+Momenta[i][j][3]*Momenta[i][k][3]);
	double distance = prod/s_com;
	//double distance = nn::pair_check(Momenta[i][j], Momenta[i][k], delta, s_com);
#ifdef DEBUG
	std::cout << "Distance is: " << distance << std::endl;
#endif
	if (distance < delta){
	  cut_near += 1; 
	}
      }
    }

#ifdef DEBUG
    std::cout << "Python min distance is:   " << python_check[i]  << std::endl;
    std::cout << "Python cut/near check is: " << python_cut_near[i] << std::endl;
    std::cout << "C++    cut/near check is: " << cut_near << std::endl;
    std::cout << "Note: here checking if cut/near >0 or not, not the actual value" <<std::endl;
#endif

    // inference

    double results_sum = 0;
    for (int j = 0; j < training_reruns; j++){
      if (cut_near >= 1){
	// infer over all pairs
	double results_pairs = 0;
	for (int k = 0; k < pairs; k++){
	  std::vector<double> input_vec(std::begin(moms[j][k]), std::end(moms[j][k]));
	  std::vector<double> result = kerasModels[j][k].compute_output(input_vec);
#ifdef DEBUG
	  std::cout << "Before destandardisation = " << result[0] << std::endl;
#endif
	  double output = nn::destandardise(result[0], metadatas[j][k][8], metadatas[j][k][9]);
#ifdef DEBUG
	  std::cout << "After destandardisation = " << output << std::endl;
	  if (output < 0){
	    std::cout << "Output is less than zero" << std::endl;
	  }
#endif
	  results_pairs += output;
	}
	results_sum += results_pairs;
      }
      else{
	std::vector<double> input_vec(std::begin(moms[j][pairs]), std::end(moms[j][pairs]));
	std::vector<double> result = kerasModels[j][pairs].compute_output(input_vec);
#ifdef DEBUG
	  std::cout << "Before destandardisation = " << result[0] << std::endl;
#endif

	double output = nn::destandardise(result[0], metadatas[j][pairs][8], metadatas[j][pairs][9]);
#ifdef DEBUG
	  std::cout << "After destandardisation = " << output << std::endl;
#endif

	results_sum += output;
      }
    }

    double average_output = results_sum/training_reruns;

#ifdef DEBUG
    if (average_output < 0){
      std::cout << "Average output is less than zero" << std::endl;
    }
#endif

    std::cout << "Python Loop( 0) = " << python_outputs[i] << std::endl;
    std::cout << "C++    Loop( 0) = " << average_output << std::endl;
    
  }
}
