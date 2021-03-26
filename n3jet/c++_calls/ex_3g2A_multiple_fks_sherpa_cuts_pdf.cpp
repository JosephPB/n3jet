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
  double s_com = 500000.0;

  //raw momenta input

  double Momenta[pspoints][legs][4] = {
				     {
				       {80.80323390148537, 0.0, 0.0, 80.8032339014854},
				       {53.57113343015938, 0.0, 0.0, -53.571133430159385},
				       {42.47353254764989, 26.671855251549694, 29.511816381880454, 14.88844512898612},
				       {52.68325739804988, -51.14726465398362, 1.1299491037855054, 12.578002365534454},
				       {39.21757738594496, 24.475409402433904, -30.641765485665946, -0.2343470231945908}
				     },
				     {
				       {88.48800215647898, 0.0, 0.0, 88.48800215647898},
				       {30.862002335592713, 0.0, 0.0, -30.862002335592702},
				       {37.52777986769674, -20.535557255118455, 24.668353258793047, 19.444729299207314},
				       {58.90087277950353, 24.239611461766692, -45.272650711571636, 28.846856811754705},
				       {22.921351844871374, -3.704054206648225, 20.604297452778603, 9.334413709924226}
				     }
  };
  
  std::string model_base = "./models/diphoton/3g2A/RAMBO/parallel_fixed/";
  std::string model_dir = {"events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/"};
  std::string pair_dirs[pairs] = {"/pair_0.02_0/",
				  "/pair_0.02_1/",
				  "/pair_0.02_2/",
				  "/pair_0.02_3/",
				  "/pair_0.02_4/",
				  "/pair_0.02_5/",
				  "/pair_0.02_6/",
				  "/pair_0.02_7/",
				  "/pair_0.02_8/"	  
  };

  std::string cut_dirs = "/cut_0.02/";

  int python_cut_near[2] = {0, 1};
  //double python_check[2] = {0.01885435257459624, 0.04757045055113991};
  double python_outputs[2] = {2.2266408e-07, 1.430258598666967e-06};

  std::vector<std::vector<std::vector<double> > > metadatas(training_reruns, std::vector<std::vector<double> > (pairs+1, std::vector<double>(10)));
  std::string model_dir_models[training_reruns][pairs+1];
  std::vector<std::vector<nn::KerasModel> > kerasModels(training_reruns, std::vector<nn::KerasModel>(pairs+1));

  for (int i = 0; i < training_reruns; i++){

    // Near networks
    for (int j = 0; j < pairs; j++){
      std::string metadata_file = model_base + model_dir + std::to_string(i) + pair_dirs[j] + "dataset_metadata.dat";
      std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
      for (int k = 0; k < 10 ; k++){
	metadatas[i][j][k] = metadata[k];
      };
      model_dir_models[i][j] = model_base + model_dir + std::to_string(i) + pair_dirs[j] + "model.nnet";
#ifdef DEBUG
      std::cout << "Loading from: " << model_dir_models[i][j] << std::endl;
#endif
      kerasModels[i][j].load_weights(model_dir_models[i][j]);
    };

    // Cut networks
    std::string metadata_file = model_base + model_dir + std::to_string(i) + cut_dirs + "dataset_metadata.dat";
    std::vector<double> metadata = nn::read_metadata_from_file(metadata_file);
    for (int k = 0; k < 10 ; k++){
      metadatas[i][pairs][k] = metadata[k];
    };
    model_dir_models[i][pairs] = model_base + model_dir + std::to_string(i) + cut_dirs + "model.nnet";
#ifdef DEBUG
    std::cout << "Loading from: " << model_dir_models[i][pairs] << std::endl;
#endif
    kerasModels[i][pairs].load_weights(model_dir_models[i][pairs]);
  }

  for (int i = 0; i < pspoints; i++){
    std::cout << "==================== Test point " << i+1 << " ====================" << std::endl;


    // standardise momenta
    //double moms[training_reruns][pairs+1][legs*4];
    std::vector<std::vector<std::vector<double>>> moms(training_reruns, std::vector<std::vector<double>>(pairs + 1, std::vector<double>(legs * 4)));

    // flatten momenta
    for (int p = 0; p < legs; p++){
      for (int mu = 0; mu < 4; mu++){
	// standardise input
#ifdef DEBUG
	std::cout << Momenta[i][p][mu] << " ";
#endif
	for (int k = 0; k < training_reruns; k++){
	  for (int j = 0; j < pairs; j++){
	    moms[k][j][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][j][mu], metadatas[k][j][4+mu]);
	  }
	  moms[k][pairs][p*4+mu] = nn::standardise(Momenta[i][p][mu], metadatas[k][pairs][mu], metadatas[k][pairs][4+mu]);
	}
      }
#ifdef DEBUG
      std::cout << std::endl;
#endif
    }
#ifdef DEBUG
    std::cout << std::endl;
#endif
#ifdef DEBUG
    std::cout << "Checking how near we are" << std::endl;
#endif
    //cut/near check
    s_com = Momenta[i][0][0]*Momenta[i][1][0]-(Momenta[i][0][1]*Momenta[i][1][1]+Momenta[i][0][2]*Momenta[i][1][2]+Momenta[i][0][3]*Momenta[i][1][3]);
    int cut_near = 0;
    for (int j = 0; j < legs-1; j++){
      for (int k = j+1; k < legs; k++){
	double prod = Momenta[i][j][0]*Momenta[i][k][0]-(Momenta[i][j][1]*Momenta[i][k][1]+Momenta[i][j][2]*Momenta[i][k][2]+Momenta[i][j][3]*Momenta[i][k][3]);
	double distance = prod/s_com;
#ifdef DEBUG
	std::cout << "Distance is: " << distance << std::endl;
#endif
	if (distance < delta){
	  cut_near += 1; 
	}
      }
    }

#ifdef DEBUG
    //std::cout << "Python min distance is:   " << python_check[i]  << std::endl;
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
	  //std::vector<double> input_vec(std::begin(moms[j][k]), std::end(moms[j][k]));
	  std::vector<double> result = kerasModels[j][k].compute_output(moms[j][k]);
#ifdef DEBUG
	  std::cout << "Before destandardisation = " << result[0] << std::endl;
#endif
	  double output = nn::destandardise(result[0], metadatas[j][k][8], metadatas[j][k][9]);
#ifdef DEBUG
	  std::cout << "After destandardisation = " << output << std::endl;
	  //if (output < 0){
	  //  std::cout << "Output is less than zero" << std::endl;
	  //}
#endif
	  results_pairs += output;
	}
	results_sum += results_pairs;
      }
      else{
	//std::vector<double> input_vec(std::begin(moms[j][pairs]), std::end(moms[j][pairs]));
	std::vector<double> result = kerasModels[j][pairs].compute_output(moms[j][pairs]);
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

    //#ifdef DEBUG
    //if (average_output < 0){
    //  std::cout << "Average output is less than zero" << std::endl;
    //}
    //#endif

    std::cout << "Python Loop( 0) = " << python_outputs[i] << std::endl;
    std::cout << "C++    Loop( 0) = " << average_output << std::endl;
    
  }
}
