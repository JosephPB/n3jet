import os
import numpy as np
np.random.seed(1337)
import json
import argparse
import cPickle as pickle
import yaml

from keras.models import Sequential, model_from_json
from keras.optimizers import Adam

from n3jet.utils.general_utils import file_exists
from modeldump import ModelDump

def parse():
    """
    Parse arguments
    """
    
    parser = argparse.ArgumentParser(description=
                                     """
                                     This is a simple script to dump Keras model into 
                                     simple format suitable for porting into pure C++ model
                                     """
    )

    parser.add_argument('-y', '--yaml_file', help="YAML file", type=str, default="False")
    parser.add_argument('-t', '--training_reruns', help="Number of training reruns", type=int, required=False)
    parser.add_argument('-b', '--model_base_dir', help="Model base directory", type=str, required=False)
    parser.add_argument('-m', '--model_dir', help="Model directory", type=str,required=False)
    parser.add_argument('-ob', '--out_base_dir', help="Output base directory in which others will be created", type=str, required=True)
    parser.add_argument('-o', '--out_dir', help="Output directory in which others will be created", type=str,required=False)
    parser.add_argument('-v', '--verbose', help="Verbose", type=bool, required=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse()

    if args.yaml_file != "False":
        file_exists(args.yaml_file)

        with open(args.yaml_file) as f:
            yaml = yaml.load(f, Loader=yaml.FullLoader)
    
        args.model_base_dir = yaml["model_base_dir"]
        args.model_dir = yaml["model_dir"]
        args.training_reruns = yaml["training"]["training_reruns"]
        args.out_dir = yaml["model_dir"]

    for i in range(args.training_reruns):
        print ('Working on training run {}'.format(i))
        mod_dir = args.model_base_dir + args.model_dir + '_{}/'.format(i)
        output = args.out_base_dir + '/' + args.out_dir
        if not os.path.exists(output):
            os.mkdir(output)
        output += '/{}/'.format(i)
        if not os.path.exists(output):
            os.mkdir(output)

        print ('Reading data from {}'.format(mod_dir))
        print ('Saving data to {}'.format(output))
            
        print ('################# Dumping model #################')
        model_dump = ModelDump(
            architecture = mod_dir + '/model_arch.json',
            weights = mod_dir + '/model_weights.h5',
            output = output + '/model.nnet',
            verbose = False,
            init = True
        )
        
        print ('################# Converting metadata #################')
        pickle_out = open(mod_dir+ "/dataset_metadata.pickle","rb")
        metadata = pickle.load(pickle_out)
        pickle_out.close()
        
        with open(output + '/dataset_metadata.dat', 'w') as fin:
            for idx, i in enumerate(metadata['x_mean']):
                if idx == len(metadata['x_mean'])-1:
                    fin.write(str(i) + "\n")
                else:
                    fin.write(str(i) + " ")
            for idx, i in enumerate(metadata['x_std']):
                if idx == len(metadata['x_std'])-1:
                    fin.write(str(i) + "\n")
                else:
                    fin.write(str(i) + " ")
            fin.write(str(metadata['y_mean']) + "\n")
            fin.write(str(metadata['y_std']) + "\n")
        
        
        
