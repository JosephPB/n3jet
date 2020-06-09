import numpy as np
np.random.seed(1337)
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
import json
import argparse

from modeldump import ModelDump

def parse():
    """
    Parse arguments
    """
    
    parser = argparse.ArgumentParser(description='This is a simple script to dump Keras model into simple format suitable for porting into pure C++ model')

    parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
    parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
    parser.add_argument('-o', '--output', help="Ouput file name", required=True)
    parser.add_argument('-v', '--verbose', help="Verbose", required=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse()

    print 'Read architecture from', args.architecture
    print 'Read weights from', args.weights
    print 'Writing to', args.output
    
    model_dump = ModelDump(args.architecture, args.weights, args.output, args.verbose, init=True)
