import sys
sys.path.append('./../')
sys.path.append('./../../utils/')
sys.path.append('./../../phase/')
sys.path.append('./../../models/')
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import cPickle as pickle
import multiprocessing
import argparse

import rambo_while
from njet_run_functions import *
from model import Model
from fks_partition import *
from keras.models import load_model
from tqdm import tqdm
from fks_utils import *
#from piecewise_utils import *
from utils import *
from uniform_utils import *
import rambo_piecewise_balance as rpb
from rambo_piecewise_balance import *

parser = argparse.ArgumentParser(description='Script for training a network and saving out the data in a .dat file')

parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
args = parser.parse_args()

#mom_file = '/mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/momenta_events_100k.npy'
#nj_file = '/mt/batch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/events_100k_loop.npy'

mom_file = './data/3g2A_test_momenta.npy'
nj_file = './data/3g2A_test_nj.npy'

test_momenta = np.load(mom_file, allow_pickle=True)
test_nj = np.load(nj_file, allow_pickle=True)

test_momenta = test_momenta.tolist()

nlegs = len(test_momenta[0])-2

NN = Model(nlegs*4,test_momenta,test_nj,all_jets=True, all_legs=False)
X_train,X_test,y_train,y_test,_,_,_,_ = NN.process_training_data()

lr=0.01

model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[20,40,20], lr=lr, epochs=1)

#mod_dir = '/scratch/jbullock/Sherpa_NJet/runs/diphoton/3g2A/RAMBO/'

#model = load_model(mod_dir + 'model')

#pickle_out = open(mod_dir + "/dataset_metadata.pickle","rb")
#metadata = pickle.load(pickle_out)
#pickle_out.close()
    
#x_mean = metadata['x_mean']
#y_mean = metadata['y_mean']
#x_std = metadata['x_std']
#y_std = metadata['y_std']

x_standard = NN.process_testing_data(moms=test_momenta,x_mean=x_mean,x_std=x_std,y_mean=y_mean,y_std=y_std)

testing_X = x_standard[:2]

pred = model.predict(testing_X)

with open ('./single_test_arch.json', 'w') as fout:
    fout.write(model.to_json())
model.save_weights(args.weights)

print ('Model JSON and weights save')

print ('Saving out sample data')

with open('./data/single_test_sample.dat', 'w') as fin:
    fin.write("{}\n".format(len(testing_X[0])))
    for idx, i in enumerate(testing_X[0]):
        if idx == len(testing_X[0]) - 1:
            fin.write(str(i) + "\n")
        else:
            fin.write(str(i) + " ")
    


