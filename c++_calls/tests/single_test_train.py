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
from utils import *
from uniform_utils import *
import rambo_piecewise_balance as rpb
from rambo_piecewise_balance import *

parser = argparse.ArgumentParser(description='Script for training a network and saving out the data in a .dat file')

parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
args = parser.parse_args()

mom_file = './data/3g2A_test_momenta.npy'
nj_file = './data/3g2A_test_nj.npy'

test_momenta = np.load(mom_file, allow_pickle=True)
test_nj = np.load(nj_file, allow_pickle=True)

test_momenta = test_momenta.tolist()

nlegs = len(test_momenta[0])-2

NN = Model((nlegs+2)*4,test_momenta,test_nj,all_jets=False, all_legs=True)
X_train,X_test,y_train,y_test,_,_,_,_ = NN.process_training_data()

lr=0.01

model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[20,40,20], lr=lr, epochs=1)

metadata = {}
metadata['x_mean'] = x_mean
metadata['y_mean'] = y_mean
metadata['x_std'] = x_std
metadata['y_std'] = y_std

pickle_out = open("./data/single_test_dataset_metadata.pickle","wb")
metadata = pickle.dump(metadata, pickle_out)
pickle_out.close()


x_standard = NN.process_testing_data(moms=test_momenta,x_mean=x_mean,x_std=x_std,y_mean=y_mean,y_std=y_std)

test_momenta = [[500., 0., 0., 500.],[500., 0., 0., -500.],[253.58419798, -239.58965912, 66.81985738, -49.36443422],
                [373.92489886, 7.43568582, -321.18384469, 191.32558238], [372.49090317, 232.1539733, 254.36398731, -141.96114816]]

testing_X = NN.process_testing_data(test_momenta)

pred_std = model.predict(testing_X)
pred_destd = NN.destandardise_data(pred_std)

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


print ('Saving out metadata')
with open('./data/single_test_dataset_metadata.dat', 'w') as fin:
    for idx, i in enumerate(x_mean):
        if idx == len(x_mean)-1:
            fin.write(str(i) + "\n")
        else:
            fin.write(str(i) + " ")
    for idx, i in enumerate(x_std):
        if idx == len(x_std)-1:
            fin.write(str(i) + "\n")
        else:
            fin.write(str(i) + " ")
    fin.write(str(y_mean) + "\n")
    fin.write(str(y_std) + "\n")

print ('Saving out std and destd inference')
with open('./data/single_test_infer.dat', 'w') as fin:
    fin.write("{}\n".format(pred_std[0][0]))
    fin.write("{}".format(pred_destd[0][0]))

print ('Model std output for C++ testing is: {}'.format(pred_std[0][0]))
print ('Model destd output for C++ testing is: {}'.format(pred_destd[0][0]))


