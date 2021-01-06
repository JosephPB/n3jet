import sys
sys.path.append('./../../')
sys.path.append('./../../utils/')
sys.path.append('./../../models/')
sys.path.append('./../../phase/')

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import pickle
import argparse
from tqdm import tqdm

from njet_run_functions import *
from model import Model
from rambo_while import *
from utils import *
from uniform_utils import *
from fks_utils import *
from fks_partition import *

parser = argparse.ArgumentParser(description='Training multiple models on the same dataset for error analysis. Here we assume that the momenta and njet files already exist and will be passed to the script by the user')

parser.add_argument(
    '--mom_file',
    dest='mom_file',
    help='destination of momenta file',
    type=str,
)

parser.add_argument(
    '--nj_file',
    dest='nj_file',
    help='NJet file',
    type=str,
)

parser.add_argument(
    '--delta_cut',
    dest='delta_cut',
    help='proximity of jets according to JADE algorithm',
    type=float,
    default=0.01,
)

parser.add_argument(
    '--delta_near',
    dest='delta_near',
    help='proximity of jets according to JADE algorithm',
    type=float,
)

parser.add_argument(
    '--model_base_dir',
    dest='model_base_dir',
    help='model base directory in which folders will be created',
    type=str,
)

parser.add_argument(
    '--model_dir',
    dest='model_dir',
    help='model directory which will be created on top of model_base_dir',
    type=str,
)

parser.add_argument(
    '--training_reruns',
    dest='training_reruns',
    help='number of training reruns for testing, default: 1',
    type=int,
    default=1,
)

parser.add_argument(
    '--all_legs',
    dest='all_legs',
    help='train on data from all legs, not just all jets, default: False',
    type=str,
    default='False',
)

parser.add_argument(
    '--all_pairs',
    dest='all_pairs',
    help='train on data from all pairs (except for initial state particles), not just all jets, default: False',
    type=str,
    default='False',
)


args = parser.parse_args()
mom_file = args.mom_file
nj_file = args.nj_file
delta_cut = args.delta_cut
delta_near = args.delta_near
model_base_dir = args.model_base_dir
model_dir = args.model_dir
training_reruns = args.training_reruns
all_legs = args.all_legs
all_pairs = args.all_pairs

lr=0.01

def file_exists(file_path):
    if os.path.exists(file_path) == True:
        pass
    else:
        raise ValueError('{} does not exist'.format(file_path))

file_exists(mom_file)
file_exists(nj_file)
file_exists(model_base_dir)

momenta = np.load(mom_file,allow_pickle=True)

print ('############### Momenta loaded ###############')

nj = np.load(nj_file,allow_pickle=True)

print ('############### NJet loaded ###############')

momenta = momenta.tolist()

print ('Training on {} PS points'.format(len(momenta)))

print ('############### Training models ###############')

if os.path.exists(model_base_dir) == False:
    os.mkdir(model_base_dir)
    print ('Creating base directory')
else:
    print ('Base directory already exists')

nlegs = len(momenta[0])-2

if all_legs == 'False':
    cut_momenta, near_momenta, near_nj, cut_nj = cut_near_split(momenta, nj, delta_cut=delta_cut, delta_near=delta_near, all_legs=False)
else:
    cut_momenta, near_momenta, near_nj, cut_nj = cut_near_split(momenta, nj, delta_cut=delta_cut, delta_near=delta_near, all_legs=True)

if all_pairs == 'False':
    pairs, near_nj_split = weighting(near_momenta, nlegs-2, near_nj)
else:
    pairs, near_nj_split = weighting_all(near_momenta, near_nj)

if all_legs == 'False':
    NN = Model((nlegs)*4,near_momenta,near_nj_split[0],all_jets=True,all_legs=False)
else:
    NN = Model((nlegs+2)*4,near_momenta,near_nj_split[0],all_jets=False,all_legs=True)
# input to Model is [no. all legs = nlegs + 1] - [incoming legs = 2] * [len 4- momentum = 4]


for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    model_dir_new = model_base_dir + model_dir + '_{}/'.format(i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')
    if model_dir != '':
        if all_legs == 'False':
            model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks_general((nlegs)*4, pairs, near_momenta, near_nj_split, delta_near, model_dir=model_dir_new, all_jets=True, all_legs=False, lr=lr)
            model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut =  train_cut_network_general((nlegs)*4, cut_momenta, cut_nj, delta_near, model_dir=model_dir_new, all_jets=True, all_legs=False, lr=lr)
        else:
            model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks_general((nlegs+2)*4, pairs, near_momenta, near_nj_split, delta_near, model_dir=model_dir_new, all_jets=False, all_legs=True, lr=lr)
            model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut =  train_cut_network_general((nlegs+2)*4, cut_momenta, cut_nj, delta_near, model_dir=model_dir_new, all_jets=False, all_legs=True, lr=lr)
            
        
print ('############### Finished ###############')
