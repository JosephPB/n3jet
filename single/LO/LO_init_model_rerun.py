import sys
sys.path.append('./../')
sys.path.append('./../utils/')
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import pickle
import argparse

from njet_run_functions import *
from model import Model
from rambo_while import *
from utils import *
from uniform_utils import *

parser = argparse.ArgumentParser(description='Testing models trained for precision and optimality error analysis')

parser.add_argument(
    '--n_gluon',
    dest='n_gluon',
    help='number of gluons',
    type=int
)

parser.add_argument(
    '--points',
    dest='points',
    help='number of training phase space points',
    type=int
)

parser.add_argument(
    '--delta',
    dest='delta',
    help='proximity of jets according to JADE algorithm',
    type=float
)

parser.add_argument(
    '--order',
    dest='order',
    help='LO or NLO, default LO',
    type=str,
    default='LO'
)

parser.add_argument(
    '--training_reruns',
    dest='training_reruns',
    help='number of training reruns for testing, default: 1',
    type=int,
    default=1,
)

parser.add_argument(
    '--generate_points',
    dest='generate_points',
    help='whether or not to generate data even if it already exists',
    default='False',
    type=str
)

parser.add_argument(
    '--save_dir',
    dest='save_dir',
    help='parent directory in which to save data and models, default: /mt/batch/jbullock/njet/RAMBO/',
    default='/mt/batch/jbullock/njet/RAMBO/',
    type=str
)

parser.add_argument(
    '--data_dir',
    dest='data_dir',
    help='data directory extension on top of save_dir, default: /data/',
    default='/data/',
    type=str
)

parser.add_argument(
    '--model_dir',
    dest='model_dir',
    help='model directory extension on top of save_dir, default: /scratch/jbullock/njet/RAMBO/models/naive/error_analysis/',
    default='/scratch/jbullock/njet/RAMBO/models/naive/error_analysis/',
    type=str
)

args = parser.parse_args()

n_gluon = args.n_gluon
points = args.points
order = args.order
delta = args.delta
training_reruns = args.training_reruns
generate_points = args.generate_points
save_dir = args.save_dir
data_dir = args.data_dir
model_dir = args.model_dir

lr=0.01

data_dir = save_dir + data_dir
if model_dir != '':
    model_dir = model_dir + '/{}_{}_{}_{}/'.format(order,n_gluon+2,delta,points)
    
mom = 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)
labs = 'NJ_{}_{}_{}_{}.npy'.format(order,n_gluon+2,delta,points)

if os.path.exists(data_dir+mom) == False or generate_points == 'True':
    print ('No directory: {}'.format(data_dir+mom))
    print ('############### Creating phasespace points ###############')
    momenta = generate(n_gluon+2,points,1000.,delta=delta)
    np.save(data_dir + 'PS{}_{}_{}'.format(n_gluon+2,delta,points), momenta)
    
else:
    print ('############### All phasespace files exist ###############')

momenta = np.load(data_dir+mom,allow_pickle=True)
momenta = momenta.tolist()

test_data, _, _ = run_njet(n_gluon)

if os.path.exists(data_dir+labs) == False or generate_points == 'True':
    print ('No directory: {}'.format(data_dir+labs))
    print ('############### Creating njet points ###############')
    NJ_treevals = generate_LO_njet(momenta, test_data)
    np.save(data_dir + 'NJ_{}_{}_{}_{}'.format(order,n_gluon+2,delta,points), NJ_treevals)
    
else:
    print ('############### All njet files exist ###############')

NJ_treevals = np.load(data_dir+labs, allow_pickle=True)

if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)
    print ('Creating directory')
else:
    print ('Directory already exists')

NN = Model((n_gluon+2-1)*4,momenta,NJ_treevals)    
    
for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    model_dir_new = model_dir + '/{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta,points,i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')
    model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[20,40,20], lr=lr, epochs=1000000)
    
    if model_dir != '':
        model.save(model_dir_new + '/model')
        metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
        pickle_out = open(model_dir_new + "/dataset_metadata.pickle","wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
