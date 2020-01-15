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

parser = argparse.ArgumentParser(description='Generate data and train models for NLO')

parser.add_argument(
    '--n_gluon',
    dest='n_gluon',
    help='number of gluons',
    type=int
)

parser.add_argument(
    '--delta',
    dest='delta',
    help='proximity of jets according to JADE algorithm',
    type=float
)

parser.add_argument(
    '--points',
    dest='points',
    help='number of trianing phase space points',
    type=int
)

parser.add_argument(
    '--generate',
    dest='generate_points',
    help='generate points even if the files already exist, default: False',
    type=str,
    default='False'
)

parser.add_argument(
    '--cores',
    dest='cores',
    help='number of cores to parallelise njet calculation, default: 1',
    type=int,
    default=1
)

parser.add_argument(
    '--save_dir',
    dest='save_dir',
    help='parent directory in which to save data and models, default: /mt/batch/jbullock/njet/RAMBO/',
    default='/mt/batch/jbullock/njet/RAMBO/',
    type=str
)

args = parser.parse_args()

n_gluon= args.n_gluon
delta = args.delta
points = args.points
generate_points = args.generate_points
cores = args.cores
save_dir = args.save_dir

## generate phase-space points

# training points

if os.path.exists(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)) == False or generate_points == 'True':
    print ('############### Generating momenta ###############')
    momenta = generate(n_gluon+2,points,1000., delta=delta)
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,points), momenta)
    
else:
    print ('############### All momenta exist ###############')


print ('############### Loading momenta ###############')
momenta = np.load(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points), allow_pickle=True)
momenta = momenta.tolist()

test_data, ptype, order = run_njet(n_gluon)

if os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False and os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet loop points and kfactor ###############')
    NJ_loop_vals, k_factor = multiprocess_generate(momenta, test_data, cores)
    np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)

elif generate_points == 'True':
    ('############### Generating njet loop points and kfactor ###############')
    NJ_loop_vals, k_factor = multiprocess_generate(momenta, test_data, cores)
    np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)

elif os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet loop points ###############')
    NJ_loop_vals, k_factor = multiprocess_generate(momenta, test_data, cores)
    np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    
elif os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet kfactor ###############')
    NJ_loop_vals, k_factor = multiprocess_generate(momenta, test_data, cores)
    np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)
    
else:
    print ('############### All njet points exist ###############')

print ('############### Loading k factor ###############')
    
k_factor = np.load(save_dir + '/data/NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, points))

print ('The length of the momenta is {} and the length of the k-factor is {}'.format(len(momenta), len(k_factor)))
# train NN

print ('############### Training models ###############')

NN = Model((n_gluon+2-1)*4,momenta,k_factor)
model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[16,32,16])
if os.path.exists(save_dir + '/models/NLO_{}_{}_{}'.format(n_gluon+2,delta,points)) == False:
    os.mkdir(save_dir + '/models/NLO_{}_{}_{}'.format(n_gluon+2,delta,points))
#else:
#    raise Exception('Model directory already exists, by continuing you will overwrite a model')

model.save(save_dir + '/models/NLO_{}_{}_{}/model'.format(n_gluon+2,delta,points))

metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}

pickle_out = open(save_dir + "/models/NLO_{}_{}_{}/dataset_metadata.pickle".format(n_gluon+2,delta,points),"wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()