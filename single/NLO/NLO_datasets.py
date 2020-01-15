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

parser = argparse.ArgumentParser(description='Generate data for NLO')

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
    '--generate_momenta',
    dest='generate_momenta',
    help='generate PS points even if the files already exist, default: False',
    type=str,
    default='False'
)

parser.add_argument(
    '--generate_njet',
    dest='generate_njet',
    help='generate NJet points even if the files already exist, default: False',
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
generate_momenta = args.generate_momenta
generate_njet = args.generate_njet
cores = args.cores
save_dir = args.save_dir

if os.path.exists(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)) == False or generate_momenta == 'True':
    print ('############### Generating momenta ###############')
    momenta = generate(n_gluon+2,points,1000., delta=delta)
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,points), momenta)
    
else:
    print ('############### All momenta exist ###############')

test_data, ptype, order = run_njet(n_gluon)

print ('############### Loading momenta ###############')
momenta = np.load(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points), allow_pickle=True)
momenta = momenta.tolist()


if os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False and os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet loop points and kfactor ###############')
    k_factor, NJ_loop_vals = multiprocess_generate_NLO_njet(momenta, n_gluon, cores)
    np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)

elif generate_njet == 'True':
    ('############### Generating njet loop points and kfactor ###############')
    k_factor, NJ_loop_vals = multiprocess_generate_NLO_njet(momenta, n_gluon, cores)
    np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)

elif os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet loop points ###############')
    k_factor, NJ_loop_vals = multiprocess_generate_NLO_njet(momenta, n_gluon, cores)
    np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    
elif os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet kfactor ###############')
    k_factor, NJ_loop_vals = multiprocess_generate_NLO_njet(momenta, n_gluon, cores)
    np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)
    
else:
    print ('############### All njet points exist ###############')
    
    
print ('############### All datasets generated ###############')