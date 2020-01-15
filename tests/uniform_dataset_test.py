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

parser = argparse.ArgumentParser(description='Generate data for LO')

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
    '--mur_factor',
    dest='mur_factor',
    help='renormalisation factor, default: 0.',
    type=float,
    default=0.
)

parser.add_argument(
    '--order',
    dest='order',
    help='LO or NLO',
    type=str
)

parser.add_argument(
    '--indices',
    dest='indices',
    help='number of points to check, default: 10',
    type=int,
    default=10
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
    help='child directory from which to load data, default: /data/',
    default='/data/',
    type=str
)

args = parser.parse_args()

n_gluon= args.n_gluon
delta = args.delta
points = args.points
mur_factor = args.mur_factor
order = args.order
indices = args.indices
save_dir = args.save_dir
data_dir = args.data_dir

data_dir = save_dir + data_dir

mom = 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)

print ('############### Loading momenta ###############')

momenta = np.load(data_dir+mom,allow_pickle=True)
momenta = momenta.tolist()

if order == 'LO':
    labs = 'NJ_LO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)
elif order == 'NLO' and mur_factor !=0.:
    labs = 'NJ_k_{}_{}_{}_{}.npy'.format(n_gluon+2, delta, points, mur_factor)
else:
    labs = 'NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, points)
    
print ('############### Loading njet points ###############')
NJet = np.load(data_dir + labs, allow_pickle=True)

print ('############### Performing random testing ###############')

indices = np.random.randint(points, size=indices)
mom_test = np.array(momenta)[indices].tolist()

if order == 'LO':
    test_data, _, _ = run_njet(n_gluon)
    NJet_test = generate_LO_njet(mom_test, test_data)
elif order == 'NLO' and mur_factor != 0.:
    test_data, ptype, order = run_njet(n_gluon, mur_factor=mur_factor)
    NJet_test,_ = generate_NLO_njet(mom_test, test_data)
else:
    test_data, _, _ = run_njet(n_gluon)
    NJet_test,_ = generate_NLO_njet(mom_test, test_data)
    
NJet_test = np.array(NJet_test)

if np.all(NJet[indices] == NJet_test):
    print ('############### Success, tests agree! ###############')
else:
    print ('############### Tests failed ###############')
    print (NJet[indices])
    print(NJet_test)
