import sys
sys.path.append('./../')
sys.path.append('./../utils/')
sys.path.append('./../RAMBO/')
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
from piecewise_utils import *
from utils import *
from uniform_utils import *
from rambo_piecewise_balance import *

parser = argparse.ArgumentParser(description='Generate data for LO')

parser.add_argument(
    '--n_gluon',
    dest='n_gluon',
    help='number of gluons',
    type=int
)

parser.add_argument(
    '--delta_cut',
    dest='delta_cut',
    help='proximity of jets according to JADE algorithm',
    type=float
)

parser.add_argument(
    '--delta_near',
    dest='delta_near',
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
    help='child directory from which to load data, default: /data/piecewise_balance/',
    default='/data/piecewise_balance/',
    type=str
)

args = parser.parse_args()

n_gluon= args.n_gluon
delta_cut = args.delta_cut
delta_near = args.delta_near
points = args.points
mur_factor = args.mur_factor
order = args.order
indices = args.indices
save_dir = args.save_dir
data_dir = args.data_dir

data_dir = save_dir + data_dir

if order == 'NLO':
    raise Exception('order = NLO not supported for cutting - please use ../RAMBO/dataset_test.py for this')

cut_mom = 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,points)
near_mom = 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,points)

print ('############### Loading momenta ###############')

cut_momenta = np.load(data_dir+cut_mom,allow_pickle=True)
cut_momenta = cut_momenta.tolist()
near_momenta = np.load(data_dir+near_mom,allow_pickle=True)
near_momenta = near_momenta.tolist()

cut_labs = 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, points)
near_labs = 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, points)    
    
print ('############### Loading njet points ###############')
cut_NJet = np.load(data_dir + cut_labs, allow_pickle=True)
near_NJet = np.load(data_dir + near_labs, allow_pickle=True)

print ('############### Performing random testing ###############')

indices = np.random.randint(points//2, size=indices)
cut_mom_test = np.array(cut_momenta)[indices].tolist()
near_mom_test = np.array(near_momenta)[indices].tolist()

if order == 'LO':
    test_data, _, _ = run_njet(n_gluon)
    cut_NJet_test, near_NJet_test = generate_LO_piecewise_njet(cut_mom_test, near_mom_test, test_data)
    
cut_NJet_test = np.array(cut_NJet_test)
near_NJet_test = np.array(near_NJet_test)

if np.all(cut_NJet[indices] == cut_NJet_test):
    print ('############### Success, cut tests agree! ###############')
else:
    print ('############### Cut tests failed ###############')
    print (NJet[indices])
    print(NJet_test)
    
if np.all(near_NJet[indices] == near_NJet_test):
    print ('############### Success, near tests agree! ###############')
else:
    print ('############### Near tests failed ###############')
    print (NJet[indices])
    print(NJet_test)