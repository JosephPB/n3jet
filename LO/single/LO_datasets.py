import sys
sys.path.append('./../../utils/')
sys.path.append('./../../phase/')
sys.path.append('./../../models/')
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
    '--generate_momenta',
    dest='generate_momenta',
    help='whether or not to generate PS points even if it already exists',
    default='False',
    type=str
)

parser.add_argument(
    '--generate_njet',
    dest='generate_njet',
    help='whether or not to generate njet points even if it already exists',
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

args = parser.parse_args()

n_gluon = args.n_gluon
points = args.points
order = args.order
delta = args.delta
generate_momenta = args.generate_momenta
generate_njet = args.generate_njet
save_dir = args.save_dir
data_dir = args.data_dir

lr=0.01

data_dir = save_dir + data_dir

    
mom = 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)
labs = 'NJ_{}_{}_{}_{}.npy'.format(order,n_gluon+2,delta,points)

if os.path.exists(data_dir+mom) == False or generate_momenta == 'True':
    print ('No directory: {}'.format(data_dir+mom))
    print ('############### Creating phasespace points ###############')
    momenta = generate(n_gluon+2,points,1000.,delta=delta)
    np.save(data_dir + 'PS{}_{}_{}'.format(n_gluon+2,delta,points), momenta)
    
else:
    print ('############### All phasespace files exist ###############')

momenta = np.load(data_dir+mom,allow_pickle=True)
momenta = momenta.tolist()

test_data, _, _ = run_njet(n_gluon)

if os.path.exists(data_dir+labs) == False or generate_njet == 'True':
    print ('No directory: {}'.format(data_dir+labs))
    print ('############### Creating njet points ###############')
    NJ_treevals = generate_LO_njet(momenta, test_data)
    np.save(data_dir + 'NJ_{}_{}_{}_{}'.format(order,n_gluon+2,delta,points), NJ_treevals)
    
else:
    print ('############### All njet files exist ###############')

NJ_treevals = np.load(data_dir+labs, allow_pickle=True)

print ('############### Finished ###############')
