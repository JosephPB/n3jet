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

parser = argparse.ArgumentParser(description='Generate data and train models for LO')

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
    '--test_points',
    dest='test_points',
    help='number of testing phase space points',
    type=int
)

parser.add_argument(
    '--save_dir',
    dest='save_dir',
    help='parent directory in which to save data and models',
    default='/mt/batch/jbullock/njet/RAMBO/',
    type=str
)

args = parser.parse_args()

n_gluon= args.n_gluon
delta = args.delta
points = args.points
test_points = args.test_points
save_dir = args.save_dir

## generate phase-space points

# training points

if os.path.exists(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)) == False:
    momenta = generate(n_gluon+2,points,1000., delta=delta)
    print ('Generated {} phase-space points'.format(len(momenta)))
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,points), momenta)

# testing points

if os.path.exists(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,test_points)) == False:
    test_momenta = generate(n_gluon+2,test_points,1000., delta=delta)
    print ('Generated {} phase-space points'.format(len(test_momenta)))
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,test_points), test_momenta)

## set up

t = 'NJ_4j' # this will be overwritten but takes some params
channel_name = 'eeqq'
channel_inc = [11,-11]
channel_out = [-1,1]
for i in range(n_gluon):
    channel_name += 'G'
    channel_out.append(21)
aspow = n_gluon
aepow = 2


## NJET

# initialise

# mods are is the module to import from e.g. testdata.NJ_2J
# tests are the testdata
# TODO: eliminate 'born', 'loop' - these both come from ML/NJ
mods, tests = action_run(t)

# curoder is the tmp order file
# curtests is passed tests from above
# run_test will also initiate the njet and pass the order file
curorder, curtests = run_tests(mods, tests)

curtests[0]['test']['params']['aspow'] = aspow
curtests[0]['test']['params']['aepow'] = aepow
curtests[0]['test']['params']['ae'] = 1.

curtests[0]['test']['data'] = \
[{'born': 0,
  #'has_lc': False,
  'inc': channel_inc,
  'loop': 0,
  'mcn': 1,
  'name': channel_name,
  'out': channel_out}]
    
test_data, ptype, order = run_batch(curorder, curtests)
print ('NJET initialised')

## training

# load phase-space points

momenta = np.load(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points), allow_pickle=True)
print ('Generated {} phase-space points'.format(len(momenta)))
momenta = momenta.tolist()

# run NJET

if os.path.exists(save_dir + '/data/NJ_LO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    NJ_vals = []
    for i in test_data:
        vals = run_cc_test(momenta, i[1], i[2])
        NJ_vals.append(vals)
    
    # select the first test of the runs
    NJ_vals = NJ_vals[0]
    
    NJ_treevals = []
    for i in NJ_vals:
        NJ_treevals.append(i[0])
        
    np.save(save_dir + '/data/NJ_LO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_treevals))
    
NJ_treevals = np.load(save_dir + '/data/NJ_LO_{}_{}_{}.npy'.format(n_gluon+2, delta, points))

# train NN

NN = Model((len(channel_out)-1)*4,momenta,NJ_treevals)
model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[16,32,16])

if os.path.exists(save_dir + '/models/LO_{}_{}_{}'.format(n_gluon+2,delta,points)) == False:
    os.mkdir(save_dir + '/models/LO_{}_{}_{}'.format(n_gluon+2,delta,points))

model.save(save_dir + '/models/LO_{}_{}_{}/model'.format(n_gluon+2,delta,points))
metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}

pickle_out = open(save_dir + "/models/LO_{}_{}_{}/dataset_metadata.pickle".format(n_gluon+2,delta,points),"wb")
pickle.dump(metadata, pickle_out)
pickle_out.close()
    
## testing

# load phase-space points

test_momenta = np.load(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,test_points), allow_pickle=True)
print ('Generated {} phase-space points'.format(len(test_momenta)))
test_momenta = test_momenta.tolist()

# run NJET

if os.path.exists(save_dir + '/data/NJ_LO_{}_{}_{}.npy'.format(n_gluon+2, delta, test_points)) == False:
    NJ_test_vals = []
    for i in test_data:
        vals = run_cc_test(test_momenta, i[1], i[2])
        NJ_test_vals.append(vals)
        
    # select the first test of the runs
    NJ_test_vals = NJ_test_vals[0]
    
    NJ_test_treevals = []
    for i in NJ_test_vals:
        NJ_test_treevals.append(i[0])
        
    np.save(save_dir + '/data/NJ_LO_{}_{}_{}'.format(n_gluon+2, delta, test_points),np.array(NJ_test_treevals))
    



