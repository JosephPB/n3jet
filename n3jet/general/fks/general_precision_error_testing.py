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
import argparse

from n3jet.utils import FKSPartition
from n3jet.utils.fks_utils import (
    get_near_networks_general,
    get_cut_nerwork,
    infer_on_near_splits,
    infer_on_cut
)
from n3jet.models import Model

parser = argparse.ArgumentParser(description=
                                 'Once models have been trained using []_init_model_testing.py,
                                 this script can be used for testing, given some testing data. 
                                 Note: this assumes that testing data has already been generated.'
)

parser.add_argument(
    '--test_mom_file',
    dest='test_mom_file',
    help='destination of testing momenta file',
    type=str,
)

parser.add_argument(
    '--test_nj_file',
    dest='test_nj_file',
    help='destination of testing NJet file',
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
test_mom_file = args.test_mom_file
test_nj_file = args.test_nj_file
delta_cut = args.delta_cut
delta_near = args.delta_near
model_base_dir = args.model_base_dir
model_dir = args.model_dir
training_reruns = args.training_reruns
all_legs = args.all_legs
all_pairs = args.all_pairs

def file_exists(file_path):
    if os.path.exists(file_path) == True:
        pass
    else:
        raise ValueError('{} does not exist'.format(file_path))

file_exists(test_mom_file)
file_exists(test_nj_file)
file_exists(model_base_dir)
file_exists(model_base_dir + model_dir + '_0')

test_momenta = np.load(test_mom_file, allow_pickle=True)

print ('############### Momenta loaded ###############')

test_nj = np.load(test_nj_file, allow_pickle=True)

print ('############### NJet loaded ###############')

test_momenta = test_momenta.tolist()

print ('Training on {} PS points'.format(len(test_momenta)))

print ('############### Inferring on models ###############')

nlegs = len(test_momenta[0])-2

if all_legs == 'False':
    fks = FKSPartition(
        momenta = test_momenta,
        labels = test_nj,
        all_legs = False
    )

    test_cut_momenta, test_near_momenta, test_near_nj, test_cut_nj = fks.cut_near_split(
        delta_cut = delta_cut,
        delta_near = delta_near
    )
else:
    fks = FKSPartition(
        momenta = test_momenta,
        labels = test_nj,
        all_legs = True
    )

    test_cut_momenta, test_near_momenta, test_near_nj, test_cut_nj = fks.cut_near_split(
        delta_cut = delta_cut,
        delta_near = delta_near
    )

pairs, test_near_nj_split = fks.weighting()
    

if all_legs == 'False':
    NN = Model(
        input_size = (nlegs)*4,
        momenta = test_near_momenta,
        labels = test_near_nj_split[0],
        all_jets=True,
        all_legs=False
    )
    
else:
    NN = Model(
        input_size = (nlegs+2)*4,
        momenta = test_near_momenta,
        labels = test_near_nj_split[0],
        all_jets=False,
        all_legs=True
    )

_,_,_,_,_,_,_,_ = NN.process_training_data()

models = []
x_means = []
y_means = []
x_stds = []
y_stds = []
model_nears = []
model_cuts = []

x_mean_nears = []
x_std_nears = []
y_mean_nears = []
y_std_nears = []

x_mean_cuts = []
x_std_cuts = []
y_mean_cuts = []
y_std_cuts = []

for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    model_dir_new = model_base_dir + model_dir + '_{}/'.format(i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')

    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = get_near_networks_general(
        NN = NN,
        pairs = pairs,
        delta_near = delta_near,
        model_dir = model_dir_new
    )
    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = get_cut_network_general(
        NN = NN,
        delta_cut = delta_cut,
        model_dir = model_dir_new
    )

    model_nears.append(model_near)
    model_cuts.append(model_cut)
    
    x_mean_nears.append(x_mean_near)
    x_std_nears.append(x_std_near)
    y_mean_nears.append(y_mean_near)
    y_std_nears.append(y_std_near)
    
    x_mean_cuts.append(x_mean_cut)
    x_std_cuts.append(x_std_cut)
    y_mean_cuts.append(y_mean_cut)
    y_std_cuts.append(y_std_cut)
    
print ('############### All models loaded ###############')
    
for i in range(training_reruns):
    print ('Predicting on model {}'.format(i))
    model_dir_new = model_base_dir + model_dir + '_{}/'.format(i)
    y_pred_near = infer_on_near_splits(
        NN = NN,
        moms = test_near_momenta,
        models = model_nears[i],
        x_mean_near = x_mean_nears[i],
        x_std_near = x_std_nears[i],
        y_mean_near = y_mean_nears[i],
        y_std_near = y_std_nears[i]
    )
    np.save(model_dir_new + '/pred_near_{}'.format(len(test_momenta)), y_pred_near)
    
for i in range(training_reruns):
    print ('Predicting on model {}'.format(i))
    model_dir_new = model_base_dir + model_dir + '_{}/'.format(i)
    y_pred_cut = infer_on_cut(
        NN = NN,
        moms = test_cut_momenta,
        model = model_cuts[i],
        x_mean_cut = x_mean_cuts[i],
        x_std_cut = x_std_cuts[i],
        y_mean_cut = y_mean_cuts[i],
        y_std_cut = y_std_cuts[i]
    )
    np.save(model_dir_new + '/pred_cut_{}'.format(len(test_momenta)), y_pred_cut)

print ('############### Finished ###############')
