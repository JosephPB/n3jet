import sys
sys.path.append('./../')
sys.path.append('./../utils')
sys.path.append('./../RAMBO')
import os

import numpy as np
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
from model_dataset import Model
from fks_partition import *
from keras.models import load_model
from tqdm import tqdm
from fks_utils import *
from piecewise_utils import *
from utils import *
from uniform_utils import *
import rambo_piecewise_balance as rpb
from rambo_piecewise_balance import *

parser = argparse.ArgumentParser(description='Training multiple models on the same dataset for error analysis')

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
    '--delta_recut',
    dest='delta_recut',
    help='proximity of jets according to JADE algorithm for recutting, default: 0.',
    type=float,
    default=0.
)

parser.add_argument(
    '--delta_renear',
    dest='delta_renear',
    help='proximity of jets according to JADE algorithm for renearing, default: 0.',
    type=float,
    default=0.
)

parser.add_argument(
    '--points',
    dest='points',
    help='number of trianing phase space points',
    type=int
)

parser.add_argument(
    '--subset_points',
    dest='subset_points',
    help='number of training phase space points',
    type=int
)

parser.add_argument(
    '--order',
    dest='order',
    help='LO or NLO, default: NLO',
    type=str,
    default='NLO'
)


parser.add_argument(
    '--training_reruns',
    dest='training_reruns',
    help='number of models trained for error analysis',
    type=int
)

parser.add_argument(
    '--generate',
    dest='generate_points',
    help='True or False as to generate data even if it already exists, default: False',
    default='False',
    type=str
)

parser.add_argument(
    '--save_dir',
    dest='save_dir',
    help='parent directory in which to save data, default: /mt/batch/jbullock/njet/RAMBO/',
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
    help='model directory extension, default: /scratch/jbullock/njet/RAMBO/models/fks_piecewise_balance/error_analysis/',
    default='/scratch/jbullock/njet/RAMBO/models/fks_piecewise_balance/error_analysis/',
    type=str
)

args = parser.parse_args()

lr=0.01

n_gluon= args.n_gluon
delta_cut = args.delta_cut
delta_near = args.delta_near
delta_recut = args.delta_recut
delta_renear = args.delta_renear
points = args.points
subset_points = args.subset_points
order = args.order
generate_points = args.generate_points
save_dir = args.save_dir
data_dir = args.data_dir
model_dir = args.model_dir
training_reruns = args.training_reruns

delta = delta_cut

if delta_renear != 0. and delta_recut ==0.:
    raise Exception('If setting delta_renear must also set delta_recut')
elif delta_renear == 0. and delta_recut !=0.:
    raise Exception('If setting delta_recut must also set delta_renear')

data_dir = save_dir + data_dir
if model_dir != '' and delta_renear == 0. and delta_recut == 0.:
    model_dir = model_dir + '/{}_virt_{}_{}_{}_{}/dataset_size/'.format(order,n_gluon+2,delta_cut,delta_near,subset_points)
elif model_dir != '' and delta_renear != 0. and delta_recut != 0.:
    model_dir = model_dir + '/{}_virt_{}_{}_{}_{}/dataset_size/'.format(order,n_gluon+2,delta_recut,delta_renear,subset_points)
    
if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)
    print ('Created directory')
else:
    print ('Directory already exists')

print ('############### Loading phase space points ###############')
momenta = np.load(data_dir + 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,points), allow_pickle=True)
momenta = momenta.tolist()
    
print ('############### Loading njet points ###############')
NLO = np.load(data_dir + 'NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points),allow_pickle=True)

virt = []
for i in NLO:
    virt.append(float(i[3][1]))
    
virt = np.array(virt)

print ('############### Splitting cut and near points ###############')
if delta_renear == 0. and delta_recut == 0.:
    print ('Using delta_near and delta_cut values')
    cut_momenta, near_momenta, NJ_near_virt, NJ_cut_virt = cut_near_split(momenta, virt, delta_cut=delta_cut, delta_near=delta_near)
else:
    print ('Using delta_renear and delta_recut values')
    cut_momenta, near_momenta, NJ_near_virt, NJ_cut_virt = cut_near_split(momenta, virt, delta_cut=delta_recut, delta_near=delta_renear)
    print ('Training on {} phase space points in total'.format(len(cut_momenta)+len(near_momenta)))
    
_, pairs = D_ij(mom=near_momenta[0],n_gluon=n_gluon)

pairs, NJ_near_virt_split = weighting(near_momenta, n_gluon, NJ_near_virt)

NN = Model((n_gluon+2-1)*4, near_momenta, NJ_near_virt_split[0])
_,_,_,_,_,_,_,_ = NN.process_training_data(moms = near_momenta, labs = NJ_near_virt_split[0])


for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    if delta_renear == 0. and delta_recut == 0.:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_cut,delta_near,subset_points,i)
    else:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_recut,delta_renear,subset_points,i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')
        
    print ('Generating data subset')
        
    near_indices = np.random.randint(0,len(near_momenta),int(subset_points*(float(len(near_momenta))/float(points))))
    cut_indices = np.random.randint(0,len(cut_momenta),int(subset_points*(float(len(cut_momenta))/float(points))))
    
    subset_near_momenta = np.array(near_momenta)[near_indices]
    subset_near_momenta_list = subset_near_momenta.tolist()
    subset_cut_momenta = np.array(cut_momenta)[cut_indices]
    subset_cut_momenta_list = subset_cut_momenta.tolist()
    
    subset_NJ_near_virt_split = []
    for i in NJ_near_virt_split:
        subset_NJ_near_virt_split.append(np.array(i)[near_indices])
    subset_NJ_cut_virt = np.array(NJ_cut_virt)[cut_indices] 
    
    assert len(subset_near_momenta) != 0
    assert len(subset_cut_momenta) != 0
    assert len(subset_NJ_near_virt_split[0]) != 0
    assert len(subset_NJ_cut_virt) != 0
    
    model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks(pairs, subset_near_momenta_list, subset_NJ_near_virt_split, order, n_gluon, delta_near, subset_points, model_dir=model_dir_new, lr=lr)
    
    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut =  train_cut_network(subset_cut_momenta_list, subset_NJ_cut_virt, order, n_gluon, delta_cut, subset_points, model_dir=model_dir_new, lr=lr)
        
print ('############### Finished ###############')