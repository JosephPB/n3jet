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
from tqdm import tqdm

from njet_run_functions import *
from model_dataset import Model
from rambo_while import *
from utils import *
from uniform_utils import *

parser = argparse.ArgumentParser(description='Training multiple models on the same dataset for error analysis')

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
    '--subset_points',
    dest='subset_points',
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
    '--redelta',
    dest='redelta',
    help='proximity of jets according to JADE algorithm, default = 0.',
    type=float,
    default=0.
)

parser.add_argument(
    '--order',
    dest='order',
    help='LO or NLO, default NLO',
    type=str,
    default='NLO'
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
    help='whether or not to generate data even if it already exists, default: False',
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
    help='model directory extension on top of save_dir, default: /scratch/jbullock/njet/RAMBO/models/naive/total_error_analysis/',
    default='/scratch/jbullock/njet/RAMBO/models/naive/error_analysis/',
    type=str
)

args = parser.parse_args()

n_gluon = args.n_gluon
points = args.points
subset_points = args.subset_points
order = args.order
delta = args.delta
redelta = args.redelta
training_reruns = args.training_reruns
generate_points = args.generate_points
save_dir = args.save_dir
data_dir = args.data_dir
model_dir = args.model_dir

if redelta == 0.:
    redelta = delta

lr=0.01

data_dir = save_dir + data_dir

if model_dir != '':
    model_dir = model_dir + '/{}_virt_{}_{}_{}/dataset_size/'.format(order,n_gluon+2,redelta,subset_points)
    
mom = 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)
labs = 'NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)

print ('############### Loading momenta ###############')

momenta = np.load(data_dir+mom,allow_pickle=True)

test_data, _, _ = run_njet(n_gluon)

print ('############### Loading njet points ###############')
NLO = np.load(data_dir + labs,allow_pickle=True)

virt = []
for i in NLO:
    virt.append(float(i[3][1]))

print ('############### Cutting momenta ###############')
print ('Original momenta cut at {}. Recutting to {}'.format(delta, redelta))

w=1000.
p_1 = np.array([w/2,0.,0.,w/2])
p_2 = np.array([w/2,0.,0.,-w/2])

cut_momenta = []
cut_virt = []

for idx, i in tqdm(enumerate(momenta)):
    close = check_all(i, delta=redelta,s_com=dot(p_1,p_2))
    if close == False:
        cut_momenta.append(i)
        cut_virt.append(virt[idx])

virt = np.array(cut_virt)
momenta = cut_momenta

print ('Training on {} PS points'.format(len(momenta)))

print ('############### Training models ###############')

if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)
    print ('Creating directory')
else:
    print ('Directory already exists')


for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    model_dir_new = model_dir + '/{}_{}_{}_{}_{}/'.format(order,n_gluon+2,redelta,subset_points,i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')
        
    indices = np.random.randint(0,points,subset_points)
    subset_momenta = np.array(momenta)[indices]
    subset_momenta_list = subset_momenta.tolist()
    subset_virt = virt[indices]
    
    NN = Model((n_gluon+2-1)*4,subset_momenta_list,subset_virt)
        
    model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[20,40,20], lr=lr, epochs=1000000)
    
    if model_dir != '':
        model.save(model_dir_new + '/model')
        metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
        pickle_out = open(model_dir_new + "/dataset_metadata.pickle","wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()

print ('############### Finished ###############')