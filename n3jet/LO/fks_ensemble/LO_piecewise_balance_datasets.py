import sys
sys.path.append('./../')
sys.path.append('./../utils')
sys.path.append('./../FKS')
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import cPickle as pickle
import multiprocessing

import rambo_while
from njet_run_functions import *
from model import Model
from fks_partition import *
from keras.models import load_model
from tqdm import tqdm
from fks_utils import *
from piecewise_utils import *
from utils import *
from uniform_utils import *
from rambo_piecewise_balance import *
#import rambo_piecewise_balance as rpb

parser = argparse.ArgumentParser(description='Generate LO piecewise balanced data')

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
    '--test_points',
    dest='test_points',
    help='number of testing phase space points, default: 0',
    type=int,
    default = 0
)

parser.add_argument(
    '--generate',
    dest='generate_points',
    help='override point generation, default: False',
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
    help='directory in which to save data, default: /data/piecewise_balance/',
    default='/data/piecewise_balance/',
    type=str
)


args = parser.parse_args()

n_gluon= args.n_gluon
delta_cut = args.delta_cut
delta_near = args.delta_near
points = args.points
test_points = args.test_points
save_dir = args.save_dir
data_dir = args.data_dir
generate_points = args.generate_points

order = 'LO'

delta = delta_cut

data_dir = save_dir + data_dir

if os.path.exists(data_dir + 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,points)) == False and os.path.exists(data_dir + 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,points)) == False: 
    print ('############### Creating both cut and near phasespace points ###############')
    cut_momenta, near_momenta = generate(n_gluon+2,points,1000., delta_cut, delta_near)
    np.save(data_dir + 'PS_cut_{}_{}_{}'.format(n_gluon+2,delta_cut,points), cut_momenta)
    np.save(data_dir + 'PS_near_{}_{}_{}'.format(n_gluon+2,delta_near,points), near_momenta)

elif generate_points == 'True': 
    print ('generate is True')
    print ('############### Creating both cut and near phasespace points ###############')
    cut_momenta, near_momenta = generate(n_gluon+2,points,1000., delta_cut, delta_near)
    np.save(data_dir + 'PS_cut_{}_{}_{}'.format(n_gluon+2,delta_cut,points), cut_momenta)
    np.save(data_dir + 'PS_near_{}_{}_{}'.format(n_gluon+2,delta_near,points), near_momenta)

elif os.path.exists(data_dir + 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,points)) == False:
    print ('############### Creating cut phasespace points ###############')
    cut_momenta, near_momenta = generate(n_gluon+2,points,1000., delta_cut, delta_near)
    np.save(data_dir + 'PS_cut_{}_{}_{}'.format(n_gluon+2,delta_cut,points), cut_momenta)
    
elif os.path.exists(data_dir + 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,points)) == False:
    print ('############### Creating near phasespace points ###############')
    cut_momenta, near_momenta = generate(n_gluon+2,points,1000., delta_cut, delta_near)
    np.save(data_dir + 'PS_near_{}_{}_{}'.format(n_gluon+2,delta_near,points), near_momenta)
else:
    print ('############### All phasespace files exist ###############')
    
print ('############### Loading momenta ###############')
    
cut_momenta = np.load(data_dir + 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,points),allow_pickle=True)

near_momenta = np.load(data_dir + 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,points),allow_pickle=True)

cut_momenta = cut_momenta.tolist()
near_momenta = near_momenta.tolist()

test_data, _, _ = run_njet(n_gluon)

if os.path.exists(data_dir + 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, points)) == False\
    and os.path.exists(data_dir + 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, points)) == False:
    print ('############### Calculating both cut and near njet points ###############')
    NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(cut_momenta, near_momenta, test_data)
    np.save(data_dir + 'NJ_LO_cut_{}_{}_{}'.format(n_gluon+2, delta_cut, points),np.array(NJ_cut_treevals))
    np.save(data_dir + 'NJ_LO_near_{}_{}_{}'.format(n_gluon+2, delta_near, points),np.array(NJ_near_treevals))
    
elif generate_points == 'True':
    print ('############### Calculating both cut and near njet points ###############')
    NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(cut_momenta, near_momenta, test_data)
    np.save(data_dir + 'NJ_LO_cut_{}_{}_{}'.format(n_gluon+2, delta_cut, points),np.array(NJ_cut_treevals))
    np.save(data_dir + 'NJ_LO_near_{}_{}_{}'.format(n_gluon+2, delta_near, points),np.array(NJ_near_treevals))

elif os.path.exists(data_dir + 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, points)) == False:
    print ('############### Calculating cut njet points ###############')
    NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(cut_momenta, near_momenta, test_data)
    np.save(data_dir + 'NJ_LO_cut_{}_{}_{}'.format(n_gluon+2, delta_cut, points),np.array(NJ_cut_treevals))
    
elif os.path.exists(data_dir + 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, points)) == False:
    print ('############### Calculating near njet points ###############')
    NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(cut_momenta, near_momenta, test_data)
    np.save(data_dir + 'NJ_LO_near_{}_{}_{}'.format(n_gluon+2, delta_near, points),np.array(NJ_near_treevals))
else:
    print ('############### All njet files exist ###############')
    
print ('############### Loading NJet points ###############')
    
NJ_cut_treevals = np.load(data_dir + 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, points), allow_pickle=True)
NJ_near_treevals = np.load(data_dir + 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, points), allow_pickle=True)

_, pairs = D_ij(mom=near_momenta[0],n_gluon=n_gluon)

if test_points != 0:
    
    if os.path.exists(data_dir + 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,test_points)) == False and os.path.exists(data_dir + 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,test_points)) == False: 
        print ('############### Creating both cut and near phasespace test_points ###############')
        cut_momenta, near_momenta = generate(n_gluon+2,test_points,1000., delta_cut, delta_near)
        np.save(data_dir + 'PS_cut_{}_{}_{}'.format(n_gluon+2,delta_cut,test_points), cut_momenta)
        np.save(data_dir + 'PS_near_{}_{}_{}'.format(n_gluon+2,delta_near,test_points), near_momenta)
    
    elif generate_points == 'True': 
        print ('generate is True')
        print ('############### Creating both cut and near phasespace test_points ###############')
        cut_momenta, near_momenta = generate(n_gluon+2,test_points,1000., delta_cut, delta_near)
        np.save(data_dir + 'PS_cut_{}_{}_{}'.format(n_gluon+2,delta_cut,test_points), cut_momenta)
        np.save(data_dir + 'PS_near_{}_{}_{}'.format(n_gluon+2,delta_near,test_points), near_momenta)
    
    elif os.path.exists(data_dir + 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,test_points)) == False:
        print ('############### Creating cut phasespace test_points ###############')
        cut_momenta, near_momenta = generate(n_gluon+2,test_points,1000., delta_cut, delta_near)
        np.save(data_dir + 'PS_cut_{}_{}_{}'.format(n_gluon+2,delta_cut,test_points), cut_momenta)
        
    elif os.path.exists(data_dir + 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,test_points)) == False:
        print ('############### Creating near phasespace test_points ###############')
        cut_momenta, near_momenta = generate(n_gluon+2,test_points,1000., delta_cut, delta_near)
        np.save(data_dir + 'PS_near_{}_{}_{}'.format(n_gluon+2,delta_near,test_points), near_momenta)
    else:
        print ('############### All phasespace files exist ###############')
    
    print ('############### Loading test momenta ###############')
        
    test_near_momenta = np.load(data_dir + 'PS_near_{}_{}_{}.npy'.format(n_gluon+2,delta_near,test_points), allow_pickle=True)
    
    test_cut_momenta = np.load(data_dir + 'PS_cut_{}_{}_{}.npy'.format(n_gluon+2,delta_cut,test_points), allow_pickle=True)
    
    test_near_momenta = test_near_momenta.tolist()
    test_cut_momenta = test_cut_momenta.tolist()
    
    if os.path.exists(data_dir + 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, test_points)) == False\
        and os.path.exists(data_dir + 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, test_points)) == False:
        print ('############### Calculating both cut and near njet test_points ###############')
        NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(test_cut_momenta, test_near_momenta, test_data)
        np.save(data_dir + 'NJ_LO_cut_{}_{}_{}'.format(n_gluon+2, delta_cut, test_points),np.array(NJ_cut_treevals))
        np.save(data_dir + 'NJ_LO_near_{}_{}_{}'.format(n_gluon+2, delta_near, test_points),np.array(NJ_near_treevals))
        
    elif generate_points == 'True':
        print ('############### Calculating both cut and near njet test_points ###############')
        NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(test_cut_momenta, test_near_momenta, test_data)
        np.save(data_dir + 'NJ_LO_cut_{}_{}_{}'.format(n_gluon+2, delta_cut, test_points),np.array(NJ_cut_treevals))
        np.save(data_dir + 'NJ_LO_near_{}_{}_{}'.format(n_gluon+2, delta_near, test_points),np.array(NJ_near_treevals))
    
    elif os.path.exists(data_dir + 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, test_points)) == False:
        print ('############### Calculating cut njet test_points ###############')
        NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(test_cut_momenta, test_near_momenta, test_data)
        np.save(data_dir + 'NJ_LO_cut_{}_{}_{}'.format(n_gluon+2, delta_cut, test_points),np.array(NJ_cut_treevals))
        
    elif os.path.exists(data_dir + 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, test_points)) == False:
        print ('############### Calculating near njet test_points ###############')
        NJ_cut_treevals, NJ_near_treevals = generate_LO_piecewise_njet(test_cut_momenta, test_near_momenta, test_data)
        np.save(data_dir + 'NJ_LO_near_{}_{}_{}'.format(n_gluon+2, delta_near, test_points),np.array(NJ_near_treevals))
    else:
        print ('############### All njet files exist ###############')
        
    print ('############### Loading NJet test points ###############')
        
    NJ_cut_test_treevals = np.load(data_dir + 'NJ_LO_cut_{}_{}_{}.npy'.format(n_gluon+2, delta_cut, test_points), allow_pickle=True)
    NJ_near_test_treevals = np.load(data_dir + 'NJ_LO_near_{}_{}_{}.npy'.format(n_gluon+2, delta_near, test_points), allow_pickle=True)
    
print ('############### All points genenrated ###############')