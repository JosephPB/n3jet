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
    '--mur_factor',
    dest='mur_factor',
    help='renormalisation factor, default: 0.',
    type=float,
    default=0.
)

parser.add_argument(
    '--test_points',
    dest='test_points',
    help='number of testing phase space points, default: 0',
    type=int,
    default=0
)

parser.add_argument(
    '--generate_momenta',
    dest='generate_momenta',
    help='generate momenta even if the files already exist, default: False',
    type=str,
    default='False'
)

parser.add_argument(
    '--generate_njet',
    dest='generate_njet',
    help='generate njet even if the files already exist, default: False',
    type=str,
    default='False'
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
test_points = args.test_points
mur_factor = args.mur_factor
generate_momenta = args.generate_momenta
generate_njet = args.generate_njet
save_dir = args.save_dir

if os.path.exists(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points)) == False or generate_momenta == 'True':
    print ('############### Generating momenta ###############')
    momenta = generate(n_gluon+2,points,1000., delta=delta)
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,points), momenta)
    
else:
    print ('############### All momenta exist ###############')

if test_points != 0 and os.path.exists(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,test_points)) == False:
    print ('############### Generating test momenta ###############')
    test_momenta = generate(n_gluon+2,test_points,1000., delta=delta)
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,test_points), test_momenta)

elif test_points != 0 and generate_points == 'True':
    print ('############### Generating test momenta ###############')
    test_momenta = generate(n_gluon+2,test_points,1000., delta=delta)
    np.save(save_dir + '/data/PS{}_{}_{}'.format(n_gluon+2,delta,test_points), test_momenta)

else:
    print ('############### All test momenta exist ###############')

if mur_factor == 0.:
    test_data, ptype, order = run_njet(n_gluon)
else:
    test_data, ptype, order = run_njet(n_gluon, mur_factor=mur_factor)

print ('############### Loading momenta ###############')
momenta = np.load(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,points), allow_pickle=True)
momenta = momenta.tolist()


if os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False and os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet loop points and kfactor ###############')
    k_factor, NJ_loop_vals = generate_NLO_njet(momenta, test_data)
    if mur_factor == 0.:
        np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
        np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)
    else:
        np.save(save_dir + '/data/NJ_NLO_{}_{}_{}_{}'.format(n_gluon+2, delta, points, mur_factor),np.array(NJ_loop_vals))
        np.save(save_dir + '/data/NJ_k_{}_{}_{}_{}'.format(n_gluon+2, delta, points, mur_factor),k_factor)

elif generate_njet == 'True':
    ('############### Generating njet loop points and kfactor ###############')
    k_factor, NJ_loop_vals = generate_NLO_njet(momenta, test_data)
    if mur_factor == 0.:
        np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
        np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)
    else:
        np.save(save_dir + '/data/NJ_NLO_{}_{}_{}_{}'.format(n_gluon+2, delta, points, mur_factor),np.array(NJ_loop_vals))
        np.save(save_dir + '/data/NJ_k_{}_{}_{}_{}'.format(n_gluon+2, delta, points, mur_factor),k_factor)
        
elif os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet loop points ###############')
    k_factor, NJ_loop_vals = generate_NLO_njet(momenta, test_data)
    if mur_factor == 0.:
        np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, points),np.array(NJ_loop_vals))
    else:    
        np.save(save_dir + '/data/NJ_NLO_{}_{}_{}_{}'.format(n_gluon+2, delta, points, mur_factor),np.array(NJ_loop_vals))
    
elif os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, points)) == False:
    ('############### Generating njet kfactor ###############')
    k_factor, NJ_loop_vals = generate_NLO_njet(momenta, test_data)
    if mur_factor == 0.:
        np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, points),k_factor)
    else:
        np.save(save_dir + '/data/NJ_k_{}_{}_{}_{}'.format(n_gluon+2, delta, points, mur_factor),k_factor)
    
else:
    print ('############### All njet points exist ###############')


if test_points != 0:
    print ('############### Loading test momenta ###############')
    test_momenta = np.load(save_dir + '/data/PS{}_{}_{}.npy'.format(n_gluon+2,delta,test_points), allow_pickle=True)
    test_momenta = test_momenta.tolist()
    
    if os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, test_points)) == False and os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, test_points)) == False:
        ('############### Generating test njet loop points and kfactor ###############')
        k_factor, NJ_loop_vals = generate_NLO_njet(test_momenta, test_data)
        if mur_factor == 0.:
            np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, test_points),np.array(NJ_loop_vals))
            np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, test_points),k_factor)
        else:
            np.save(save_dir + '/data/NJ_NLO_{}_{}_{}_{}'.format(n_gluon+2, delta, test_points, mur_factor),np.array(NJ_loop_vals))
            np.save(save_dir + '/data/NJ_k_{}_{}_{}_{}'.format(n_gluon+2, delta, test_points, mur_factor),k_factor)
    
    elif generate_njet == 'True':
        ('############### Generating test njet loop points and kfactor ###############')
        k_factor, NJ_loop_vals = generate_NLO_njet(test_momenta, test_data)
        if mur_factor == 0.:    
            np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, test_points),np.array(NJ_loop_vals))
            np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, test_points),k_factor)
        else:
            np.save(save_dir + '/data/NJ_NLO_{}_{}_{}_{}'.format(n_gluon+2, delta, test_points, mur_factor),np.array(NJ_loop_vals))
            np.save(save_dir + '/data/NJ_k_{}_{}_{}_{}'.format(n_gluon+2, delta, test_points, mur_factor),k_factor)
    
    elif os.path.exists(save_dir + '/data/NJ_NLO_{}_{}_{}.npy'.format(n_gluon+2, delta, test_points)) == False:
        ('############### Generating test njet loop points ###############')
        k_factor, NJ_loop_vals = generate_NLO_njet(test_momenta, test_data)
        if mur_factor == 0.:
            np.save(save_dir + '/data/NJ_NLO_{}_{}_{}'.format(n_gluon+2, delta, test_points),np.array(NJ_loop_vals))
        else:
            np.save(save_dir + '/data/NJ_NLO_{}_{}_{}_{}'.format(n_gluon+2, delta, test_points, mur_factor),np.array(NJ_loop_vals))
        
    elif os.path.exists(save_dir + '/data/NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, test_points)) == False:
        ('############### Generating test njet kfactor ###############')
        k_factor, NJ_loop_vals = generate_NLO_njet(test_momenta, test_data)
        if mur_factor == 0.:
            np.save(save_dir + '/data/NJ_k_{}_{}_{}'.format(n_gluon+2, delta, test_points),k_factor)
        else:
            np.save(save_dir + '/data/NJ_k_{}_{}_{}_{}'.format(n_gluon+2, delta, test_points, mur_factor),k_factor)
    else:
        print ('############### All test njet points exist ###############')
    
else:
    print ('############### Skipping test point generation ###############')
    
print ('############### All datasets generated ###############')