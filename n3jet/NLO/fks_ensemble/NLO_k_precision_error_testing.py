import sys
sys.path.append('./../')
sys.path.append('./../utils')
sys.path.append('./../RAMBO')
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
import multiprocessing
import argparse

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
import rambo_piecewise_balance as rpb
from rambo_piecewise_balance import *


parser = argparse.ArgumentParser(description='Testing models trained for precision and optimality error analysis')

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
    help='number of training phase space points',
    type=int
)

parser.add_argument(
    '--test_points',
    dest='test_points',
    help='number of test phase space points',
    type=int
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
    '--cores',
    dest='cores',
    help='number of cores available for PS weighting and model training, default: 1',
    type=int,
    default=1,
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
    help='model directory extension on top of save_dir, default: /scratch/jbullock/njet/RAMBO/models/fks_piecewise_balance/error_analysis/',
    default='/scratch/jbullock/njet/RAMBO/models/fks_piecewise_balance/error_analysis/',
    type=str
)

args = parser.parse_args()

n_gluon= args.n_gluon
delta_cut = args.delta_cut
delta_near = args.delta_near
delta_recut = args.delta_recut
delta_renear = args.delta_renear
points = args.points
test_points = args.test_points
order = args.order
cores = args.cores
generate_points = args.generate_points
save_dir = args.save_dir
data_dir = args.data_dir
model_dir = args.model_dir
training_reruns = args.training_reruns

generate_points = 'False'
training = False
lr=0.01
delta = delta_cut

if delta_renear != 0. and delta_recut ==0.:
    raise Exception('If setting delta_renear must also set delta_recut')
elif delta_renear == 0. and delta_recut !=0.:
    raise Exception('If setting delta_recut must also set delta_renear')

data_dir = save_dir + data_dir
if model_dir != '' and delta_renear == 0. and delta_recut == 0.:
    model_dir = model_dir + '/{}_k_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_cut,delta_near,points)
elif model_dir != '' and delta_renear != 0. and delta_recut != 0.:
    model_dir = model_dir + '/{}_k_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_recut,delta_renear,points)

if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)
    print ('Created directory')
else:
    print ('Model directory already exists')
    
print ('############### Loading phase space points ###############')
momenta = np.load(data_dir + 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,test_points), allow_pickle=True)

momenta = momenta.tolist()

print ('############### Loading njet points ###############')
virt = np.load(data_dir + 'NJ_k_{}_{}_{}.npy'.format(n_gluon+2, delta, test_points), allow_pickle=True)
   
print ('############### Splitting cut and near points ###############')
if delta_renear == 0. and delta_recut == 0.:
    print ('Using delta_near and delta_cut values')
    cut_momenta, near_momenta, NJ_near_virt, NJ_cut_virt = cut_near_split(momenta, virt, delta_cut=delta_cut, delta_near=delta_near)
else:
    print ('Using delta_renear and delta_recut values')
    cut_momenta, near_momenta, NJ_near_virt, NJ_cut_virt = cut_near_split(momenta, virt, delta_cut=delta_recut, delta_near=delta_renear)
    print ('Testing on {} phase space points in total'.format(len(cut_momenta)+len(near_momenta)))

test_points = len(cut_momenta)+len(near_momenta)
_, pairs = D_ij(mom=near_momenta[0],n_gluon=n_gluon)

if cores == 0:
    cores = len(pairs)

pairs, NJ_near_virt_split = multiprocess_weighting(near_momenta, n_gluon, NJ_near_virt, cores)    
    
NN = Model((n_gluon+2-1)*4, near_momenta, NJ_near_virt_split[0])
_,_,_,_,_,_,_,_ = NN.process_training_data(moms = near_momenta, labs = NJ_near_virt_split[0])
    
    
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
    if delta_renear == 0. and delta_recut == 0.:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_cut,delta_near,points,i)
    else:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_recut,delta_renear,points,i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')
    if training == True:
        model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks(pairs, near_momenta, NJ_near_virt_split, order, n_gluon, delta_near, points, model_dir=model_dir_new, lr=lr)
        model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut =  train_cut_network(cut_momenta, NJ_cut_virt, order, n_gluon, delta_cut, points, model_dir=model_dir_new, lr=lr)
    else:
        model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = get_near_networks(NN, pairs, order, n_gluon, delta_near, points, model_dir_new)
        model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = get_cut_network(NN, order, n_gluon, delta_cut, points, model_dir_new)
    
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
    
y_pred_nears = []
for i in range(training_reruns):
    print ('Predicting on model {}'.format(i))
    if delta_renear == 0. and delta_recut == 0.:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_cut,delta_near,points,i)
    else:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_recut,delta_renear,points,i)
    y_pred_near = infer_on_near_splits(NN, near_momenta, model_nears[i], x_mean_nears[i], x_std_nears[i], y_mean_nears[i], y_std_nears[i])
    y_pred_nears.append(y_pred_near)
    np.save(model_dir_new + '/pred_near_{}'.format(test_points),y_pred_near)
    
y_pred_cuts = []
for i in range(training_reruns):
    print ('Predicting on model {}'.format(i))
    if delta_renear == 0. and delta_recut == 0.:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_cut,delta_near,points,i)
    else:
        model_dir_new = model_dir + '/{}_{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta_recut,delta_renear,points,i)
    y_pred_cut = infer_on_cut(NN, cut_momenta, model_cuts[i], x_mean_cuts[i], x_std_cuts[i], y_mean_cuts[i], y_std_cuts[i])
    y_pred_cuts.append(y_pred_cut)
    np.save(model_dir_new + '/pred_cut_{}'.format(test_points),y_pred_cut)

print ('############### Concatenating data ###############')
    
test_momenta_array = np.concatenate((cut_momenta,near_momenta))
NJ_test = np.concatenate((NJ_cut_virt,NJ_near_virt))
y_preds = []
for i in range(training_reruns):
    y_pred = np.concatenate((y_pred_cuts[i],y_pred_nears[i]))
    y_preds.append(y_pred)
    
    
indices = np.arange(test_points)
np.random.shuffle(indices)

y_preds_shuffle = []
for i in range(training_reruns):
    y_preds_shuffle.append(np.array(y_preds[i])[indices])
    
test_momenta_array_shuffle = np.array(test_momenta_array)[indices]
NJ_test_shuffle = np.array(NJ_test)[indices]


test_momenta_array = np.array(test_momenta_array)
cs_range = np.arange(10000,(len(test_momenta_array)/10000)*10001,10000)

NJ_cs = []
NJ_MC_std = []

for i in cs_range:

    NJ_to_sum = NJ_test_shuffle[:i]
    NJ_cs.append(np.sum(NJ_to_sum))
    
    f = np.sum(NJ_test_shuffle[:i])/i
    f_2 = np.sum(NJ_test_shuffle[:i]**2)/i
    std = np.sqrt((f_2-f**2)/i)
    NJ_MC_std.append(std)

NN_css = []
NN_MC_stds = []
for j in range(training_reruns):
    NN_cs = []
    NN_MC_std = []
    for i in cs_range:
        NN_to_sum = y_preds_shuffle[j][:i]
        NN_cs.append(np.sum(NN_to_sum))
        
        f = np.sum(y_preds_shuffle[j][:i])/i
        f_2 = np.sum(y_preds_shuffle[j][:i]**2)/i
        std = np.sqrt((f_2-f**2)/i)
        NN_MC_std.append(std)
        
    NN_css.append(NN_cs)
    NN_MC_stds.append(NN_MC_std)
    
print ('############### Plotting convergence ###############')
    
fig = plt.figure(1)   
plt.plot(cs_range, np.array(NJ_cs)/cs_range, label = 'NJet')

for i in range(training_reruns):
    plt.plot(cs_range, np.array(NN_css[i])/cs_range)

plt.legend()
if order == 'NLO':
    plt.title('{} '.format(order)+r'differential k-factor against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
else:
    plt.title('{} '.format(order)+r'differential cross section against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
plt.savefig(model_dir + '/cs_convergence_{}_{}.png'.format(points,test_points), dpi = 250,bbox_inches='tight')
plt.close()

NN_css = np.array(NN_css)

NN_means = []
NN_stds = []
for i in range(len(cs_range)):
    NN_means.append(np.mean(NN_css[:,i]))
    NN_stds.append(np.std(NN_css[:,i]/cs_range[i],ddof=1))
    
NN_stds = np.array(NN_stds)
NN_std_errs = NN_stds/np.array(np.sqrt(training_reruns))
NN_MC_std = np.array(NN_MC_stds[0])

# Run without MC error on the dataset 
#NN_err = np.sqrt(NN_stds**2+NN_MC_std**2)
NN_err = np.array(NN_stds)
#NN_MC_std_err = np.sqrt(np.array(NN_std_errs)**2+NN_MC_std**2)
NN_MC_std_err = np.array(NN_std_errs)

print ('############### Plotting mean with errors ###############')

fig = plt.figure(1)
plt.errorbar(cs_range, np.array(NN_means)/cs_range, yerr=NN_err)
plt.errorbar(cs_range, np.array(NJ_cs)/cs_range, label = 'NJet', yerr=NJ_MC_std, alpha=0.5)
plt.legend()
if order == 'NLO':
    plt.title('Average {} '.format(order)+r'differential k-factor against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
else:
    plt.title('Average {} '.format(order)+r'differential cross section against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
plt.savefig(model_dir + '/cs_mean_convergence_{}_{}.png'.format(points,test_points), dpi = 250,bbox_inches='tight')
plt.close()

print ('############### Plotting mean with standard errors ###############')

fig = plt.figure(1)
plt.errorbar(cs_range, np.array(NN_means)/cs_range, yerr=np.array(NN_MC_std_err))
plt.errorbar(cs_range, np.array(NJ_cs)/cs_range, label = 'NJet', yerr=NJ_MC_std, alpha=0.5)
plt.legend()
if order == 'NLO':
    plt.title('Average {} '.format(order)+r'differential k-factor against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
else:
    plt.title('Average {} '.format(order)+r'differential cross section against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
plt.savefig(model_dir + '/cs_mean_convergence_stderr_{}_{}.png'.format(points, test_points), dpi = 250,bbox_inches='tight')
plt.close()

print ('############### Finished ###############')

