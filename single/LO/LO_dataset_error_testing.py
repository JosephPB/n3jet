import sys
sys.path.append('./../')
sys.path.append('./../utils/')
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import pickle
import argparse
from keras.models import load_model

from njet_run_functions import *
from model_dataset import Model
from rambo_while import *
from uniform_utils import *
from utils import *

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
    '--test_points',
    dest='test_points',
    help='number of testing phase space points',
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
    '--training_reruns',
    dest='training_reruns',
    help='number of training reruns for testing, default: 1',
    type=int,
    default=1,
)

parser.add_argument(
    '--generate_points',
    dest='generate_points',
    help='whether or not to generate data even if it already exists',
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
    help='model directory extension on top of save_dir, default: /scratch/jbullock/njet/RAMBO/models/naive/error_analysis/',
    default='/scratch/jbullock/njet/RAMBO/models/naive/error_analysis/',
    type=str
)

args = parser.parse_args()

n_gluon = args.n_gluon
points = args.points
test_points = args.test_points
order = args.order
delta = args.delta
training_reruns = args.training_reruns
generate_points = args.generate_points
save_dir = args.save_dir
data_dir = args.data_dir
model_dir = args.model_dir

lr=0.01

data_dir = save_dir + data_dir
if model_dir != '':
    model_dir = model_dir + '/{}_{}_{}_{}/dataset_size/'.format(order,n_gluon+2,delta,points)
    

if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)
    print ('Creating directory')
else:
    print ('Model directory already exists')
    
mom = 'PS{}_{}_{}.npy'.format(n_gluon+2,delta,test_points)
labs = 'NJ_{}_{}_{}_{}.npy'.format(order,n_gluon+2,delta,test_points)

if os.path.exists(data_dir+mom) == False or generate_points == 'True':
    print ('No directory: {}'.format(data_dir+mom))
    print ('############### Creating phasespace test_points ###############')
    momenta = generate(n_gluon+2,test_points,1000.,delta=delta)
    np.save(data_dir + 'PS{}_{}_{}'.format(n_gluon+2,delta,test_points), momenta)
    
else:
    print ('############### All phasespace files exist ###############')

momenta = np.load(data_dir+mom,allow_pickle=True)
momenta = momenta.tolist()

test_data, _, _ = run_njet(n_gluon)

if os.path.exists(data_dir+labs) == False or generate_points == 'True':
    print ('No directory: {}'.format(data_dir+labs))
    print ('############### Creating njet test_points ###############')
    NJ_treevals = generate_LO_njet(momenta, test_data)
    np.save(data_dir + 'NJ_{}_{}_{}_{}'.format(order,n_gluon+2,delta,test_points), NJ_treevals)
    
else:
    print ('############### All njet files exist ###############')

NJ_treevals = np.load(data_dir+labs, allow_pickle=True)

if os.path.exists(model_dir) == False:
    os.mkdir(model_dir)
    print ('Creating directory')
else:
    print ('Directory already exists')

NN = Model((n_gluon+2-1)*4,momenta,NJ_treevals)
_,_,_,_,_,_,_,_ = NN.process_training_data()

models = []
x_means = []
y_means = []
x_stds = []
y_stds = []

for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    model_dir_new = model_dir + '/{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta,points,i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')
    model = load_model(model_dir_new + 'model')
    models.append(model)
    
    pickle_out = open(model_dir_new + "/dataset_metadata.pickle","rb")
    metadata = pickle.load(pickle_out)
    pickle_out.close()
    
    x_means.append(metadata['x_mean'])
    y_means.append(metadata['y_mean'])
    x_stds.append(metadata['x_std'])
    y_stds.append(metadata['y_std'])
    
print ('############### All models loaded ###############')

y_preds = []
for i in range(training_reruns):
    test = i
    print ('Predicting on model {}'.format(i))
    model_dir_new = model_dir + '/{}_{}_{}_{}_{}/'.format(order,n_gluon+2,delta,points,i)
    x_standard = NN.process_testing_data(moms=momenta,x_mean=x_means[test],x_std=x_stds[test],y_mean=y_means[test],y_std=y_stds[test])
    pred = models[test].predict(x_standard)
    y_pred = NN.destandardise_data(pred.reshape(-1),x_mean=x_means[test],x_std=x_stds[test],y_mean=y_means[test],y_std=y_stds[test])
    y_preds.append(y_pred)
    np.save(model_dir_new + '/pred_{}'.format(test_points), y_pred)
    
indices = np.arange(test_points)
np.random.shuffle(indices)

y_preds_shuffle = []
for i in range(training_reruns):
    y_preds_shuffle.append(np.array(y_preds[i])[indices])

test_momenta_array_shuffle = np.array(momenta)[indices]
NJ_test_shuffle = np.array(NJ_treevals)[indices]

cs_range = np.arange(10000,(len(test_momenta_array_shuffle)/10000)*10001,10000)

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
    plt.title('{} '.format(order)+r'virtual correction for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g'))
else:
    plt.title('{} '.format(order)+r'cross section for $e^+e^-\rightarrow\,q\bar{q}$'+'{} naive'.format(n_gluon*'g'))
plt.savefig(model_dir + '/cs_convergence_{}_{}_naive.png'.format(points,test_points), dpi = 250,bbox_inches='tight')
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

NN_err = np.sqrt(NN_stds**2+NN_MC_std**2)
NN_MC_std_err = np.sqrt(np.array(NN_std_errs)**2+NN_MC_std**2)


print ('############### Plotting mean with errors ###############')

fig = plt.figure(1)
plt.errorbar(cs_range, np.array(NN_means)/cs_range, yerr=NN_stds)
plt.errorbar(cs_range, np.array(NJ_cs)/cs_range, label = 'NJet', yerr=NJ_MC_std, alpha=0.5)
plt.legend()
if order == 'NLO':
    plt.title('Average {} '.format(order)+r'virtual correction for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g'))
else:
    plt.title('Average {} '.format(order)+r'cross section for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g'))
plt.savefig(model_dir + '/cs_mean_convergence_{}_{}_naive.png'.format(points,test_points), dpi = 250,bbox_inches='tight')
plt.close()

print ('############### Plotting mean with standard errors ###############')

fig = plt.figure(1)
plt.errorbar(cs_range, np.array(NN_means)/cs_range, yerr=np.array(NN_std_errs))
plt.errorbar(cs_range, np.array(NJ_cs)/cs_range, label = 'NJet', yerr=NJ_MC_std, alpha=0.5)
plt.legend()
if order == 'NLO':
    plt.title('Average {} '.format(order)+r'virtual correction for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g'))
else:
    plt.title('Average {} '.format(order)+r'cross section for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g'))
plt.savefig(model_dir + '/cs_mean_convergence_stderr_{}_{}_naive.png'.format(points,test_points), dpi = 250,bbox_inches='tight')
plt.close()

print ('############### Finished ###############')