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
import yaml

from n3jet.utils import FKSPartition
from n3jet.models import Model

parser = argparse.ArgumentParser(description=
                                 """
                                 Training multiple models on the same dataset for error analysis. 
                                 Here we assume that the momenta and njet files already exist and 
                                 will be passed to the script by the user
                                 """
)

parser.add_argument(
    '--yaml_file',
    dest='yaml_file',
    help='YAML file with config parameters',
    type=str,
    default = "False"
)

parser.add_argument(
    '--mom_file',
    dest='mom_file',
    help='destination of momenta file',
    type=str,
)

parser.add_argument(
    '--nj_file',
    dest='nj_file',
    help='NJet file',
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



args = parser.parse_args()
yaml_file = args.yaml_file
mom_file = args.mom_file
nj_file = args.nj_file
delta_cut = args.delta_cut
model_base_dir = args.model_base_dir
model_dir = args.model_dir
training_reruns = args.training_reruns
all_legs = args.all_legs

lr=0.01

def file_exists(file_path):
    if os.path.exists(file_path) == True:
        pass
    else:
        raise ValueError('{} does not exist'.format(file_path))

if yaml_file != "False":
    file_exists(yaml_file)

    with open(yaml_file) as f:
        yaml = yaml.load(f, Loader=yaml.FullLoader)
    
    mom_file = yaml["training"]["mom_file"]
    nj_file = yaml["training"]["nj_file"]
    delta_cut = yaml["delta_cut"]
    model_base_dir = yaml["model_base_dir"]
    model_dir = yaml["model_dir"]
    training_reruns = yaml["training_reruns"]
    all_legs = yaml["all_legs"]
    layers = yaml["training"].get("layers", [20,40,20])
    
file_exists(mom_file)
file_exists(nj_file)
file_exists(model_base_dir)

momenta = np.load(mom_file,allow_pickle=True)

print ('############### Momenta loaded ###############')

nj = np.load(nj_file,allow_pickle=True)

print ('############### NJet loaded ###############')

momenta = momenta.tolist()

print ('Training on {} PS points'.format(len(momenta)))

print ('############### Training models ###############')

if os.path.exists(model_base_dir) == False:
    os.mkdir(model_base_dir)
    print ('Creating base directory')
else:
    print ('Base directory already exists')

nlegs = len(momenta[0])-2

if all_legs == "True":
    print ('Recutting for all legs')
    fks = FKSPartition(
        momenta = momenta,
        labels = nj,
        all_legs = True
    )
    cut_momenta, near_momenta, cut_nj, near_nj = fks.cut_near_split(delta_cut=delta_cut, delta_near=0.02)
    momenta = np.concatenate((cut_momenta, near_momenta))
    nj = np.concatenate((cut_nj, near_nj))
    indices = np.arange(len(nj))
    np.random.shuffle(indices)
    momenta = momenta[indices]
    nj = nj[indices]
    momenta = momenta.tolist()
    
for i in range(training_reruns):
    print ('Working on model {}'.format(i))
    model_dir_new = model_base_dir + model_dir + '_{}/'.format(i)
    print ('Looking for directory {}'.format(model_dir_new))
    if os.path.exists(model_dir_new) == False:
        os.mkdir(model_dir_new)
        print ('Directory created')
    else:
        print ('Directory already exists')

    if all_legs == 'False':    
        NN = Model(
            intput_size = nlegs*4,
            momenta = momenta,
            labels = nj,
            all_jets=True,
            all_legs=False
        )
    elif all_legs == 'True':
        NN = Model(
            intput_size = nlegs*4,
            momenta = momenta,
            labels = nj,
            all_jets=False,
            all_legs=True
        )

    else:
        raise ValueError("all_legs is neither True nor False, but is {}".format(all_legs))


    model, x_mean, x_std, y_mean, y_std = NN.fit(layers=layers, lr=lr, epochs=1000000)
    
    if model_dir != '':
        model.save(model_dir_new + '/model')
        with open (model_dir_new + '/model_arch.json', 'w') as fout:
            fout.write(model.to_json())
        model.save_weights(model_dir_new + '/model_weights.h5')
        metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
        pickle_out = open(model_dir_new + "/dataset_metadata.pickle","wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()

print ('############### Finished ###############')
