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
from n3jet.utils.fks_utils import (
    train_near_networks_general,
    train_cut_network_general
)
from n3jet.utils.general_utils import (
    bool_convert,
    file_exists
)
from n3jet.models import Model

def parse():
    """
    Parse arguments
    """

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

    return args


class FKSModelRun:

    def __init__(
            mom_file,
            nj_file,
            delta_cut,
            delta_near,
            model_base_dir,
            model_dir,
            training_reruns,
            all_legs,
            all_pairs,
            lr=0.01
    ):
        self.mom_file = mom_file
        self.nj_file = nj_file
        self.delta_cut = delta_cut
        self.delta_near = delta_near
        self.model_base_dir = model_base_dir
        self.model_dir = model_dir
        self.training_reruns = training_reruns
        self.all_legs = all_legs
        self.all_pairs = all_pairs
        self.lr=lr

        file_exists(mom_file)
        file_exists(nj_file)
        file_exists(model_base_dir)

        if os.path.exists(model_base_dir) == False:
            os.mkdir(model_base_dir)
            print ('Creating base directory')
        else:
            print ('Base directory already exists')

    @classmethod
    def from_yaml(yaml_file, training=True):
        "Initiate from YAML file"
        file_exists(yaml_file)

        with open(yaml_file) as f:
            yaml = yaml.load(f, Loader=yaml.FullLoader)

        if training:
            mom_file = yaml["training"]["mom_file"]
            nj_file = yaml["training"]["nj_file"]
        else:
            mom_file = yaml["testing"]["mom_file"]
            nj_file = yaml["testing"]["nj_file"]
        delta_cut = yaml["delta_cut"]
        delta_near = yaml["delta_near"]
        model_base_dir = yaml["model_base_dir"]
        model_dir = yaml["model_dir"]
        training_reruns = yaml["training"]["training_reruns"]
        all_legs = bool_convert(yaml["all_legs"])
        all_pairs = bool_convert(yaml["all_pairs"])
        layers = yaml["training"].get("layers", [20,40,20])
        lr = yaml["training"].get("lr", 0.01)
        
        return FKSModelRun(
            mom_file = mom_file,
            nj_file = nj_file,
            delta_cut = delta_cut,
            delta_near = delta_near,
            model_base_dir = model_base_dir,
            model_dir = model_dir,
            training_reruns = training_reruns,
            all_legs = all_legs,
            all_pairs = all_pairs,
            layers = layers,
            lr = lr
        )

    def load_data(self):

        momenta = np.load(mom_file,allow_pickle=True)
        print ('############### Momenta loaded ###############')
        
        nj = np.load(nj_file,allow_pickle=True)
        print ('############### NJet loaded ###############')

        momenta = momenta.tolist()
        print ('Training on {} PS points'.format(len(momenta)))

        self.nlegs = len(momenta[0])-2

        return momenta, nj
    
    
    def split_data(self, momenta, nj):

        fks = FKSPartition(
            momenta = momenta,
            labels = nj,
            all_legs = self.all_legs
        )
            
        cut_momenta, near_momenta, cut_nj, near_nj = fks.cut_near_split(
            delta_cut = self.delta_cut,
            delta_near = self.delta_near
        )

        pairs, near_nj_split = fks.weighting()

        return cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split

    def train_networks(self, cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split):

        for i in range(self.training_reruns):
            print ('Working on model {}'.format(i))
            model_dir_new = self.model_base_dir + self.model_dir + '_{}/'.format(i)
            print ('Looking for directory {}'.format(model_dir_new))
            
            if os.path.exists(model_dir_new) == False:
                os.mkdir(model_dir_new)
                print ('Directory created')
            else:
                print ('Directory already exists')

            
            if self.all_legs:
                all_jets = True
            else:
                all_jets = False
            
            model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = train_near_networks_general(
                input_size = (self.nlegs)*4,
                pairs = pairs,
                near_momenta = near_momenta,
                NJ_split = near_nj_split,
                delta_near = self.delta_near,
                model_dir = model_dir_new,
                all_jets=all_jets,
                all_legs=self.all_legs,
                lr=self.lr,
                layers=layers
            )
            model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut =  train_cut_network_general(
                input_size = (self.nlegs)*4,
                cut_momenta = cut_momenta,
                NJ_cut = cut_nj,
                delta_cut = self.delta_cut,
                model_dir = model_dir_new,
                all_jets=all_jets,
                all_legs=self.all_legs,
                lr=self.lr,
                layers=layers
            )

    def load_models(self, cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split):

        if self.all_legs:
                all_jets = True
        else:
            all_jets = False

        NN = Model(
            input_size = (self.nlegs)*4,
            momenta = near_momenta,
            labels = near_nj_split[0],
            all_jets=all_jets,
            all_legs=self.all_legs
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

        for i in range(self.training_reruns):
            print ('Working on model {}'.format(i))
            model_dir_new = self.model_base_dir + self.model_dir + '_{}/'.format(i)
            print ('Looking for directory {}'.format(model_dir_new))
            if os.path.exists(model_dir_new) == False:
                os.mkdir(model_dir_new)
                print ('Directory created')
            else:
                print ('Directory already exists')

            model_near, x_mean_near, x_std_near, y_mean_near, y_std_near = get_near_networks_general(
                NN = NN,
                pairs = pairs,
                delta_near = self.delta_near,
                model_dir = model_dir_new
            )
            model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = get_cut_network_general(
                NN = NN,
                delta_cut = self.delta_cut,
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

        return model_nears, model_cuts, x_mean_nears, x_mean_cuts, x_std_nears, x_std_cuts, y_mean_nears, y_mean_cuts, y_std_nears, y_std_cuts

    def test_networks(
            near_momenta,
            cut_momenta,
            near_nj_split,
            model_nears,
            model_cuts,
            x_mean_nears,
            x_mean_cuts,
            x_std_nears,
            x_std_cuts,
            y_mean_nears,
            y_mean_cuts,
            y_std_nears,
            y_std_cuts
    ):

        NN = Model(
            input_size = (self.nlegs)*4,
            momenta = near_momenta,
            labels = near_nj_split[0],
            all_jets=all_jets,
            all_legs=self.all_legs
        )

        for i in range(self.training_reruns):
            print ('Predicting on model {}'.format(i))
            model_dir_new = self.model_base_dir + self.model_dir + '_{}/'.format(i)
            y_pred_near = infer_on_near_splits(
                NN = NN,
                moms = near_momenta,
                models = model_nears[i],
                x_mean_near = x_mean_nears[i],
                x_std_near = x_std_nears[i],
                y_mean_near = y_mean_nears[i],
                y_std_near = y_std_nears[i]
            )
            np.save(model_dir_new + '/pred_near_{}'.format(len(near_momenta + cut_momenta)), y_pred_near)
            
        for i in range(self.training_reruns):
            print ('Predicting on model {}'.format(i))
            model_dir_new = self.model_base_dir + self.model_dir + '_{}/'.format(i)
            y_pred_cut = infer_on_cut(
                NN = NN,
                moms = cut_momenta,
                model = model_cuts[i],
                x_mean_cut = x_mean_cuts[i],
                x_std_cut = x_std_cuts[i],
                y_mean_cut = y_mean_cuts[i],
                y_std_cut = y_std_cuts[i]
            )
            np.save(model_dir_new + '/pred_cut_{}'.format(len(near_momenta + cut_momenta)), y_pred_cut)
    

    def train(self):

        momenta, nj = self.load_data()
        cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split = self.split_data(momenta, nj)
        self.train_networks(cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split)
            
        print ('############### Finished ###############')

    def test(self):
        momenta, nj = self.load_data()
        
        cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split = self.split_data(momenta, nj)
    
        model_nears, model_cuts, x_mean_nears, x_mean_cuts, x_std_nears, x_std_cuts, y_mean_nears, y_mean_cuts, y_std_nears, y_std_cuts = self.load_models(NN, cut_momenta, near_momenta, cut_nj, near_nj, pairs, near_nj_split)
        
        self.test_networks(
            near_momenta,
            cut_momenta,
            near_nj_split,
            model_nears,
            model_cuts,
            x_mean_nears,
            x_mean_cuts,
            x_std_nears,
            x_std_cuts,
            y_mean_nears,
            y_mean_cuts,
            y_std_nears,
            y_std_cuts
        )
        
        print ('############### Finished ###############')
        

if __name__ == "__main__":

    args = parse()
    
    yaml_file = bool_convert(args.yaml_file)
    mom_file = args.mom_file
    nj_file = args.nj_file
    delta_cut = args.delta_cut
    delta_near = args.delta_near
    model_base_dir = args.model_base_dir
    model_dir = args.model_dir
    training_reruns = args.training_reruns
    all_legs = bool_convert(args.all_legs)
    all_pairs = bool_convert(args.all_pairs)

    
