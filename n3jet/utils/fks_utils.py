import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
from keras.models import load_model
from tqdm import tqdm

# python 2/3 compatibility
try:
    import cPickle as pickle
except:
    import pickle

from n3jet.models import Model

###############################################################################################
###############################      TRAINING ON NEAR NETWORKS  ###############################
###############################################################################################

def train_near_networks(
        pairs,
        near_momenta,
        NJ_split,
        order,
        n_gluon,
        delta_near,
        points,
        model_dir = '',
        **kwargs
):
    '''
    Train 'near' networks on pairs of jets
    
    :param pairs: array of pairs of jet positions
    :param near_momenta: list of PS points between delta_near and delta_cut
    :param NJ_split: array of NJet results weighted by different partition functions
    :param model_dir: the directory in which to create sub-directories to save the networks
    '''
    
    lr = kwargs.get('lr', 0.001)
    layers = kwargs.get('layers', [20,40,20])
    print ('Using learning rate {}'.format(lr))
    epochs = kwargs.get('epochs', 1000000)
    
    if type(near_momenta) != list:
        raise AssertionError('Momentum must be in the form of a list')
    
    NN_near = []
    model_near = []
    
    x_mean_near = []
    x_std_near = []
    y_mean_near = []
    y_std_near = []
    for idx,i in enumerate(pairs):
        NN = Model((n_gluon+2-1)*4,near_momenta, NJ_split[idx])
        
        model, x_mean, x_std, y_mean, y_std = NN.fit(layers=layers, lr=lr, epochs=epochs)
        
        NN_near.append(NN)
        model_near.append(model)
        x_mean_near.append(x_mean)
        x_std_near.append(x_std)
        y_mean_near.append(y_mean)
        y_std_near.append(y_std)
        
        
        if model_dir != '':
            pair_dir = model_dir + '/{}_near_{}_{}_{}_{}_{}/'.format(
                order,i[0],i[1],n_gluon+2,delta_near,points
            )
        
            if os.path.exists(pair_dir) == False:
                os.mkdir(pair_dir)
            
            model.save(pair_dir + '/model')
            metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
            pickle_out = open(pair_dir + "/dataset_metadata.pickle","wb")
            pickle.dump(metadata, pickle_out)
            pickle_out.close()
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near


def train_near_networks_general(
        input_size,
        pairs,
        near_momenta,
        NJ_split,
        delta_near,
        model_dir = '',
        all_jets=False,
        all_legs=False,
        **kwargs
):
    '''
    Train 'near' networks on pairs of jets
    '''
    
    lr = kwargs.get('lr', 0.001)
    layers = kwargs.get('layers', [20,40,20])
    print ('Using learning rate {}'.format(lr))
    epochs = kwargs.get('epochs', 1000000)
    
    if type(near_momenta) != list:
        raise AssertionError('Momentum must be in the form of a list')
    
    NN_near = []
    model_near = []
    
    x_mean_near = []
    x_std_near = []
    y_mean_near = []
    y_std_near = []
    for idx,i in enumerate(pairs):
        NN = Model(input_size,near_momenta, NJ_split[idx], all_jets, all_legs)
        
        model, x_mean, x_std, y_mean, y_std = NN.fit(layers=layers, lr=lr, epochs=epochs)
        
        NN_near.append(NN)
        model_near.append(model)
        x_mean_near.append(x_mean)
        x_std_near.append(x_std)
        y_mean_near.append(y_mean)
        y_std_near.append(y_std)
        
        
        if model_dir != '':
            pair_dir = model_dir + 'pair_{}_{}'.format(delta_near,idx)
        
            if os.path.exists(pair_dir) == False:
                os.mkdir(pair_dir)
            njet
            model.save(pair_dir + '/model')
            with open (pair_dir + '/model_arch.json', 'w') as fout:
                fout.write(model.to_json())
            model.save_weights(pair_dir + '/model_weights.h5')
            metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
            pickle_out = open(pair_dir + "/dataset_metadata.pickle","wb")
            pickle.dump(metadata, pickle_out)
            pickle_out.close()
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near


def get_near_networks(NN, pairs, order, n_gluon, delta_near, points, model_dir):
    '''
    Retrieve the near networks given the assigned file structure
    
    :param NN: NN object from the Model class
    :param pairs: array of numbers defining jet positions
    :param order: LO or NLO
    :param n_gluon: the number of gluon jets
    :param delta_near: the secondary 'cut' defining the region 'close to' the cut boundary
    :param points: the number of training points
    :param model_dir: the directory in which to create sub-directories to save the networks
    '''
    model_near = []

    x_mean_near = []
    x_std_near = []
    y_mean_near = []
    y_std_near = []
    for idx,i in enumerate(pairs):
        
        pair_dir = model_dir + '{}_near_{}_{}_{}_{}_{}'.format(order, i[0],i[1],n_gluon+2,delta_near,points)
        
        model = load_model(
            pair_dir + '/model',
            custom_objects={'root_mean_squared_error':NN.root_mean_squared_error}
        )
        model_near.append(model)
        pickle_out = open(pair_dir + "/dataset_metadata.pickle","rb")
        metadata = pickle.load(pickle_out)
        pickle_out.close()
        
        x_mean_near.append(metadata['x_mean'])
        y_mean_near.append(metadata['y_mean'])
        x_std_near.append(metadata['x_std'])
        y_std_near.append(metadata['y_std'])
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near
    
def get_near_networks_general(NN, pairs, delta_near, model_dir):
    '''
    Retrieve the near networks given the assigned file structure
    '''
    model_near = []

    x_mean_near = []
    x_std_near = []
    y_mean_near = []
    y_std_near = []
    for idx,i in enumerate(pairs):
        
        pair_dir = model_dir + 'pair_{}_{}'.format(delta_near,idx)
        
        model = load_model(
            pair_dir + '/model',
            custom_objects={'root_mean_squared_error':NN.root_mean_squared_error}
        )
        model_near.append(model)
        pickle_out = open(pair_dir + "/dataset_metadata.pickle","rb")
        metadata = pickle.load(pickle_out)
        pickle_out.close()
        
        x_mean_near.append(metadata['x_mean'])
        y_mean_near.append(metadata['y_mean'])
        x_std_near.append(metadata['x_std'])
        y_std_near.append(metadata['y_std'])
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near

###############################################################################################
###############################      TRAINING ON CUT NETWORKS  ###############################
###############################################################################################


def train_cut_network(
        cut_momenta,
        NJ_cut,order,
        n_gluon,
        delta_cut,
        points,
        model_dir = '',
        **kwargs
):
    
    lr = kwargs.get('lr', 0.001)
    
    NN_cut = Model((n_gluon+2-1)*4,cut_momenta,NJ_cut)
    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = NN_cut.fit(layers=[16,32,16], lr=lr)
    
    if model_dir != '':
        if not os.path.exists(
                model_dir + '{}_cut_{}_{}_{}'.format(order,n_gluon+2,delta_cut,points)
        ):
            os.mkdir(model_dir + '{}_cut_{}_{}_{}'.format(order,n_gluon+2,delta_cut,points))
        
        model_cut.save(model_dir + '{}_cut_{}_{}_{}/model'.format(order,n_gluon+2,delta_cut,points))
        metadata = {
            'x_mean': x_mean_cut,
            'x_std': x_std_cut,
            'y_mean': y_mean_cut,
            'y_std': y_std_cut
        }
        
        pickle_out = open(model_dir + "{}_cut_{}_{}_{}/dataset_metadata.pickle".format(order,n_gluon+2,delta_cut,points),"wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut


def train_cut_network_general(
        input_size,
        cut_momenta,
        NJ_cut,
        delta_near,
        model_dir = '',
        all_jets = False,
        all_legs=False,
        **kwargs
):
    
    lr = kwargs.get('lr', 0.001)
    
    NN_cut = Model(input_size,cut_momenta,NJ_cut, all_jets, all_legs)
    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = NN_cut.fit(layers=[16,32,16], lr=lr)
    
    if model_dir != '':
        cut_dir = model_dir + 'cut_{}'.format(delta_near)
        
        if not os.path.exists(cut_dir):
            os.mkdir(cut_dir)
        
        model_cut.save(cut_dir + '/model')

        with open (cut_dir + '/model_arch.json', 'w') as fout:
            fout.write(model_cut.to_json())
        model_cut.save_weights(cut_dir + '/model_weights.h5')

        metadata = {
            'x_mean': x_mean_cut,
            'x_std': x_std_cut,
            'y_mean': y_mean_cut,
            'y_std': y_std_cut
        }
        
        pickle_out = open(cut_dir + '/dataset_metadata.pickle',"wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut


def get_cut_network(NN, order, n_gluon, delta_cut, points, model_dir):
    
    model_cut = load_model(
        model_dir + '{}_cut_{}_{}_{}/model'.format(
            order,n_gluon+2,delta_cut,points
        ),
        custom_objects={'root_mean_squared_error':NN.root_mean_squared_error}
    )
    
    pickle_out = open(model_dir + "{}_cut_{}_{}_{}/dataset_metadata.pickle".format(order,n_gluon+2,delta_cut,points),"rb")
    metadata = pickle.load(pickle_out)
    pickle_out.close()
    
    x_mean_cut = (metadata['x_mean'])
    y_mean_cut = (metadata['y_mean'])
    x_std_cut = (metadata['x_std'])
    y_std_cut = (metadata['y_std'])
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut


def get_cut_network_general(NN, delta_near, model_dir):

    cut_dir = model_dir + 'cut_{}'.format(delta_near)
    
    model_cut = load_model(
        cut_dir + '/model',
        custom_objects={'root_mean_squared_error':NN.root_mean_squared_error}
    )
    
    pickle_out = open(cut_dir + '/dataset_metadata.pickle',"rb")
    metadata = pickle.load(pickle_out)
    pickle_out.close()
    
    x_mean_cut = (metadata['x_mean'])
    y_mean_cut = (metadata['y_mean'])
    x_std_cut = (metadata['x_std'])
    y_std_cut = (metadata['y_std'])
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut


###############################################################################################
###############################     INFERRING ON NEAR NETWORKS  ###############################
###############################################################################################


def infer_on_near_splits(NN, moms, models, x_mean_near, x_std_near, y_mean_near, y_std_near):
    '''
    Infer on near networks
    
    :param moms: list of testing momenta
    :param models: array of near network models
    '''
    
    y_pred_nears = np.zeros(len(moms))
    for i in range(len(models)):
        test = i
        x_standard_near = NN.process_testing_data(
            moms=moms,
            x_mean=x_mean_near[test],
            x_std=x_std_near[test],
            y_mean=y_mean_near[test],
            y_std=y_std_near[test]
        )
        
        pred_near = models[test].predict(x_standard_near)
        y_pred_near = NN.destandardise_data(
            pred_near.reshape(-1),
            x_mean=x_mean_near[test],
            x_std=x_std_near[test],
            y_mean=y_mean_near[test],
            y_std=y_std_near[test]
        )
        y_pred_nears += np.array(y_pred_near)
    return y_pred_nears

def infer_on_near_splits_separate(
        NN,
        moms,
        models,
        x_mean_near,
        x_std_near,
        y_mean_near,
        y_std_near
):
    '''
    Infer on near networks
    
    :param moms: list of testing momenta
    :param models: array of near network models
    '''
    
    y_preds_nears = []
    y_pred_nears = np.zeros(len(moms))
    for i in range(len(models)):
        test = i
        x_standard_near = NN.process_testing_data(
            moms=moms,
            x_mean=x_mean_near[test],
            x_std=x_std_near[test],
            y_mean=y_mean_near[test],
            y_std=y_std_near[test]
        )
        pred_near = models[test].predict(x_standard_near)
        y_pred_near = NN.destandardise_data(
            pred_near.reshape(-1),
            x_mean=x_mean_near[test],
            x_std=x_std_near[test],
            y_mean=y_mean_near[test],
            y_std=y_std_near[test]
        )
        y_pred_nears += np.array(y_pred_near)
        y_preds_nears.append(y_pred_near)
    return y_preds_nears, y_pred_nears


def infer_on_cut(NN, moms, model, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut):
    
    x_standard_cut = NN.process_testing_data(
        moms=moms,
        x_mean=x_mean_cut,
        x_std=x_std_cut,
        y_mean=y_mean_cut,
        y_std=y_std_cut
    )
    
    pred_cut = model.predict(x_standard_cut)
    
    y_pred_cuts = NN.destandardise_data(
        pred_cut.reshape(-1),
        x_mean=x_mean_cut,
        x_std=x_std_cut,
        y_mean=y_mean_cut,
        y_std=y_std_cut
    )
    
    return y_pred_cuts



