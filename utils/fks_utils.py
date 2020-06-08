import sys
sys.path.append('./../models/')
sys.path.append('./../phase/')
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
#import pickle
import cPickle as pickle
import multiprocessing

from njet_run_functions import *
from model import Model
from rambo_piecewise_balance import *
from fks_partition import *
from keras.models import load_model
from tqdm import tqdm

def cut_near_split(test_momenta, NJ_test, delta_cut, delta_near, all_legs=False):
    '''
    Split momenta into near and cut arrays - near is the region close to the PS cuts and the cut region is the rest of the cut PS
    :param test_momenta: list of momenta
    :param NJ_test: array of NJet ground truth results
    :param delta_cut: the PS cut delta
    :param delta_near: the secondary 'cut' defining the region 'close to' the cut boundary
    '''
    if type(test_momenta) != list:
        raise AssertionError('Momentum must be in the form of a list')
    
    test_cut_momenta = []
    test_near_momenta = []
    NJ_near_test_treevals = []
    NJ_cut_test_treevals = []
    for idx, i in tqdm(enumerate(test_momenta), total = len(test_momenta)):
        if all_legs:
            close, min_distance = check_all(i, delta=delta_cut,s_com=dot(i[0],i[1]),all_legs=True)
        else:
            close, min_distance = check_all(i, delta=delta_cut,s_com=dot(i[0],i[1]),all_legs=False)
        if close == False:
            if min_distance < delta_near:
                test_near_momenta.append(i)
                NJ_near_test_treevals.append(NJ_test[idx])
            else:
                test_cut_momenta.append(i)
                NJ_cut_test_treevals.append(NJ_test[idx])
    return test_cut_momenta, test_near_momenta, NJ_near_test_treevals, NJ_cut_test_treevals


def weighting(moms, n_gluon, labs):
    '''
    Weights scattering amplitudes according to the different partition function for pairs of particle
    
    :param moms: list of momenta
    :param n_gluon: the number of gluon jets
    :param labs: NJet labels
    '''
    
    if type(moms) != list:
        raise AssertionError('Momentum must be in the form of a list')
    
    D_1, pairs = D_ij(moms[0],n_gluon)
    S_near = []
    
    for idx, i in enumerate(pairs):
        print ('Pair {} of {}'.format(idx+1, len(pairs)))
        S = []
        for j in tqdm(moms):
            S.append(S_ij(j,n_gluon,i[0],i[1]))
        S_near.append(np.array(S))
    S_near = np.array(S_near)
    
    labs_split = []
    for i in S_near:
        labs_split.append(labs*i)
    
    return pairs, labs_split

def weighting_all(moms, labs):
    if type(moms) != list:
        raise AssertionError('Momentum must be in the form of a list')

    D_1, pairs = D_ij_all(moms[0])
    S_near = []

    for idx, i in enumerate(pairs):
        print ('Pair {} of {}'.format(idx+1, len(pairs)))
        S = []
        for j in tqdm(moms):
            S.append(S_ij_all(j, i[0], i[1]))
        S_near.append(np.array(S))
    S_near = np.array(S_near)

    labs_split = []
    for i in S_near:
        labs_split.append(labs*i)

    return pairs, labs_split

def pair_off(moms,pairs,n_gluon,S_near,job_index,job_indices):
    S_middle = []
    for idx, i in enumerate(pairs):
        S = []
        for j in tqdm(moms):
            S.append(S_ij(j,n_gluon,i[0],i[1]))
        S_middle.append(np.array(S))
    S_near.append(S_middle)
    job_indices.append(job_index)

def multiprocess_weighting(moms, n_gluon, labs, cores):
    '''
    Weights scattering amplitudes according to the different partition function for pairs of particle
    
    :param moms: list of momenta
    :param n_gluon: the number of gluon jets
    :param labs: NJet labels
    '''
    
    if type(moms) != list:
        raise AssertionError('Momentum must be in the form of a list')
    
    if type(moms) != list:
        raise AssertionError('The momentum array is not a list')
    D_1, pairs = D_ij(moms[0],n_gluon)
    
    indices = np.arange(len(pairs)+1, step = len(pairs)/cores)
    if indices[-1] != len(pairs):
        indices[-1] = len(pairs)
    
    processes = []
    manager = multiprocessing.Manager()
    S_n =  manager.list([])
    job_indices = manager.list([])
    
    job_index = 0
    for idx in range(len(indices)-1):
        p = multiprocessing.Process(target=pair_off, args=(moms, pairs[indices[idx]:indices[idx+1]],n_gluon,S_n,job_index,job_indices))
        processes.append(p)
        p.start()
        job_index += 1
        
    for process in processes:
        process.join()
        
    job_indices_sort, S_n_sort = zip(*sorted(zip(list(job_indices), list(S_n))))

    S_near = []
    for i in S_n_sort:
        for j in i:
            S_near.append(j*labs)
        
    return pairs, S_near

def train_near_networks(pairs, near_momenta, NJ_split, order, n_gluon, delta_near, points, model_dir = '', **kwargs):
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
            pair_dir = model_dir + '/{}_near_{}_{}_{}_{}_{}/'.format(order,i[0],i[1],n_gluon+2,delta_near,points)
        
            if os.path.exists(pair_dir) == False:
                os.mkdir(pair_dir)
            
            model.save(pair_dir + '/model')
            metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
            pickle_out = open(pair_dir + "/dataset_metadata.pickle","wb")
            pickle.dump(metadata, pickle_out)
            pickle_out.close()
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near


def train_near_networks_general(input_size, pairs, near_momenta, NJ_split, delta_near, model_dir = '', all_jets=False, all_legs=False, **kwargs):
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
            
            model.save(pair_dir + '/model')
            with open (pair_dir + '/model_arch.json', 'w') as fout:
                fout.write(model.to_json())
            model.save_weights(pair_dir + '/model_weights.h5')
            metadata = {'x_mean': x_mean, 'x_std': x_std, 'y_mean': y_mean, 'y_std': y_std}
            pickle_out = open(pair_dir + "/dataset_metadata.pickle","wb")
            pickle.dump(metadata, pickle_out)
            pickle_out.close()
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near

def train_network(moms, labs, n_gluon, layers, lr, models, x_means, x_stds, y_means, y_stds, indices, index):
    NN = Model((n_gluon+2-1)*4,moms,labs)
    model, x_mean, x_std, y_mean, y_std = NN.fit(layers=[20,40,20], lr=lr)
    
    indices.append(index)
    models.append(model)
    x_means.append(x_mean)
    x_stds.append(x_std)
    y_means.append(y_mean)
    y_stds.append(y_std)
    
    

def multiprocess_train_near_networks(pairs, near_momenta, NJ_split, order, n_gluon, delta_near, points, model_dir = '', **kwargs):
    
    lr = kwargs.get('lr', 0.001)
    print ('Using learning rate {}'.format(lr))
    
    if type(near_momenta) != list:
        raise AssertionError('Momentum must be in the form of a list')
        
    processes = []
    manager = multiprocessing.Manager()
    
    indices = manager.list([])
    
    m_n = manager.list([])
    
    x_m_n = manager.list([])
    x_s_n = manager.list([])
    y_m_n = manager.list([])
    y_s_n = manager.list([])
    
    for idx,i in enumerate(pairs):
        p = multiprocessing.Process(target=train_network, args=(near_momenta,NJ_split[idx],n_gluon,[20,40,20],lr,m_n,x_m_n,x_s_n,y_m_n,y_s_n,indices,idx))
        
        processes.append(p)
        p.start()
    
    for process in processes:
        process.join()
    
    indices_sort,model_near,x_mean_near,x_std_near,y_mean_near,y_std_near = zip(*sorted(zip(indices,m_n,x_m_n,x_s_n,y_m_n,y_s_n)))
    
    if model_dir != '':
        for idx,i in enumerate(pairs):    
            pair_dir = model_dir + '/{}_near_{}_{}_{}_{}_{}'.format(order,i[0],i[1],n_gluon+2,delta_near,points)
        
            if os.path.exists(pair_dir) == False:
                os.mkdir(pair_dir)
            
            model_near[idx].save(pair_dir + '/model')
            metadata = {'x_mean': x_mean_near[idx], 'x_std': x_std_near[idx], 'y_mean': y_mean_near[idx], 'y_std': y_std_near[idx]}
            pickle_out = open(pair_dir + "/dataset_metadata.pickle","wb")
            pickle.dump(metadata, pickle_out)
            pickle_out.close()
    
    
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
        
        model = load_model(pair_dir + '/model',custom_objects={'root_mean_squared_error':NN.root_mean_squared_error})
        model_near.append(model)
        pickle_out = open(pair_dir + "/dataset_metadata.pickle","rb")
        metadata = pickle.load(pickle_out)
        pickle_out.close()
        
        x_mean_near.append(metadata['x_mean'])
        y_mean_near.append(metadata['y_mean'])
        x_std_near.append(metadata['x_std'])
        y_std_near.append(metadata['y_std'])
        
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
        
        model = load_model(pair_dir + '/model',custom_objects={'root_mean_squared_error':NN.root_mean_squared_error})
        model_near.append(model)
        pickle_out = open(pair_dir + "/dataset_metadata.pickle","rb")
        metadata = pickle.load(pickle_out)
        pickle_out.close()
        
        x_mean_near.append(metadata['x_mean'])
        y_mean_near.append(metadata['y_mean'])
        x_std_near.append(metadata['x_std'])
        y_std_near.append(metadata['y_std'])
        
    return model_near, x_mean_near, x_std_near, y_mean_near, y_std_near


def train_cut_network_general(input_size, cut_momenta, NJ_cut, delta_near, model_dir = '', all_jets = False, all_legs=False, **kwargs):
    
    lr = kwargs.get('lr', 0.001)
    
    NN_cut = Model(input_size,cut_momenta,NJ_cut, all_jets, all_legs)
    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = NN_cut.fit(layers=[16,32,16], lr=lr)
    
    if model_dir != '':
        cut_dir = model_dir + 'cut_{}'.format(delta_near)
        
        if os.path.exists(cut_dir) == False:
            os.mkdir(cut_dir)
        
        model_cut.save(cut_dir + '/model')
        with open (cut_dir + '/model_arch.json', 'w') as fout:
            fout.write(model_cut.to_json())
        model_cut.save_weights(cut_dir + '/model_weights.h5')
        metadata = {'x_mean': x_mean_cut, 'x_std': x_std_cut, 'y_mean': y_mean_cut, 'y_std': y_std_cut}
        
        pickle_out = open(cut_dir + '/dataset_metadata.pickle',"wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut

def train_cut_network(cut_momenta, NJ_cut,order, n_gluon, delta_cut, points, model_dir = '', **kwargs):
    
    lr = kwargs.get('lr', 0.001)
    
    NN_cut = Model((n_gluon+2-1)*4,cut_momenta,NJ_cut)
    model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut = NN_cut.fit(layers=[16,32,16], lr=lr)
    
    if model_dir != '':
        if os.path.exists(model_dir + '{}_cut_{}_{}_{}'.format(order,n_gluon+2,delta_cut,points)) == False:
            os.mkdir(model_dir + '{}_cut_{}_{}_{}'.format(order,n_gluon+2,delta_cut,points))
        
        model_cut.save(model_dir + '{}_cut_{}_{}_{}/model'.format(order,n_gluon+2,delta_cut,points))
        metadata = {'x_mean': x_mean_cut, 'x_std': x_std_cut, 'y_mean': y_mean_cut, 'y_std': y_std_cut}
        
        pickle_out = open(model_dir + "{}_cut_{}_{}_{}/dataset_metadata.pickle".format(order,n_gluon+2,delta_cut,points),"wb")
        pickle.dump(metadata, pickle_out)
        pickle_out.close()
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut

def get_cut_network_general(NN, delta_near, model_dir):

    cut_dir = model_dir + 'cut_{}'.format(delta_near)
    
    model_cut = load_model(cut_dir + '/model',custom_objects={'root_mean_squared_error':NN.root_mean_squared_error})
    
    pickle_out = open(cut_dir + '/dataset_metadata.pickle',"rb")
    metadata = pickle.load(pickle_out)
    pickle_out.close()
    
    x_mean_cut = (metadata['x_mean'])
    y_mean_cut = (metadata['y_mean'])
    x_std_cut = (metadata['x_std'])
    y_std_cut = (metadata['y_std'])
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut


def get_cut_network(NN, order, n_gluon, delta_cut, points, model_dir):
    
    model_cut = load_model(model_dir + '{}_cut_{}_{}_{}/model'.format(order,n_gluon+2,delta_cut,points),custom_objects={'root_mean_squared_error':NN.root_mean_squared_error})
    
    pickle_out = open(model_dir + "{}_cut_{}_{}_{}/dataset_metadata.pickle".format(order,n_gluon+2,delta_cut,points),"rb")
    metadata = pickle.load(pickle_out)
    pickle_out.close()
    
    x_mean_cut = (metadata['x_mean'])
    y_mean_cut = (metadata['y_mean'])
    x_std_cut = (metadata['x_std'])
    y_std_cut = (metadata['y_std'])
    
    return model_cut, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut

def infer_on_near_splits(NN, moms, models, x_mean_near, x_std_near, y_mean_near, y_std_near):
    '''
    Infer on near networks
    
    :param moms: list of testing momenta
    :param models: array of near network models
    '''
    
    y_pred_nears = np.zeros(len(moms))
    for i in range(len(models)):
        test = i
        x_standard_near = NN.process_testing_data(moms=moms,x_mean=x_mean_near[test],x_std=x_std_near[test],y_mean=y_mean_near[test],y_std=y_std_near[test])
        pred_near = models[test].predict(x_standard_near)
        y_pred_near = NN.destandardise_data(pred_near.reshape(-1),x_mean=x_mean_near[test],x_std=x_std_near[test],y_mean=y_mean_near[test],y_std=y_std_near[test])
        y_pred_nears += np.array(y_pred_near)
    return y_pred_nears

def infer_on_near_splits_separate(NN, moms, models, x_mean_near, x_std_near, y_mean_near, y_std_near):
    '''
    Infer on near networks
    
    :param moms: list of testing momenta
    :param models: array of near network models
    '''
    
    y_preds_nears = []
    y_pred_nears = np.zeros(len(moms))
    for i in range(len(models)):
        test = i
        x_standard_near = NN.process_testing_data(moms=moms,x_mean=x_mean_near[test],x_std=x_std_near[test],y_mean=y_mean_near[test],y_std=y_std_near[test])
        pred_near = models[test].predict(x_standard_near)
        y_pred_near = NN.destandardise_data(pred_near.reshape(-1),x_mean=x_mean_near[test],x_std=x_std_near[test],y_mean=y_mean_near[test],y_std=y_std_near[test])
        y_pred_nears += np.array(y_pred_near)
        y_preds_nears.append(y_pred_near)
    return y_preds_nears, y_pred_nears


def infer_on_cut(NN, moms, model, x_mean_cut, x_std_cut, y_mean_cut, y_std_cut):
    
    x_standard_cut = NN.process_testing_data(moms=moms,x_mean=x_mean_cut,x_std=x_std_cut,y_mean=y_mean_cut,y_std=y_std_cut)
    pred_cut = model.predict(x_standard_cut)
    y_pred_cuts = NN.destandardise_data(pred_cut.reshape(-1),x_mean=x_mean_cut,x_std=x_std_cut,y_mean=y_mean_cut,y_std=y_std_cut)
    
    return y_pred_cuts



