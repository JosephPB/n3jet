import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import pickle
from njet_run_functions import *

known = {
        21: (2, 2),
        1: (2, 1),
        2: (2, 1),
        3: (2, 1),
        4: (2, 1),
        5: (2, 1),
        # 6 : (2,1),  # skip the t-quark
        22: (2, 0),
        25: (1, 0),
        11: (2, 0),
        12: (2, 0),
        13: (2, 0),
        14: (2, 0),
        15: (2, 0),
        16: (2, 0),
    }
# helicity dof's and colour dof's (0 - singlet, 1 - fundamental, 2 - adjoint)

def get_alpha_powers(process_in, process_out):
    full = process_in + process_out

    aspow = sum(known[abs(p)][1] != 0 for p in full) - 2
    aepow = len(full) - (aspow + 2)

    print ('AlphasPower = {}, AlphaPower = {}'.format(aspow, aepow))


    return aspow, aepow

def run_njet_generic(process, **kwargs):
    '''
    :param process: 2D list of different lengths
        process[0] = array of incoming particle MC numbers
        process[1] = array of outgoing particle MC numbers
    :param aspow: alpha_s power
    :param awpow: alpha_e power
    '''
    
    run_accuracy = kwargs.get('run_accuracy', False)
    mur = kwargs.get('mur', None)
    
    process_in = process[0]
    process_out = process[1]
    
    t = 'NJ_4j_test'
    channel_name = 'eeqq'
    channel_inc = process_in
    channel_out = process_out
    
    mods, tests = action_run(t)
    
    curorder, curtests = run_tests(mods, tests)

    aspow, aepow = get_alpha_powers(process_in, process_out)
    
    curtests[0]['test']['params']['aspow'] = aspow
    curtests[0]['test']['params']['aepow'] = aepow
    curtests[0]['test']['params']['ae'] = 1.
    
    if mur is not None:
        mur = mur #*707.1067811865476 #91.188
        curtests[0]['test']['params']['mur'] = mur
        
    
    # add error checking to order file
    if run_accuracy == True:
        curorder += '\nNJetReturnAccuracy yes'
        
    curtests[0]['test']['data'] = \
    [{'born': 0,
      #'has_lc': False,
      'inc': channel_inc,
      'loop': 0,
      'mcn': 1,
      'name': channel_name,
      'out': channel_out}]
    
    # pass the curtests to the run_bactj function which will run njet_init    
    test_data, ptype, order = run_batch(curorder, curtests)
    
    return test_data, ptype, order


def run_njet(n_gluon, **kwargs):
    '''
    Run function for njet initilisation specifically designed for e+e-->q\bar{q} + jets
    '''
    
    run_accuracy = kwargs.get('run_accuracy', False)
    mur = kwargs.get('mur_factor', None)
    
    
    print (mur)
    t = 'NJ_4j_test'
    channel_name = 'eeqq'
    channel_inc = [11,-11]
    channel_out = [-1,1]
    for i in range(n_gluon):
        channel_name += 'G'
        channel_out.append(21)
    aspow = n_gluon
    aepow = 2
    
    mods, tests = action_run(t)
    
    curorder, curtests = run_tests(mods, tests)
    
    
    
    curtests[0]['test']['params']['aspow'] = aspow
    curtests[0]['test']['params']['aepow'] = aepow
    curtests[0]['test']['params']['ae'] = 1.
    
    if mur is not None:
        mur = mur*707.1067811865476
        curtests[0]['test']['params']['mur'] = mur
        
    
    # add error checking to order file
    if run_accuracy == True:
        curorder += '\nNJetReturnAccuracy yes'
    
    #print (curtests[0]['test']['params']['mur'])
    
    curtests[0]['test']['data'] = \
    [{'born': 0,
      #'has_lc': False,
      'inc': channel_inc,
      'loop': 0,
      'mcn': 1,
      'name': channel_name,
      'out': channel_out}]
    
    # pass the curtests to the run_bactj function which will run njet_init    
    test_data, ptype, order = run_batch(curorder, curtests)
    
    return test_data, ptype, order


def load_momenta(data_dir, file, python_3 = False):
    if python_3 == True:
        momenta = np.load(data_dir + file, allow_pickle = True, encoding='latin1')
    else:
        momenta = np.load(data_dir + file, allow_pickle = True)
    return momenta

def save_momenta(data_dir, file, momenta):
    np.save(data_dir + file + '.npy', momenta, allow_pickle = True)

def load_njet(data_dir, file, python_3 = False):
    if python_3 == True:
        njet = np.load(data_dir + file, allow_pickle = True, encoding='latin1')
    else:
        njet = np.load(data_dir + file, allow_pickle = True)
    return njet

def save_njet(data_dir, file, njet):
    np.save(data_dir + file + '.npy', njet, allow_pickle = True)

def dot(p1,p2):
    'Minkowski metric dot product'
    prod = p1[0]*p2[0]-(p1[1]*p2[1]+p1[2]*p2[2]+p1[3]*p2[3])
    return prod

def cs(test_momenta_array, NJ_test, y_pred):
    '''
    Calculate cross section and MC errors
    '''
    
    test_momenta_array = np.array(test_momenta_array)
    cs_range = np.arange(10000,(len(test_momenta_array)/10000)*10001,10000)
    
    NJ_cs = []
    NJ_std = []
    NN_cs = []
    NN_std = []
    for i in cs_range:
        # cs
        NJ_to_sum = NJ_test[:i]
        NJ_cs.append(np.sum(NJ_to_sum))
        # error
        f = np.sum(NJ_test[:i])/i
        f_2 = np.sum(NJ_test[:i]**2)/i
        std = np.sqrt((f_2-f**2)/i)
        NJ_std.append(std)
        
        # cs
        NN_to_sum = y_pred[:i]
        NN_cs.append(np.sum(NN_to_sum))
        # error
        f = np.sum(y_pred[:i])/i
        f_2 = np.sum(y_pred[:i]**2)/i
        std = np.sqrt((f_2-f**2)/i)
        NN_std.append(std)
    return cs_range, NJ_cs, NJ_std, NN_cs, NN_std
