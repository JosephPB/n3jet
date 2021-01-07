#############################################################
#                                                           #
# -Set of custom run functions for running njet in Python   #
# -Set of custom run functions for generating MEs from njet #
#                                                           #
#                                                           #
#                                                           #
#############################################################


import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import pickle

from n3jet.utils.njet_run_functions import (
    action_run,
    run_tests,
    run_batch,
    run_cc_test,
    run_generic_test
)

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
    
    curtests[0]['test']['data'] = \
    [{'born': 0,
      #'has_lc': False,
      'inc': channel_inc,
      'loop': 0,
      'mcn': 1,
      'name': channel_name,
      'out': channel_out}]
    
    # pass the curtests to the run_batch function which will run njet_init    
    test_data, ptype, order = run_batch(curorder, curtests)
    
    return test_data, ptype, order


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
    
    # pass the curtests to the run_batch function which will run njet_init    
    test_data, ptype, order = run_batch(curorder, curtests)
    
    return test_data, ptype, order


def generate_LO_njet(momenta, test_data):
    
    NJ_vals = []
    for i in test_data:
        vals = run_cc_test(momenta, i[1], i[2])
        NJ_vals.append(vals)
    
    # select the first test of the runs
    NJ_vals = NJ_vals[0]
    
    NJ_treevals = []
    for i in NJ_vals:
        NJ_treevals.append(i[0])
        
    return NJ_treevals


def generate_NLO_njet(momenta, test_data, VIEW = 'NJ', k = True):
    '''
    Generates NLO virtual correction and k-factors (1-loop/born) from NJet
    :param momenta: list of 4-momenta
    :param test_data: test_data generated by run_njet(_generic)() in n3jet/utils/run_njet_functions/
    :param VIEW: 'NJ' = (1-loop*born)/born = 1-loop, 'MC' = 1-loop*born
    :param k: True returns k-factor (i.e. if VIEW = 'NJ' then returns 1-loop/born, but if 'MC' returns 1-loop)
    '''
    
    NJ_loop_vals = []
    for i in test_data:
        vals = run_generic_test(momenta, i[1], i[2], VIEW = VIEW)
        NJ_loop_vals.append(vals)
    
    # select the first test of the runs
    NJ_loop_vals = NJ_loop_vals[0][0]
        
    A0 = []
    A1_2 = []
    A1_2_error = []
    A1_1 = []
    A1_1_error = []
    A1_0 = []
    A1_0_error = []
    for i in range(len(NJ_loop_vals)):
        A0.append(NJ_loop_vals[i][0][1])
        A1_2.append(NJ_loop_vals[i][1][1])
        A1_2_error.append(NJ_loop_vals[i][1][2])
        A1_1.append(NJ_loop_vals[i][2][1])
        A1_1_error.append(NJ_loop_vals[i][2][2])
        A1_0.append(NJ_loop_vals[i][3][1])
        A1_0_error.append(NJ_loop_vals[i][3][2])
        
    NJ_treevals = np.array(A0)
    A1_0 = np.array(A1_0)

    if k == True:
        k_factor = A1_0/NJ_treevals
        return k_factor, NJ_loop_vals
    else:
        return A1_0, NJ_loop_vals
