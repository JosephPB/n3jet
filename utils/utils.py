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


def plot_cs(test_momenta_array, cs_range, NJ_cs, NJ_std, NN_cs, NN_std, order, n_gluon, points, test_points, save_dir = ''):
    '''
    Plot cross section for different numbers of PS points for both the NJet output and the NN prediction and save output
    
    :param cs_range: array of slices of numbers of PS points
    :param *_cs: array of cross section bin values
    :param *_std: array of MC errors associated to each bin
    :param order: LO or NLO
    :param n_gluon: number of gluons in the process
    :param save_dir: directory within which to save plot
    '''
    
    std_error = [1,2,3]
    plots = []
    for i in std_error:
        fig = plt.figure(1)
        plt.errorbar(cs_range,np.array(NJ_cs)/np.array(cs_range), label = 'NJ', yerr=np.array(NJ_std)*i)
        plt.errorbar(cs_range,np.array(NN_cs)/np.array(cs_range), label = 'NN_{}'.format(points), yerr=np.array(NN_std)*i)
        plt.legend()
        if order == 'NLO':
            plt.title('{} '.format(order)+r'k-factor for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g')+' {}std error'.format(i))
        else:
            plt.title('{} '.format(order)+r'cross section for $e^+e^-\rightarrow\,q\bar{q}$'+'{}'.format(n_gluon*'g')+' {}std error'.format(i))
        plt.ylabel('cross section')
        plt.xlabel('number of phase space points')
        plt.xticks(np.arange(0,(len(test_momenta_array)/1000000)*1000001,1000000))
        plots.append(fig)
        plt.close()
        
    if save_dir != '':
        for idx, i in enumerate(plots):
            i.savefig(save_dir + '/cs_convergence_{}_{}.png'.format(test_points,idx), dpi = 250, bbox_inches='tight')
    return plots

def plot_tran_mom(test_momenta_array, NJ_test, y_pred, order, n_gluon, delta, points, save_dir = ''):
    '''
    Plot histrogram of transverse momenta of the leading energy jet
    
    :param test_momenta_array: array of all test moments
    :param NJ_test: numpy array of the NJet ground truth results
    :param y_pred: NN prediction
    :param order: LO or NLO
    :param n_gluon: number of gluons in the process
    :param save_dir: directory within which to save plot
    '''
    
    plot = []
    leading_energy = []
    for i in test_momenta_array:
        element = np.argmax(i[2:,0])
        leading_energy.append(i[2+element])
    
    transverse_mom = lambda mom: np.sqrt(mom[1]**2+mom[2]**2)
    
    tran_momenta = []
    for i in leading_energy:
        tran_momenta.append(transverse_mom(i))
    
    tran_momenta = np.array(tran_momenta)
    
    bin_range = np.arange(0,np.max(tran_momenta)+10,10)
    
    NJ_bin_vals = []
    NN_bin_vals = []
    for i in range(len(bin_range)-1):
        NJ_vals = NJ_test[np.where(np.logical_and(tran_momenta>bin_range[i], tran_momenta<=bin_range[i+1]))[0]]
        NJ_bin_vals.append(np.sum(NJ_vals))
        NN_vals = y_pred[np.where(np.logical_and(tran_momenta>bin_range[i], tran_momenta<=bin_range[i+1]))[0]]
        NN_bin_vals.append(np.sum(NN_vals))
        
    per_diff = (np.array(NJ_bin_vals)-np.array(NN_bin_vals))*100/np.array(NJ_bin_vals)
    
    diff = np.array(NJ_bin_vals)-np.array(NN_bin_vals)
    
    
    fig = plt.figure(1)
    
    ax1 = fig.add_axes((.1,.4,.8,.8))
    if order == 'NLO':
        plt.title('{} '.format(order)+r'differential k-factor against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
    else:
        plt.title('{} '.format(order)+r'differential cross section against $p_T$ for $e^+e^-\rightarrow\,q\bar{q}$'+'{} w/ FKS'.format(n_gluon*'g'))
    plt.bar(bin_range[1:],NN_bin_vals,width=8, color = 'orange', label = 'NN_{}'.format(points))
    plt.bar(bin_range[1:],NJ_bin_vals,width=8, color = 'blue', label = 'NJ', alpha = 0.5)
    plt.legend()
    plt.ylabel(r'$\frac{d\sigma}{dp_T}$', rotation = 0, fontsize = 10)
    ax1.set_xticklabels([])
    
    ax3=fig.add_axes((.1,.1,.8,.3))
    plt.plot(bin_range[1:],per_diff,'or')
    plt.xlabel('transverse momentum')
    plt.ylabel('% diff')
    plot.append(fig)
    plt.close()

    if save_dir != '':
        plot[0].savefig(save_dir + '/dcs_dpg.png'.format(order,n_gluon+2,delta,points), dpi = 250,bbox_inches='tight')
    return plot
