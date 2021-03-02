import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from matplotlib import rc
import time
import pickle

def bool_convert(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise ValueError("Value is neither True not False")

def file_exists(file_path):
    if os.path.exists(file_path) == True:
        pass
    else:
        raise ValueError('{} does not exist'.format(file_path))

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
