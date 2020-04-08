import pickle as pkl

with open('./global_dict.pkl', 'rb') as f:
    global_dict = pkl.load(f)

import os
import sys
sys.path.append(global_dict['NJET_BLHA'])
sys.path.append(global_dict['N3JET_BASE'] + '/phase/')
sys.path.append(global_dict['N3JET_BASE'] + '/utils/')

#from ctypes import c_int, c_double, c_char_p, POINTER

import numpy as np
from njet_run_functions import *

contract_file = 'OLE_contract_diphoton.lh'
mom_data_dir = './'
mom_file = 'ex_diphoton_mom'

njet_data_dir = './'
njet_file = 'ex_diphoton_njet'


print ( "  NJet: simple example of the BLHA interface")

olp = njet.OLP()

status = njet_init(contract_file)

if status == True:
    print ("OLP read in correctly")
else:
    print ("seems to be a problem with the contract file...")

test =  [[5.0000000000000000E+00,0.0000000000000000E+00,0.0000000000000000E+00,5.0000000000000000E+00],
        [5.0000000000000000E+00,0.0000000000000000E+00,0.0000000000000000E+00,-5.0000000000000000E+00],
        [1.1752772962487221E+00,4.2190326218557050E-01,-7.9632631758439487E-03,-1.0969097259457921E+00],
        [3.8897413519622193E+00,7.2863133605774177E-02,-2.4238266256582408E+00,-3.0413554934726399E+00],
        [1.4151041459885225E+00,-5.7523953762081992E-01,5.2094215883579842E-01,1.1833167308456300E+00],
        [3.5198772058005368E+00,8.0473141829475237E-02,1.9108477299982864E+00,2.9549484885728017E+00]]

mur = 91.188
alphas = 0.118
alpha = 1./137.035999084

rval = olp.OLP_EvalSubProcess(1, test, mur=mur,retlen=11)

print (rval)
