import pickle as pkl

with open('./global_dict.pkl', 'rb') as f:
    global_dict = pkl.load(f)

import os
import sys
sys.path.append(global_dict['NJET_BLHA'])
sys.path.append(global_dict['N3JET_BASE'] + '/phase/')
sys.path.append(global_dict['N3JET_BASE'] + '/utils/')

import numpy as np
import argparse

from rambo_while import generate
import njet

parser = argparse.ArgumentParser(description='NJet Python interface for single process BLHA files')

parser.add_argument(
    '--order',
    dest='order',
    help='name of order file, default: None',
    type=str,
    default='None'
)

parser.add_argument(
    '--contract',
    dest='contract',
    help='name of contract file, default: None',
    type=str,
    default='None'
)

parser.add_argument(
    '--proc_order',
    dest='proc_order',
    help='process order file to create contract file and save in same directory as order file, default: False',
    type=str,
    default='False'
)

parser.add_argument(
    '--mom',
    dest='mom_file',
    help='destination of momenta file',
    type=str,
)

parser.add_argument(
    '--generate_mom',
    dest='generate_mom',
    help='generate momenta file even if it already exists, default: False',
    type=str,
    default='False'
)

parser.add_argument(
    '--nj_file',
    dest='nj_file',
    help='NJet file',
    type=str,
)

parser.add_argument(
    '--generate_nj',
    dest='generate_nj',
    help='generate NJet file even if it already exists, default: False',
    type=str,
    default='False'    
)

args = parser.parse_args()

order_file = args.order
contract_file = args.contract
proc_order = args.proc_order
mom_file = args.mom_file
generate_mom = args.generate_mom
nj_file = args.nj_file
generate_nj = args.generate_nj

def convert_bool(variable_name, variable):
    if variable == 'True':
        variable = True
    elif variable == 'False':
        variable = False
    else:
        raise ValueError('{} must either take on the value of True or False'.format(variable_name))

convert_bool('proc_order', proc_order)

if order_file == 'None':
    order_file = None

if contract_file == 'None':
    contract_file = None

if proc_order == False and contract == None:
    raise ValueError('in order for contract to be None, proc_order must be set to True')

convert_bool('generate_mom', generate_mom)

convert_bool('generate_nj', generate_nj)


###########################  Up tp here #################################
# TODO
# Work out some more arguments
# Parse AmplitudeType
# Parse process
# Add BLHA 1 or 2 flag
# Work out corret output type
### Add options for:
######### full array
######### just finitie part (or tree)
######### drop epsilon
# Think about implementing accuracy
# Think about how much can be parsed from order and contract file
# Think about how to be dynamic in reading in the dictionary - maybe this could be processed once by this file instead as well(?)
# If we know the directories we are reading libraries from then we will know if we are printing out helicities or not and whether the file should be moved somewhere - this might even be the more pythonic thing to do

contract_file = 'OLE_contract_2j.lh'
mom_data_dir = './'
mom_file = 'ex_diphoton_mom'

njet_data_dir = './'
njet_file = 'ex_diphoton_njet'


print ( "  NJet: simple example of the BLHA interface")

olp = njet.OLP

status = olp.OLP_Start(contract_file)

if status == True:
    print ("OLP read in correctly")
else:
    print ("seems to be a problem with the contract file...")


momenta = generate(4,100,1000.,0.01)
momenta = momenta.tolist()

#test = momenta[0]

test = [[0.5000000000000000E+03,  0.0000000000000000E+00,  0.0000000000000000E+00,  0.5000000000000000E+03],
      [0.5000000000000000E+03,  0.0000000000000000E+00,  0.0000000000000000E+00, -0.5000000000000000E+03],
      [0.4999999999999998E+03,  0.1109242844438328E+03,  0.4448307894881214E+03, -0.1995529299308788E+03],
      [0.5000000000000000E+03, -0.1109242844438328E+03, -0.4448307894881214E+03,  0.1995529299308787E+03]]

#test =  [[5.0000000000000000E+00,0.0000000000000000E+00,0.0000000000000000E+00,5.0000000000000000E+00],
#        [5.0000000000000000E+00,0.0000000000000000E+00,0.0000000000000000E+00,-5.0000000000000000E+00],
#        [1.1752772962487221E+00,4.2190326218557050E-01,-7.9632631758439487E-03,-1.0969097259457921E+00],
#        [3.8897413519622193E+00,7.2863133605774177E-02,-2.4238266256582408E+00,-3.0413554934726399E+00],
#        [1.4151041459885225E+00,-5.7523953762081992E-01,5.2094215883579842E-01,1.1833167308456300E+00],
#        [3.5198772058005368E+00,8.0473141829475237E-02,1.9108477299982864E+00,2.9549484885728017E+00]]

mur = 91.188
alphas = 0.118
#alpha = 1./137.035999084
alpha = 1.

olp.OLP_SetParameter(alpha=alpha,alphas=alphas,set_alpha=True)

acc = 0
rval = olp.OLP_EvalSubProcess2(1, test, mur, retlen=11, acc=acc)

#rval = olp.OLP_EvalSubProcess(1, test, mur=mur, alpha=alpha, alphas=alphas, retlen=11)

print (rval)
