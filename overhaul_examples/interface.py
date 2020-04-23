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
from tqdm import tqdm

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

# This could be made redundant by parsing the BLHA file to access the number of particles in the process
parser.add_argument(
    '--nlegs',
    dest='nlegs',
    help='number of external legs - only needed if not giving a mom file',
    type=int,
)

parser.add_argument(
    '--nps',
    dest='nps',
    help='number of phase-space points',
    type=int,
)

parser.add_argument(
    '--mom_file',
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

# Could read this in from the BLHA file
parser.add_argument(
    '--amp_type',
    dest='amp_type',
    help='amplitude type - set: tree, loop, loopsq (Note: 6-point diphoton uses loopsq)',
    type=str,
)

# Could read this in from the BLHA file
parser.add_argument(
    '--mur',
    dest='mur',
    help='renormalisation scale - currently this can not by dynamic, default: 91.188',
    type=float,
    default=91.188    
)
# Could read this in from the BLHA file
parser.add_argument(
    '--alpha',
    dest='alpha',
    help='alpha parameter value, default: 1. (Note: 6-point diphoton uses 1./137.035999084)',
    type=float,
    default=1.    
)

# Could read this in from the BLHA file
parser.add_argument(
    '--alphas',
    dest='alphas',
    help='alphas parameter value, default: 1. (Note: e+e- uses 0.07957747155, 6-point diphoton uses 0.118)',
    type=float,
    default=1.    
)

# Could read this in from the BLHA file
parser.add_argument(
    '--blha',
    dest='blha',
    help='Use BLHA1 or BLHA2, set 1 or 2 - affects EvalSubProcess type used, default: 2',
    type=int,
    default=2    
)

parser.add_argument(
    '--debug',
    dest='debug',
    help='in debug mode, value of first momenta value is printed out, set True or False, default: False',
    type=str,
    default='False'    
)

args = parser.parse_args()

order_file = args.order
contract_file = args.contract
proc_order = args.proc_order
nlegs = args.nlegs
nps = args.nps
mom_file = args.mom_file
generate_mom = args.generate_mom
nj_file = args.nj_file
generate_nj = args.generate_nj
amp_type = args.amp_type
mur = args.mur
alpha = args.alpha
alphas = args.alphas
blha = args.blha
debug = args.debug

def convert_bool(variable_name, variable):
    if variable == 'True':
        return True
    elif variable == 'False':
        return False
    else:
        raise ValueError('{} must either take on the value of True or False'.format(variable_name))

proc_order = convert_bool('proc_order', proc_order)

if order_file == 'None':
    order_file = None

if contract_file == 'None':
    contract_file = None

if nlegs == 'None':
    nlegs = None
    
if proc_order == False and contract == None:
    raise ValueError('in order for contract to be None, proc_order must be set to True')


generate_mom = convert_bool('generate_mom', generate_mom)

generate_nj = convert_bool('generate_nj', generate_nj)

debug = convert_bool('debug', debug)

######### ERROR CHECKING ############

# Check if njet file setting are correct - otherwise don't bother
if os.path.exists(nj_file) == False or generate_nj == True:
    pass
else:
    raise ValueError('Something does not look right with your momentum settings - check nj_file and generate_nj flags')

# Check BLHA set correctly
if blha == 1 or blha == 2:
    pass
else:
    raise ValueError('The --blha flag must either be 1 or 2')

# Check amplitude type
if amp_type == 'tree' or amp_type == 'loop' or amp_type == 'loopsq':
    pass
else:
    raise ValueError('The --amp_type flag must be one of: tree, loop or loopsq')

###########################  Up to here #################################
# TODO
# Work out some more arguments
# Parse AmplitudeType
# Parse process
# Work out corret output type
### Add options for:
######### full array
######### just finitie part (or tree)
######### drop epsilon
# Think about implementing accuracy
# Think about how much can be parsed from order and contract file
# Think about how to be dynamic in reading in the dictionary - maybe this could be processed once by this file instead as well(?)
# If we know the directories we are reading libraries from then we will know if we are printing out helicities or not and whether the file should be moved somewhere - this might even be the more pythonic thing to do

print (os.path.exists(order_file))
print (type(proc_order))

if os.path.exists(order_file) == True and proc_order == True:
    print ("Creating contract file")
    os.system("python contract.py --order {} --contract {}".format(order_file, contract_file))

print ( "  NJet: simple example of the BLHA interface")

## Initiailise NJet routines ##
olp = njet.OLP

status = olp.OLP_Start(contract_file)

if status == True:
    print ("OLP read in correctly")
else:
    print ("seems to be a problem with the contract file...")

## Momenta ##       
if os.path.exists(mom_file) == False or generate_mom == True:
    print ('Gennerating momenta')
    # Currently this is fixing the com at 10000 and the JADE delta at 0.01
    momenta = generate(nlegs-2,nps,1000.,0.01)
    np.save(mom_file, momenta, allow_pickle = True)
elif os.path.exists(mom_file) == True and generate_mom == False:
    print ('Loading momenta')
    momenta = np.load(mom_file, allow_pickle = True)
else:
    raise ValueError('Something does not look right with your momentum settings - check mom_file and generate_mom flags')
momenta = momenta.tolist()
print ('Momenta success')


#test = [[0.5000000000000000E+03,  0.0000000000000000E+00,  0.0000000000000000E+00,  0.5000000000000000E+03],
#      [0.5000000000000000E+03,  0.0000000000000000E+00,  0.0000000000000000E+00, -0.5000000000000000E+03],
#      [0.4999999999999998E+03,  0.1109242844438328E+03,  0.4448307894881214E+03, -0.1995529299308788E+03],
#      [0.5000000000000000E+03, -0.1109242844438328E+03, -0.4448307894881214E+03,  0.1995529299308787E+03]]

#test =  [[5.0000000000000000E+00,0.0000000000000000E+00,0.0000000000000000E+00,5.0000000000000000E+00],
#        [5.0000000000000000E+00,0.0000000000000000E+00,0.0000000000000000E+00,-5.0000000000000000E+00],
#        [1.1752772962487221E+00,4.2190326218557050E-01,-7.9632631758439487E-03,-1.0969097259457921E+00],
#        [3.8897413519622193E+00,7.2863133605774177E-02,-2.4238266256582408E+00,-3.0413554934726399E+00],
#        [1.4151041459885225E+00,-5.7523953762081992E-01,5.2094215883579842E-01,1.1833167308456300E+00],
#        [3.5198772058005368E+00,8.0473141829475237E-02,1.9108477299982864E+00,2.9549484885728017E+00]]


## NJet ##
rvals = []
if blha == 2:
    print ('Using BLHA2')
    olp.OLP_SetParameter(alpha=alpha,alphas=alphas,set_alpha=True)
    acc = 0
    if debug == True:
        rval = olp.OLP_EvalSubProcess2(1, momenta[0], mur, retlen=11, acc=acc)
        print ('Momenta: {}'.format(momenta[0]))
        print ('Values: {}'.format(rval))
        print ('Accuracy {}'.format(acc))
    else:
        for momentum in tqdm(momenta):
            rval = olp.OLP_EvalSubProcess2(1, momentum, mur, retlen=11, acc=acc)
            rvals.append(rval)
else:
    print ('Using BLHA1')
    if debug == True:
        rval = olp.OLP_EvalSubProcess(1, momenta[0], mur=mur, alpha=alpha, alphas=alphas, retlen=11)
        print ('Momenta: {}'.format(momenta[0]))
        print ('Values: {}'.format(rval))
    else:
        for momentum in tqdm(momenta):
            rval = olp.OLP_EvalSubProcess(1, momentum, mur=mur, alpha=alpha, alphas=alphas, retlen=11)
            rvals.append(rval)
print ('NJet successully run and points generated')

## Saving NJet ##
if amp_type == 'tree':
    if debug == True:
        print ("Tree    : {}".format(rval[0]))
    else:
        print ("Formatting and cleaning output")
        output = []
        for i in tqdm(rvals):
            output.append(i[0])
        np.save(nj_file, output, allow_pickle=True)
elif amp_type == 'loop':
    if debug == True:
        print ("Tree    : {}".format(rval[3]))
        print ("Loop(-2): {}".format(rval[0]))
        print ("Loop(-1): {}".format(rval[1]))
        print ("Loop(0) : {}".format(rval[2]))
    else:
        print ("Formatting and cleaning output")
        output = []
        for i in tqdm(rvals):
            to_add = []
            # Currently 3rd entry is the accuracy which we are not recording if debug = False
            to_add.append(['A0',   rval[3]])
            to_add.append(['A1_2', rval[0]])
            to_add.append(['A1_1', rval[1]])
            to_add.append(['A1_0', rval[2]])
            output.append(to_append)
            np.save(nj_file, output, allow_pickle=True)
elif amp_type == 'loopsq':
    if debug == True:
        print ("Loop(-4): {}".format(rval[0]))
        print ("Loop(-3): {}".format(rval[1]))
        print ("Loop(-2): {}".format(rval[2]))
        print ("Loop(-1): {}".format(rval[3]))
        print ("Loop(0) : {}".format(rval[4]))
    else:
        print ("Formatting and cleaning output")
        output = []
        for i in tqdm(rvals):
            to_add = []
            # Currently 3rd entry is the accuracy which we are not recording if debug = False
            to_add.append(['A1_4', rval[0]])
            to_add.append(['A1_3', rval[1]])
            to_add.append(['A1_2', rval[2]])
            to_add.append(['A1_1', rval[3]])
            to_add.append(['A1_0', rval[4]])
            output.append(to_append)
            np.save(nj_file, output, allow_pickle=True)
        



