import os
import pickle as pkl
import argparse

parser = argparse.ArgumentParser(description='Run njet/blha/njet.py to produce contract file')

parser.add_argument(
    '--order',
    dest='order',
    help='name of order file',
    type=str
)

parser.add_argument(
    '--contract',
    dest='contract',
    help='name of contract file',
    type=str
)

args = parser.parse_args()

order_file = args.order
contract_file = args.contract

with open('./global_dict.pkl', 'rb') as f:
    global_dict = pkl.load(f)

    
blha_dir = global_dict['NJET_BLHA']
print ('Looking in {}'.format(blha_dir))

os.system("python {} -o {} {}".format(blha_dir + '/njet.py', contract_file, order_file))

print ("Created contract file")