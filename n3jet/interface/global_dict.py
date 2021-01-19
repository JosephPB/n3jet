import pickle as pkl

global_dict = {
'NJET_BLHA': '/mt/home/jbullock/njet/njet-develop/blha/',
}

with open('./global_dict.pkl', 'wb') as f:
    pkl.dump(global_dict, f)
