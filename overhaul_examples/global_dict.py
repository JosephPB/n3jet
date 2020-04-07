import pickle as pkl

global_dict = {
'NJET_BLHA': '/mt/home/jbullock/njet/njet-develop/blha/',
'N3JET_BASE': '/mt/home/jbullock/n3jet/'
}

with open('./global_dict.pkl', 'wb') as f:
    pkl.dump(global_dict, f)
