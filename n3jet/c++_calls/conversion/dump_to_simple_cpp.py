import numpy as np
np.random.seed(1337)
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
import json
import argparse

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=
                                 """
                                 This is a simple script to dump Keras model
                                 into simple format suitable for porting into pure C++ model
                                 """
                                   
)

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-o', '--output', help="Ouput file name", required=True)
parser.add_argument('-v', '--verbose', help="Verbose", required=False)
args = parser.parse_args()

print ('Read architecture from {}'.format(args.architecture))
print ('Read weights from {}'.format(args.weights))
print ('Writing to {}'.format(args.output))

arch = open(args.architecture).read()
model = model_from_json(arch)
model.load_weights(args.weights)
model.compile(optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False), loss = 'mean_squared_error')
arch = json.loads(arch)

with open(args.output, 'w') as fout:
    fout.write('layers ' + str(len(model.layers)) + '\n')

    layers = []
    for ind, l in enumerate(arch["config"]["layers"]):
        if args.verbose:
            print (ind, l['class_name'])
        fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')

        if args.verbose:
            print (str(ind), l['class_name'])
        layers += [l['class_name']]

        if l['class_name'] == 'Activation':
            fout.write(l['config']['activation'] + '\n')
        if l['class_name'] == 'MaxPooling2D':
            fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
        if l['class_name'] == 'Dense':

            # go through weight layers
            W = model.layers[ind].get_weights()[0]
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
            for w in W:
                fout.write('[')
                for i in w:
                    fout.write(str(i) + ' ')
                fout.write(']' + '\n')
                
            # go through bias terms
            W = model.layers[ind].get_weights()[1]
            fout.write('[')
            for i in W:
                fout.write(str(i) + ' ')
            fout.write(']' + '\n')

            # W = model.layers[ind].get_weights()[0]

            # if args.verbose:
            #     print (W.shape)
            # fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')

            # for w in W:
            #     fout.write(str(w) + '\n')
            # fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
