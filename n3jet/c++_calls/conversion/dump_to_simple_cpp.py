import numpy as np
np.random.seed(1337)
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
import json
import argparse

np.set_printoptions(threshold=np.inf)
parser = argparse.ArgumentParser(description=
                                 'This is a simple script to dump Keras model
                                   into simple format suitable for porting into pure C++ model'
)

parser.add_argument('-a', '--architecture', help="JSON with model architecture", required=True)
parser.add_argument('-w', '--weights', help="Model weights in HDF5 format", required=True)
parser.add_argument('-o', '--output', help="Ouput file name", required=True)
parser.add_argument('-v', '--verbose', help="Verbose", required=False)
args = parser.parse_args()

print 'Read architecture from', args.architecture
print 'Read weights from', args.weights
print 'Writing to', args.output

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
            print ind, l['class_name']
        fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')

        if args.verbose:
            print str(ind), l['class_name']
        layers += [l['class_name']]
        if l['class_name'] == 'Convolution2D':
            W = model.layers[ind].get_weights()[0]
            if args.verbose:
                print W.shape
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + ' ' + str(W.shape[2]) + ' ' + str(W.shape[3]) + ' ' + l['config']['border_mode'] + '\n')

            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    for k in range(W.shape[2]):
                        fout.write(str(W[i,j,k]) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')

        if l['class_name'] == 'Activation':
            fout.write(l['config']['activation'] + '\n')
        if l['class_name'] == 'MaxPooling2D':
            fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
        if l['class_name'] == 'Dense':
            W = model.layers[ind].get_weights()[0]

            if args.verbose:
                print W.shape
            fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')

            for w in W:
                fout.write(str(w) + '\n')
            fout.write(str(model.layers[ind].get_weights()[1]) + '\n')
