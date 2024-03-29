import numpy as np
np.random.seed(1337)
from keras.models import Sequential, model_from_json
from keras.optimizers import Adam
import json
import argparse
np.set_printoptions(threshold=np.inf)

class ModelDump:
    """
    Read in model architecture and weights and dump to .nnet format for C++ inference
    """

    def __init__(self, architecture, weights, output, verbose=False, init=False):
        """
        Parameters
        ----------

        architecture: path to architecture file
        weights: path to .h5 weights file
        output: path to output .nnet file
        verbose: (bool) if True print out stages
        """
        self.architecture = architecture
        self.weights = weights
        self.output = output
        self.verbose = verbose

        if init:
            self.load_model()
            self.write_output()

    def load_model(self):
        'Load model architecture and weights and compile'

        arch = open(self.architecture).read()
        self.model = model_from_json(arch)
        self.model.load_weights(self.weights)
        self.model.compile(
            optimizer = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False),
            loss = 'mean_squared_error'
        )
        self.arch = json.loads(arch)

    def write_output(self):
        'Write out architecture and weights in format for C++ inference'

        with open(self.output, 'w') as fout:
            fout.write('layers ' + str(len(self.model.layers)) + '\n')

            layers = []
            for ind, l in enumerate(self.arch["config"]["layers"]):
                if self.verbose:
                    print ind, l['class_name']
                fout.write('layer ' + str(ind) + ' ' + l['class_name'] + '\n')

                if self.verbose:
                    print str(ind), l['class_name']
                layers += [l['class_name']]

                if l['class_name'] == 'Activation':
                    fout.write(l['config']['activation'] + '\n')
                if l['class_name'] == 'MaxPooling2D':
                    fout.write(str(l['config']['pool_size'][0]) + ' ' + str(l['config']['pool_size'][1]) + '\n')
                if l['class_name'] == 'Dense':

                    # go through weight layers
                    W = self.model.layers[ind].get_weights()[0]
                    fout.write(str(W.shape[0]) + ' ' + str(W.shape[1]) + '\n')
                    for w in W:
                        fout.write('[')
                        for i in w:
                            fout.write(str(i) + ' ')
                        fout.write(']' + '\n')

                    # go through bias terms
                    W = self.model.layers[ind].get_weights()[1]
                    fout.write('[ ')
                    for i in W:
                        fout.write(str(i) + ' ')
                    fout.write(']' + '\n')

