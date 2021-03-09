import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler

import tensorflow as tf
from tensorflow.keras import activations

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.wrappers.scikit_learn import KerasRegressor
from keras import metrics
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import keras.backend as K

class Model:
    
    def __init__(
            self,
            input_size,
            momenta,
            labels,
            all_jets = False,
            all_legs = False,
            model_dataset = False,
            high_precision = False
    ):
        '''
        :param input_size: the flattened input dim for the model
            e.g. 3 jets has input_dim of (3-1)*4=8
        :param momenta: input momenta in NJET format (i.e. [num points, num jets, 4])
        :param labels: labels
        '''
        self.input_size = input_size
        self.momenta = momenta
        self.labels = labels
        self.all_jets = all_jets
        self.all_legs = all_legs
        self.model_dataset = model_dataset
        self.high_precision = high_precision

        if self.high_precision:
            K.set_floatx('float64')
    
    def standardise(self, data):
        '''standardise data
        :param data: an array over which to standardise (this array may be a variable column) 
        '''
        array = np.array(data)
        mean = np.mean(array)
        std = np.std(array)
        standard = (array-mean)/(std)
        return mean, std, standard

    def normalise(self, data):
        array = np.array(data)
        minimum = np.min(array)
        maximum = np.max(array)
        norm = (array-minimum)/(maximum-minimum)
        return minimum, maximum, norm
        
    def root_mean_squared_error(self, y_true, y_pred):
        'custom loss function RMSE'
        return K.sqrt(K.mean(K.square(y_pred - y_true)))
    
    def process_training_data(self, random_state=42, scaling='standardise', **kwargs):
        '''
        training data must be standardised and split for training and validation
        **kwargs can take on:
        :param moms: the PS points in format [no_PS_points, points, 4]
        :param labs: ground truth labels of squared matrix elements
        '''
        moms = kwargs.get('moms', self.momenta)
        labs = kwargs.get('labs', self.labels)

        if self.all_legs == True:
            momenta = np.array(moms)
        elif self.all_jets == True:
            momenta = np.array(moms)[:,2:,:] #include all outgoing jets
        else:
            momenta = np.array(moms)[:,3:,:] #pick out all but one jet
            
        labels = np.array(labs)
        
        x_standard = momenta.reshape(-1,4).copy() #shape for standardising each momentum element
        self.x_mean = np.zeros(4)
        self.x_std = np.zeros(4)

        if scaling == 'standardise':
        
            self.x_mean[0],self.x_std[0],x_standard[:,0] = self.standardise(momenta.reshape(-1,4)[:,0])
            self.x_mean[1],self.x_std[1],x_standard[:,1] = self.standardise(momenta.reshape(-1,4)[:,1])
            self.x_mean[2],self.x_std[2],x_standard[:,2] = self.standardise(momenta.reshape(-1,4)[:,2])
            self.x_mean[3],self.x_std[3],x_standard[:,3] = self.standardise(momenta.reshape(-1,4)[:,3])

            self.y_mean, self.y_std, y_standard = self.standardise(labels)

        elif scaling == 'normalise':

            self.x_mean[0],self.x_std[0],x_standard[:,0] = self.normalise(momenta.reshape(-1,4)[:,0])
            self.x_mean[1],self.x_std[1],x_standard[:,1] = self.normalise(momenta.reshape(-1,4)[:,1])
            self.x_mean[2],self.x_std[2],x_standard[:,2] = self.normalise(momenta.reshape(-1,4)[:,2])
            self.x_mean[3],self.x_std[3],x_standard[:,3] = self.normalise(momenta.reshape(-1,4)[:,3])

            self.y_mean, self.y_std, y_standard = self.normalise(labels)
            
        else:
            raise ValueError('scaling must being either normalise or standardise and you have used {}'.format(scaling))
        
        x_standard = x_standard.reshape(-1,self.input_size) #shape for passing into network

        if self.model_dataset:
            X_train, X_test, y_train, y_test = train_test_split(x_standard, y_standard, test_size=0.2)
        else:
            X_train, X_test, y_train, y_test = train_test_split(x_standard, y_standard, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test, self.x_mean, self.x_std, self.y_mean, self.y_std   
    
    def baseline_model_dataset(self, layers, lr=0.001, activation='tanh', loss='mean_squared_error'):
        'define and compile model with fixing weight initialisers and a random dataset'
        # create model
        # at some point can use new Keras tuning feature for optimising this model

        seeds = [
            1337,
            1337+123,
            1337+345,
            1337+545,
            1337-123,
            1337-345,
            1337+567,
            1337-567,
            1337-189,
            1337+189,
            1337+194,
            1337-194,
            1337-347,
            1337+347,
            1337-545
        ]

        if len(layers) > len(seeds)-1:
            raise Exception(
                'the number of layers cannot be more than {}, you have defined {} layers'.format(
                    len(seeds)-1, len(layers)
                )
            )
        
        model = Sequential()
        model.add(Dense(layers[0], input_dim=(self.input_size), kernel_initializer = glorot_uniform(seed=seeds[0])))
        if activation == 'tanh':
            model.add(Activation(activations.tanh))
        elif activation == 'relu':
            model.add(Activation(activations.relu))
        else:
            raise ValueError('activation supported are either tanh or relu, you have used {}'.format(activation))

        for i in range(1,len(layers)):
            model.add(Dense(layers[i], kernel_initializer = glorot_uniform(seed=seeds[i])))
            if activation == 'tanh':
                model.add(Activation(activations.tanh))
            elif activation == 'relu':
                model.add(Activation(activations.relu))

        model.add(Dense(1, kernel_initializer = glorot_uniform(seed=seeds[-1])))
        # Compile model
        model.compile(optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False), loss = loss)
        
        return model

    def baseline_model(self, layers, lr=0.001, activation='tanh', loss='mean_squared_error'):
        'define and compile model with a fixed dataset but random weights'
        # create model
        # at some point can use new Keras tuning feature for optimising this model
        model = Sequential()
        model.add(Dense(layers[0], input_dim=(self.input_size)))
        if activation == 'tanh':
            model.add(Activation(activations.tanh))
        elif activation == 'relu':
            model.add(Activation(activations.relu))
        else:
            raise ValueError('activation supported are either tanh or relu, you have used {}'.format(activation))
        
        for i in range(1, len(layers)):
            model.add(Dense(layers[i]))
            if activation == 'tanh':
                model.add(Activation(activations.tanh))
            elif activation == 'relu':
                model.add(Activation(activations.relu))

        model.add(Dense(1))
        # Compile model
        model.compile(optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, amsgrad=False), loss = loss)
        
        return model

    
    def fit(
            self,
            scaling='standardise',
            layers=[32,16,8],
            epochs=10000,
            lr=0.001,
            activation='tanh',
            loss='mean_squared_error',
            **kwargs
    ):
        '''
        fit model
        :param layers: an array of lengeth 3 providing the number of hidden nodes in the three layers
        '''

        if activation == 'relu' and scaling !='normalise':
            raise ValueError('if activation is set to relu then scaling must be normalised')
        
        random_state = kwargs.get('random_state', 42)
        
        X_train, X_test, y_train, y_test,_,_,_,_ = self.process_training_data(random_state=random_state, scaling=scaling)
        print ('The training dataset has size {}'.format(X_train.shape))

        if self.model_dataset:
            self.model = self.baseline_model_dataset(layers=layers, lr=lr, activation=activation, loss=loss)
        else:
            self.model = self.baseline_model(layers=layers, lr=lr, activation=activation, loss=loss)
        ES = EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, restore_best_weights=True)

        if self.model_dataset:
            self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[ES], batch_size=512, shuffle=False)
        else:
            self.model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test), callbacks=[ES], batch_size=512)
        
        return self.model, self.x_mean, self.x_std, self.y_mean, self.y_std
        
    def standardise_test(self, data, mean, std):
        array = np.array(data)
        standard = (array-mean)/(std)
        return standard

    def normalise_test(self, data, minimum, maximum):
        array = np.array(data)
        norm = (array-minimum)/(maximum-minimum)
        return norm
        
    def process_testing_data(self, moms, scaling='standardise', **kwargs):
        '''
        **kwargs can take on:
        :param x_mean, x_std, y_mean, y_std: mean and std of x and y values if 
        not (properly) provided by class e.g. if using a pretrained model with 
        known mean and std
        '''
            
        labs = kwargs.get('labs', None)
        if self.all_legs == True:
            momenta = np.array(moms)
        elif self.all_jets == True:
            momenta = np.array(moms)[:,2:,:] #include all outgoing jets
        else:
            momenta = np.array(moms)[:,3:,:] #pick out all but one jet

        y_mean = kwargs.get('y_mean', self.y_mean)
        
        print_y = kwargs.get('print_y', True)
        if print_y == True:
            print ('Using y_mean of {} instead of {}'.format(y_mean, self.y_mean))
        y_std = kwargs.get('y_std', self.y_std)
        x_mean = kwargs.get('x_mean', self.x_mean)
        x_std = kwargs.get('x_std', self.x_std)
        
        if labs is not None:
            labels = np.array(labs)
        
        x_standard = momenta.reshape(-1,4).copy() #shape for standardising each momentum element

        if scaling == 'standardise':
        
            x_standard[:,0] = self.standardise_test(momenta.reshape(-1,4)[:,0],x_mean[0],x_std[0])
            x_standard[:,1] = self.standardise_test(momenta.reshape(-1,4)[:,1],x_mean[1],x_std[1])
            x_standard[:,2] = self.standardise_test(momenta.reshape(-1,4)[:,2],x_mean[2],x_std[2])
            x_standard[:,3] = self.standardise_test(momenta.reshape(-1,4)[:,3],x_mean[3],x_std[3])
            
            x_standard = x_standard.reshape(-1,self.input_size) #shape for passing into network
        
            if labs is not None:
                y_standard = self.standardise_test(labels,y_mean,y_std)
                return x_standard, y_standard
            else:
                return x_standard
    

            
        elif scaling == 'normalise':

            x_standard[:,0] = self.normalise_test(momenta.reshape(-1,4)[:,0],x_mean[0],x_std[0])
            x_standard[:,1] = self.normalise_test(momenta.reshape(-1,4)[:,1],x_mean[1],x_std[1])
            x_standard[:,2] = self.normalise_test(momenta.reshape(-1,4)[:,2],x_mean[2],x_std[2])
            x_standard[:,3] = self.normalise_test(momenta.reshape(-1,4)[:,3],x_mean[3],x_std[3])
            
            x_standard = x_standard.reshape(-1,self.input_size) #shape for passing into network
        
            if labs is not None:
                y_standard = self.standardise_test(labels,y_mean,y_std)
                return x_standard, y_standard
            else:
                return x_standard

        else:
            raise ValueError('scaling must being either normalise or standardise and you have used {}'.format(scaling))
    
    def destandardise(self, data, mean, std):
        'destandardise array for inference and comparison'
        array = np.array(data)
        return (array*std) + mean

    def denormalise(seld, data, minimum, maximum):
        array = np.array(data)
        return array*(maximum-minimum) + minimum
    
    def destandardise_data(self, y_pred, x_pred=None, scaling='standardise', **kwargs):
        '''
        destandardise any standardised data
        :param y_pred: squared matrix element values
        :param x_pred: optional parameter of momenta values to be destandardised
        **kwargs can take on:
        :param x_mean, x_std, y_mean, y_std: mean and std of x and y values if not (properly) provided by class e.g. if using a pretrained model with known mean and std
        
        note: when initialising the class with the data used to train a pretrained model, the standardised data will be the same as used in training if the dataset is loaded and passed correctly as the mean and std is independent of the data splitting
        '''
        
        y_mean = kwargs.get('y_mean', self.y_mean)
        y_std = kwargs.get('y_std', self.y_std)
        x_mean = kwargs.get('x_mean', self.x_mean)
        x_std = kwargs.get('x_std', self.x_std)
        
        if scaling == 'standardise':
            y_destandard = self.destandardise(y_pred,y_mean,y_std)
        
            if x_pred is not None:
                x_pred = x_pred.reshape(-1,4)
                x_destandard = x_pred.copy()
                
                x_destandard[:,0] = self.destandardise(x_pred[:,0],x_mean[0],x_std[0])
                x_destandard[:,1] = self.destandardise(x_pred[:,1],x_mean[1],x_std[1])
                x_destandard[:,2] = self.destandardise(x_pred[:,2],x_mean[2],x_std[2])
                x_destandard[:,3] = self.destandardise(x_pred[:,3],x_mean[3],x_std[3])
            
                x_destandard = x_destandard.reshape(-1,int((self.input_size)/4),4)
            
                return x_destandard, y_destandard
        
            else:
                return y_destandard
        
        elif scaling == 'normalise':
            y_destandard = self.denormalise(y_pred,y_mean,y_std)
        
            if x_pred is not None:
                x_pred = x_pred.reshape(-1,4)
                x_destandard = x_pred.copy()
                
                x_destandard[:,0] = self.denormalise(x_pred[:,0],x_mean[0],x_std[0])
                x_destandard[:,1] = self.denormalise(x_pred[:,1],x_mean[1],x_std[1])
                x_destandard[:,2] = self.denormalise(x_pred[:,2],x_mean[2],x_std[2])
                x_destandard[:,3] = self.denormalise(x_pred[:,3],x_mean[3],x_std[3])
            
                x_destandard = x_destandard.reshape(-1,int((self.input_size)/4),4)
            
                return x_destandard, y_destandard
        
            else:
                return y_destandard
        
        else:
            raise ValueError('scaling must being either normalise or standardise and you have used {}'.format(scaling))
        
        
        
        
       
    
