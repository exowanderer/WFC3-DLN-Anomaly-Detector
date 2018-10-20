import numpy as np

from keras.models import Sequential
from keras.layers.merge import Add
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers import Input, concatenate as layers_concatenate

from keras.layers.normalization import BatchNormalization
from keras import backend as K

def inception_module( input_layer, activation='elu', n_towers = 3, 
                      pool_size = 1, stride_size=1,
                      depths=[64,64,64,64], kernel_sizes = [3,5,1]):
    
    depthSpare = depths[0]
    ones_kernel = (1,1)
    kernels_size = []
    kernels_size = [(ksize, ksize) for ksize in kernel_sizes]
    
    stride_sizes = (stride_size, stride_size)
    
    towers = [Conv2D(depthSpare, ones_kernel, padding='same', activation=activation)(input_layer)]
    
    for k in range(n_towers-1):
        tower = Conv2D(depth[k+1], ones_kernel, padding='same', activation=activation)(input_layer)
        tower = Conv2D(depth[k+1], kernels_size[k], padding='same', activation=activation)(tower)
        towers.append(tower)
    
    """
    tower_1 = Conv2D(depth1, ones_kernel, padding='same', activation=activation)(input_layer)
    tower_1 = Conv2D(depth1, kernel_size_1, padding='same', activation=activation)(tower_1)
    
    tower_2 = Conv2D(depth2, ones_kernel, padding='same', activation=activation)(input_layer)
    tower_2 = Conv2D(depth2, kernel_size_2, padding='same', activation=activation)(tower_2)
    """
    
    # I assume that inception requires at least on MaxPool layer(?)
    tower = MaxPooling2D(kernels_size[-1], strides=stride_sizes, padding='same')(input_layer)
    tower = Conv2D(depths[-1], pool_size, padding='same', activation=activation)(tower)
    towers.append(tower)
    
    # towers = [tower_1, tower_2, tower_3]
    
    return layers_concatenate(towers, axis = 3)

class MynCeptionNet:
    @staticmethod
    # def build(width, height, depth, classes,
    def build(input_layer, classes, 
                activation='elu', n_layers=1, depth0=64, 
                n_towers = 3, kernel_sizes = [3,5,1], 
                dropout_rate=0.5, pool_size=1,
                stride_size=1, use_bias=False, 
                n_skip_junc_gap=0):
        
        # could kernel_sizes == [5,3,1] instead? 
        #   Then kernel_sizes == [max_kernel, max_kernel-2, ..., max_kernel-2*n_towers]
        
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        chanDim = 2 # Tensorflow
        
        network = input_layer
        for k in range(n_layers):
            network = inception_module(network, activation='elu', 
                                      n_towers=n_towers, pool_size=pool_size,
                                      depths=[depth0]*(n_towers+1), # +1 for the spare Conv2D layer
                                      kernel_sizes = kernel_sizes)
            
            network = BatchNormalization(axis=chanDim)(network)
            
            if n_skip_junc_gap > 0 and k > 0 and k + 1 % n_skip_junc_gap == 0:
                with BatchNormalization(axis=chanDim)(input_layer) as skip_layer:
                    # avoid layer confusion later by dissolving `skip_layer` automatically
                    network = Add()([network, skip_layer])
        
        network = AveragePooling2D(pool_size=(1,1), padding='valid')(network)
        network = Dropout(rate= dropout_rate)(network)
        
        network = Flatten()(network)
        
        # return the constructed network architecture
        return Dense(classes, activation='softmax')(network)

if __name__ == '__main__':
    width, height, depth, classes = 10,10,3,2
    
    input_layer = Input(shape=(width, height, depth))
    network = MeCeptionNet(input_layer, classes, # width, height, depth
                activation='elu', n_layers=5, depth0=32, 
                kernel_size=3, dropout_rate=[0.25,0.5], pool_size=2,
                stride_size=2, use_bias=False, zero_pad=False, 
                zero_pad_size=1)
    
    model = Model(inputs = input_layer, outputs = network)
