from keras.models import Sequential
from keras.layers.merge import Add
from keras.layers.core import Flatten, Dense, Dropout, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D

import numpy as np

from keras.layers.normalization import BatchNormalization
from keras import backend as K

def inception_module(input_layer, activation='elu', n_towers = 3, pool_size = 1,
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
    tower = MaxPooling2D(kernel_size_1, strides=stride_size, padding='same')(input_layer)
    tower = Conv2D(depth[k+2], pool_size, padding='same', activation=activation)(tower_3)
    towers.append(tower)
    
    # towers = [tower_1, tower_2, tower_3]
    
    return keras.layers.concatenate(towers, axis = 3)

class MeCeptionNet:
    @staticmethod
    def build(width, height, depth, classes, 
                activation='elu', n_layers=1, depth0=64, 
                n_towers = 3, kernel_sizes = [3,5,1], 
                dropout_rate=0.5, pool_size=1,
                stride_size=2, use_bias=False, 
                n_skip_junc_gap=0):
        
        # could kernel_sizes == [5,3,1] instead? 
        #   Then kernel_sizes == [max_kernel, max_kernel-2, ..., max_kernel-2*n_towers]
        
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        chanDim = 2 # Tensorflow
        
        # monochromatic images: depth == 1
        inputShape = (height, width, depth)
        
        model = Input(shape = inputShape)
        
        """
        model = inception_module(model, activation='elu', 
                                  depth1 = 64, depth2 = 64, depth3 = 64,
                                  kernel_size_1 = 3, kernel_size_2 = 5, stride_size = 1)
        
        model = BatchNormalization(axis=chanDim)(model)
        """
        for k in range(n_layers):
            model = inception_module(model, activation='elu', 
                                      n_towers=n_towers, pool_size=pool_size,
                                      depths=[depth0]*(n_towers+1), # +1 for the spare Conv2D layer
                                      kernel_sizes = kernel_sizes)
            
            model = BatchNormalization(axis=chanDim)(model)
            
            if n_skip_junc_gap > 0 and k +1 % n_skip_junc_gap == 0:
                with BatchNormalization(axis=chanDim)(input_layer) as skip_layer:
                    # avoid layer confusion later by dissolving `skip_layer` automatically
                    model = Add()([model, skip_layer])
        
        model = AveragePooling2D(pool_size=(1,1), padding='valid')
        model = Dropout(rate= dropout_rate)(model)
        
        model = Flatten()(model)
        
        # return the constructed network architecture
        return Dense(classes, activation='softmax')(model)

if __name__ == '__main__':
    width, height, depth, classes = 10,10,3,2
    
    network = MeCeptionNet(width, height, depth, classes, 
                activation='elu', n_layers=5, depth0=32, 
                kernel_size=3, dropout_rate=[0.25,0.5], pool_size=2,
                stride_size=2, use_bias=False, zero_pad=False, 
                zero_pad_size=1)
    
    n_samples = 10
    input_img = np.random.normal(0,1, (n_samples, width, height, depth))
    
    model = Model(inputs = input_img, outputs = network)
