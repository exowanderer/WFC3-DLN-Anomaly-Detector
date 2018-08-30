from keras.models import Sequential
from keras.layers.merge import Add
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D. AveragePooling2D
from keras.optimizers import SGD

import cv2, numpy as np

from keras.layers.normalization import BatchNormalization
from keras import backend as K

def inception_module(input_layer, activation='elu', 
                      depth1 = 64, depth2 = 64, depth3 = 64,
                      kernel_size_1 = 3, kernel_size_2 = 5, stride_size = 1):
    
    if depth1 != depth2 and depth1 != depth3:
        print('[WARNING] Inception technically is defined as depth1 = depth2 = depth3 = 64')
    
    kernel_size_0 = (1,1)
    kernel_size_1 = (kernel_size_1, kernel_size_1)
    kernel_size_2 = (kernel_size_2, kernel_size_2)
    stride_size = (stride_size,stride_size)
    
    tower_1 = Conv2D(depth1, kernel_size_0, padding='same', activation=activation)(input_layer)
    tower_1 = Conv2D(depth1, kernel_size_1, padding='same', activation=activation)(tower_1)
    
    tower_2 = Conv2D(depth2, kernel_size_0, padding='same', activation=activation)(input_layer)
    tower_2 = Conv2D(depth2, kernel_size_2, padding='same', activation=activation)(tower_2)
    
    tower_3 = MaxPooling2D(kernel_size_1, strides=stride_size, padding='same')(input_layer)
    tower_3 = Conv2D(depth3, kernel_size_0, padding='same', activation=activation)(tower_3)
    
    return keras.layers.concatenate([tower_1, tower_2, tower_3], axis = 3)

class MeCeptionNet:
    @staticmethod
    def build(width, height, depth, classes, 
                activation='elu', n_layers=1, depth0=32, 
                kernel_size=3, dropout_rate=0.5, pool_size=2,
                stride_size=2, use_bias=False, zero_pad=False, 
                zero_pad_size=1, n_skip_junc_gap=0):
        
        # VGG Default for depth0 is 64 (I'm making it smaller)
        if isinstance(dropout_rate, float):
            dropout_rate = [dropout_rate, dropout_rate]
        
        if not isinstance(dropout_rate, (list, tuple, np.ndarray)):
            raise ValueError('`dropout_rate` must be either a float, or a 2-element list/tuple/array')
        
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        inputShape = (height, width, depth)
        
        chanDim = 2 # Tensorflow
        
        from keras.layers import Input
        input_img = Input(shape = inputShape)
        
        # CONV => eLU => POOL
        model = inception_module(input_layer, activation='elu', 
                                  depth1 = 64, depth2 = 64, depth3 = 64,
                                  kernel_size_1 = 3, kernel_size_2 = 5, stride_size = 1)
        
        model = BatchNormalization(axis=chanDim)(model)
        
        for k in range(1, n_layers):
            model = inception_module(model, activation='elu', 
                                      depth1 = 64, depth2 = 64, depth3 = 64,
                                      kernel_size_1 = 3, kernel_size_2 = 5, stride_size = 1)
            
            model = BatchNormalization(axis=chanDim)(model)
            
            if n_skip_junc_gap > 0 and k % n_skip_junc_gap == 0:
                jump_layer = BatchNormalization(axis=chanDim)(input_layer)
                model = Add()([model, jump_layer])
        
        model = AveragePooling2D(pool_size=(1,1), padding='valid')
        model = Dropout(rate= dropout_rate)(model)
        
        model = Flatten()(model)
        
        # return the constructed network architecture
        reutrn Dense(classes, activation='softmax')(model)

if __name__ == '__main__':
    width, height, depth, classes = 10,10,3,2
    
    network = MeCeptionNet(width, height, depth, classes, 
                activation='elu', n_layers=5, depth0=32, 
                kernel_size=3, dropout_rate=[0.25,0.5], pool_size=2,
                stride_size=2, use_bias=False, zero_pad=False, 
                zero_pad_size=1)
    
    n_samples = 10
    input_img = np.random.normal(0,1, (n_samples, width, height, depth)
    
    Model(inputs = input_img, outputs = network)
