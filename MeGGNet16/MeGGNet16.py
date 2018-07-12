from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

from keras.layers.normalization import BatchNormalization
from keras import backend as K

class MeGGNet16:
    @staticmethod
    def build(width, height, depth, classes, 
                activation='elu', n_layers=5, depth0=32, 
                kernel_size=3, dropout_rate=[0.25,0.5], pool_size=2,
                stride_size=2, use_bias=False, zero_pad=False, 
                zero_pad_size=1):
        
        # VGG Default for depth0 is 64 (I'm making it smaller)
        if isinstance(dropout_rate, float):
            dropout_rate = [dropout_rate, dropout_rate]
        
        if not isinstance(dropout_rate, (list, tuple, np.ndarray)):
            raise ValueError('`dropout_rate` must be either a float, or a 2-element list/tuple/array')
        
        if n_layers > 5:
            print('Max Number of Layers is 5; setting `n_layers=5`')
            n_layers = 5
        
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            print("Not Using TensorFlow")
            inputShape = (depth, height, width)
            chanDim = 1
        
        # CONV => eLU => POOL
        ''' Gospel of Alex
            with largest images, we may want to use larger kernel_sizes and larger strides with maxPooling
        
            example: kernel_size (3,3) --> (10,10) or even (20,20) (still smaller than the spatial scales now)
                -- could try up to (120,120) to keep the same 'size scale' that we have now: 100x100 images at (3,3) kernel_sizes
            
        '''
        model = Sequential()
        
        # if n_layers > 0:
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size),input_shape=inputShape))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((pool_size, pool_size), strides=(stride_size, stride_size)))
        model.add(Dropout(dropout_rate[0]))
        
        # if n_layers > 1:
        depth0 = 2*depth0
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((pool_size, pool_size), strides=(stride_size, stride_size)))
        model.add(Dropout(dropout_rate[0]))
        
        # if n_layers > 2:
        depth0 = 2*depth0
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((pool_size, pool_size), strides=(stride_size, stride_size)))
        model.add(Dropout(dropout_rate[0]))
        
        # if n_layers > 3:
        depth0 = 2*depth0
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((pool_size, pool_size), strides=(stride_size, stride_size)))
        model.add(Dropout(dropout_rate[0]))
        
        # if n_layers > 4:
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        if zero_pad: model.add(ZeroPadding2D((zero_pad_size, zero_pad_size)))
        model.add(Conv2D(depth0, (kernel_size, kernel_size), activation=activation, padding="same", use_bias=use_bias))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((pool_size, pool_size), strides=(stride_size, stride_size)))
        model.add(Dropout(dropout_rate[0]))
        
        depth0 = 8*depth0
        model.add(Flatten())
        model.add(Dense(depth0, activation=activation))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(dropout_rate[1]))
        model.add(Dense(depth0, activation=activation))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(dropout_rate[1]))
        model.add(Dense(classes, activation='softmax'))
        
        # return the constructed network architecture
        return model