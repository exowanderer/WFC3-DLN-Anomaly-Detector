from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

from keras.layers.normalization import BatchNormalization
from keras import backend as K

class vgg_net_16:
    @staticmethod
    def build(width, height, depth, classes, n_layers=5, depth0=64):
        if n_layers > 5: n_layers = 5
        
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
        
        if n_layers > 0:
            model.add(ZeroPadding2D((1,1),input_shape=inputShape))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            model.add(Dropout(0.25))
        
        if n_layers > 1:
            depth0 = 2*depth0
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            model.add(Dropout(0.25))
        
        if n_layers > 2:
            depth0 = 2*depth0
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            model.add(Dropout(0.25))
        
        if n_layers > 3:
            depth0 = 2*depth0
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            model.add(Dropout(0.25))
        
        if n_layers > 4:
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            # model.add(ZeroPadding2D((1,1)))
            model.add(Conv2D(depth0, (3, 3), activation='elu', padding="same", use_bias=False))
            model.add(BatchNormalization(axis=chanDim))
            model.add(MaxPooling2D((2,2), strides=(2,2)))
            model.add(Dropout(0.25))
        
        depth0 = 8*depth0
        model.add(Flatten())
        model.add(Dense(depth0, activation='elu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(depth0, activation='elu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        
        # return the constructed network architecture
        return model