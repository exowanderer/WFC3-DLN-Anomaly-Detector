from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

from keras.layers.normalization import BatchNormalization
from keras import backend as K

class VGGNet16:
    @staticmethod
    def build(width, height, depth, classes):
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
        
        model.add(ZeroPadding2D((1,1),input_shape=inputShape))
        model.add(Conv2D(64, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        # model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, (3, 3), activation='relu')#, padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Dropout(0.5))
        model.add(Dense(classes, activation='softmax'))
        
        # return the constructed network architecture
        return model