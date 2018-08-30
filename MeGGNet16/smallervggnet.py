# Keras Implementation of a Light VGG16 DLN from https://www.pyimagesearch.com/2018/04/16/keras-and-convolutional-neural-networks-cnns/
# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K

class SmallerVGGNet:
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
        '''
            Got a tip from Andrej Karpathy to set use_bias=False on the Conv2D in conjunction with BatchNormalization
                (https://twitter.com/karpathy/status/1013244313327681536) 
        '''
        
        
        model.add(Conv2D(filters=32,             # original settings
                         kernel_size=(3, 3),     # original settings
                         padding='same',         # original settings: maybe like zero-padding -- keep same dimensions at all costs
                         input_shape=inputShape, # original settings: kwarg
                         strides=(1, 1), 
                         data_format=None, 
                         dilation_rate=(1, 1), 
                         activation=None, 
                         use_bias=False, 
                         kernel_initializer='glorot_uniform', 
                         bias_initializer='zeros', 
                         kernel_regularizer=None, 
                         bias_regularizer=None, 
                         activity_regularizer=None, 
                         kernel_constraint=None, 
                         bias_constraint=None))
        
        # model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("elu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        
        # (CONV => eLU) * 2 => POOL
        model.add(Conv2D(64, (3, 3), activation='elu', padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), activation='elu', padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # (CONV => eLU) * 2 => POOL
        model.add(Conv2D(128, (3, 3), activation='elu', padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), activation='elu', padding="same", use_bias=False))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # first (and only) set of FC => eLU layers
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("elu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        # softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        
        # return the constructed network architecture
        return model