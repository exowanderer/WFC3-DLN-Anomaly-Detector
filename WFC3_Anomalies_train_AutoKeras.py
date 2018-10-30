import argparse
# from multiprocessing import cpu_count
from functools import partial

def in_range(value, min=0, max=1, dtype=float):
    ivalue = dtype(value)
    if min <= ivalue <= max:
        return ivalue
    
    raise argparse.ArgumentTypeError("{} is invalid; the `dropout_rate` must be either from [0,1] (inclusive).".format(value))
    
def greater_than(value, min=0):
    ivalue = int(value)
    if min <= ivalue:
         return ivalue
    
    raise argparse.ArgumentTypeError("{} is invalid; `n_layers` must be either 1, 2, 3, 4, or 5.".format(value))

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-td", "--train_data", type=str, required=False, help="path to input dataset (i.e., directory of images)", default='train')
ap.add_argument("-vd", "--validation_data", type=str, required=False, help="path to input dataset (i.e., directory of images)", default='validation')
ap.add_argument("-m", "--model", type=str, required=False, help="path to output model", default='rename_me_wfc3_MeGGNet_model')
ap.add_argument("-l", "--labelbin", type=str, required=False, help="path to output label binarizer", default='rename_me_lb')
ap.add_argument("-p", "--plot", type=str, required=False, help="path to output accuracy/loss plot", default="rename_me_wfc3_MeGGNet_model_loss_acc.png")
ap.add_argument("-nc", "--ncores", type=int, required=False, help="number of cpu cores to use; default == ALL", default=1)
ap.add_argument("-ni", "--niters", type=int, required=False, help="number of iterations to use; default == 10", default=10)
ap.add_argument("-lr", "--l_rate", type=float, required=False, help="initial learning rate", default=1e-3)
ap.add_argument("-bs", "--batch_size", type=int, required=False, help="batch_size per iteration", default=32)
ap.add_argument("-is", "--image_size", type=int, required=False, help="batch_size per iteration", default=100)

ap.add_argument('-a', '--activation', type=str, required=False, default='elu', help='Select which activation function to use between each Conv2D layer.')
ap.add_argument('-nl', '--n_layers', type=int, choices=range(1,6), required=False, default=1, help='Select the number of convolutional layers from 1 to 5.')
ap.add_argument('-d0', '--depth0', type=partial(greater_than, min=1), required=False, default=32, help='The depth of the first Conv2D layer; subsequent layers are double in depth, and half in width.')
ap.add_argument('-ks', '--kernel_size', type=partial(greater_than, min=3), required=False, default=5, help='Select the size of the Conv2D kernel (symmetric).')
ap.add_argument('-dr0', '--dropout_rate0', type=in_range, required=False, default=0.25, help='Select the Conv2D layer dropout rate.')
ap.add_argument('-dr1', '--dropout_rate1', type=in_range, required=False, default=0.50, help='Select the Top, Dense layer dropout rate.')
ap.add_argument('-ps', '--pool_size', type=partial(greater_than, min=2), required=False, default=2, help='The size of the MaxPool2D pool size (symmetric).')
ap.add_argument('-ss', '--stride_size', type=partial(greater_than, min=2), required=False, default=2, help='The size of the MaxPool2D stride size (symmetric).')
ap.add_argument('-b', '--use_bias', type=bool, required=False, default=False, help='Select whether to activate a bias term for each Conv2D layer (not recomended).')
ap.add_argument('-zp', '--zero_pad', type=bool, required=False, default=False, help="Select whether to zero pad between each Conv2D layer (nominally taken care of inside Conv2D(padding='same')).")
ap.add_argument('-zps', '--zero_pad_size', type=partial(greater_than, min=1), required=False, default=1, help="Select the kernel size for the zero pad between each Conv2D layer.")

try:
    args = vars(ap.parse_args())
    inputs_found = True
except:
    inputs_found = False
# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

EPOCHS = args["niters"] if inputs_found else ap['niters'].get_default()
INIT_LR = args["l_rate"] if inputs_found else ap['l_rate'].get_default()
BATCH_SIZE = args["batch_size"] if inputs_found else ap['batch_size'].get_default()
IM_SIZE = args['image_size'] if inputs_found else ap['image_size'].get_default()
IMAGE_DIMS = (IM_SIZE,IM_SIZE,1)

ACTIVATION = args['activation'] if inputs_found else ap['activation'].get_default()
N_LAYERS = args['n_layers'] if inputs_found else ap['n_layers'].get_default()
DEPTH0 = args['depth0'] if inputs_found else ap['depth0'].get_default()
KERNEL_SIZE = args['kernel_size'] if inputs_found else ap['kernel_size'].get_default()
DROPOUT_RATE0 = args['dropout_rate0'] if inputs_found else ap['dropout_rate0'].get_default()
DROPOUT_RATE1 = args['dropout_rate1'] if inputs_found else ap['dropout_rate1'].get_default()
POOL_SIZE = args['pool_size'] if inputs_found else ap['pool_size'].get_default()
STRIDE_SIZE = args['stride_size'] if inputs_found else ap['stride_size'].get_default()
USE_BIAS = args['use_bias'] if inputs_found else ap['use_bias'].get_default()
ZERO_PAD = args['zero_pad'] if inputs_found else ap['zero_pad'].get_default()
ZERO_PAD_SIZE = args['zero_pad_size'] if inputs_found else ap['zero_pad_size'].get_default()

from matplotlib import use
use('Agg')

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
import os
# import pickle
import random

# import the necessary packages
from imutils import paths

from autokeras import ImageClassifier

from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
# from MyBox_O_DNNs.MeGGNet16 import MeGGNet16
from time import time
from tqdm import tqdm

from tensorflow import ConfigProto, Session

# initialize the data and labels
trainX = []
testX = []

trainY = []
testY = []

args["train_data"] = r"/home/ubuntu/Research/HST_Public_DLN/Data/train/"
args["validation_data"] = r"/home/ubuntu/Research/HST_Public_DLN/Data/validation/"
test_dir = r"/home/ubuntu/Research/HST_Public_DLN/Data/test/"

# grab the image paths and randomly shuffle them
print("[INFO] loading training images...")
train_filenames = sorted(list(paths.list_images(args["train_data"])))

random.seed(42)
random.shuffle(train_filenames)

# loop over the input images
for imagePath in tqdm(train_filenames, total=len(train_filenames)):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)[:,:,:1]
    trainX.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2] # /path/to/data/class_name/filename.jpg
    trainY.append(label)

# grab the image paths and randomly shuffle them
print("[INFO] loading validation images...")
validation_filenames = sorted(list(paths.list_images(args["validation_data"])))

random.seed(42)
random.shuffle(validation_filenames)

# loop over the input images
for imagePath in tqdm(validation_filenames, total=len(validation_filenames)):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)[:,:,:1]
    testX.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2] # /path/to/data/class_name/filename.jpg
    testY.append(label)

# scale the raw pixel intensities to the range [0, 1]
trainX = np.array(trainX, dtype="float16") / 255.0
testX = np.array(testX, dtype="float16") / 255.0
trainY = np.array(trainY)
testY = np.array(testY)
print("[INFO] data  matrix: {:.2f}MB".format(trainX.nbytes / (1024 * 1000.0)))
print("[INFO] data  shape : {}".format(trainX.shape))
print("[INFO] label shape : {}".format(trainY.shape))

# trainX = trainX.reshape(trainX.shape + (1,))
# testX = testX.reshape(testX.shape + (1,))

print(trainX.shape, trainY.shape, testX.shape, testY.shape)

clf = ImageClassifier(path='autokeras_output/', verbose=True, augment=False)
clf.fit(trainX, trainY, time_limit=12 * 60 * 60)
clf.final_fit(trainX, trainY, testX, testY, retrain=True)
y = clf.evaluate(testX, testY)

print(y * 100)

joblib.dump( clf, 'wfc3_autokeras_model.joblib.save')

'''
from keras.datasets import mnist
from autokeras import ImageClassifier

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))
	
    clf = ImageClassifier(verbose=True, augment=False)
    clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
    clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
    y = clf.evaluate(x_test, y_test)
    print(y * 100)
'''