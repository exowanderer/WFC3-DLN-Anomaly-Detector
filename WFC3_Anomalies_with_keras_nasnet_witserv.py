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

ap.add_argument("-td", "--train_data", type=str, required=False, default='train', 
    help="path to input dataset (i.e., directory of images)")
ap.add_argument("-vd", "--validation_data", type=str, required=False, default='validation', 
    help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", type=str, required=False, default='rename_me_wfc3_MeGGNet_model', 
    help="path to output model")
ap.add_argument("-l", "--labelbin", type=str, required=False, default='rename_me_lb', 
    help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, required=False, default="rename_me_wfc3_MeGGNet_model_loss_acc.png", 
    help="path to output accuracy/loss plot")
ap.add_argument("-nc", "--ncores", type=int, required=False, default=1, 
    help="number of cpu cores to use; default == ALL")
ap.add_argument("-ni", "--niters", type=int, required=False, default=10, 
    help="number of iterations to use; default == 10")
ap.add_argument("-lr", "--l_rate", type=float, required=False, default=1e-3, 
    help="initial learning rate")
ap.add_argument("-bs", "--batch_size", type=int, required=False, default=32, 
    help="batch_size per iteration")
ap.add_argument("-is", "--image_size", type=int, required=False, default=100, 
    help="batch_size per iteration")

ap.add_argument('-a', '--activation', type=str, required=False, default='elu', 
    help='Select which activation function to use between each Conv2D layer.')
ap.add_argument('-nl', '--n_layers', type=int, choices=range(1,6), required=False, default=1, 
    help='Select the number of convolutional layers from 1 to 5.')
ap.add_argument('-d0', '--depth0', type=partial(greater_than, min=1), required=False, default=32, 
    help='The depth of the first Conv2D layer; subsequent layers are double in depth, and half in width.')
ap.add_argument('-ks', '--kernel_size', type=partial(greater_than, min=3), required=False, default=5, 
    help='Select the size of the Conv2D kernel (symmetric).')
ap.add_argument('-dr0', '--dropout_rate0', type=in_range, required=False, default=0.25, 
    help='Select the Conv2D layer dropout rate.')
ap.add_argument('-dr1', '--dropout_rate1', type=in_range, required=False, default=0.50, 
    help='Select the Top, Dense layer dropout rate.')
ap.add_argument('-ps', '--pool_size', type=partial(greater_than, min=2), required=False, default=2, 
    help='The size of the MaxPool2D pool size (symmetric).')
ap.add_argument('-ss', '--stride_size', type=partial(greater_than, min=2), required=False, default=2, 
    help='The size of the MaxPool2D stride size (symmetric).')
ap.add_argument('-b', '--use_bias', type=bool, required=False, default=False, 
    help='Select whether to activate a bias term for each Conv2D layer (not recomended).')
ap.add_argument('-zp', '--zero_pad', type=bool, required=False, default=False, 
    help="Select whether to zero pad between each Conv2D layer (nominally taken care of inside Conv2D(padding='same')).")
ap.add_argument('-zps', '--zero_pad_size', type=partial(greater_than, min=1), required=False, default=1, 
    help="Select the kernel size for the zero pad between each Conv2D layer.")

ap.add_argument('-ft', '--fine_tune', type=bool, required=False, default=False, 
    help="True: Only fit the top layer; False: Fit all layers")
ap.add_argument('-mva', '--min_val_acc', type=in_range, required=False, default=0.7, 
    help='The minimum validation accuracy to begin checking for Early Stopping Criteria.')

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

FINE_TUNE = args['fine_tune'] if inputs_found else ap['fine_tune'].get_default()
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

from keras import backend as K
from keras import Model
from keras.applications import NASNetLarge
# from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from MyBox_O_DNNs.MeGGNet16 import MeGGNet16
from time import time
from tqdm import tqdm

from tensorflow import ConfigProto, Session

# initialize the data and labels
# trainX = []
# testX = []
#
# trainY = []
# testY = []

# args["train_data"] = r"/home/ubuntu/Research/HST_Public_DLN/Data/train/"
# args["validation_data"] = r"/home/ubuntu/Research/HST_Public_DLN/Data/validation/"
# test_dir = r"/home/ubuntu/Research/HST_Public_DLN/Data/test/"

args["train_data"] = os.environ['HOME'] + r"/Research/QuickLookDLN/dataset_all/"
# grab the image paths and randomly shuffle them
print("[INFO] loading training images...")
data_filenames = sorted(list(paths.list_images(args["train_data"])))

random.seed(42)
random.shuffle(data_filenames)

dataX = []
dataY = []

data_filenames_strat = {}
for fname in data_filenames:
    class_label = os.path.dirname(fname).split(os.path.sep)[-1]
    if class_label not in data_filenames_strat.keys():
        data_filenames_strat[class_label] = fname

# loop over the input images
for imagePath in tqdm(data_filenames_strat.values(), total=len(data_filenames)):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)[:,:,:1]
    dataX.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = os.path.dirname(imagePath).split(os.path.sep)[-1]
    # label = imagePath.split(os.path.sep)[-2] # /path/to/data/class_name/filename.jpg
    dataY.append(label)

# scale the raw pixel intensities to the range [0, 1]
dataX = np.array(dataX, dtype="float16") / 255.0
dataY = np.array(dataY)
print("[INFO] data  matrix: {:.2f}MB".format(dataX.nbytes / (1024 * 1000.0)))
print("[INFO] data  shape : {}".format(dataX.shape))
print("[INFO] label shape : {}".format(dataY.shape))

test_size = 0.2 # 20%
idx_train, idx_test = train_test_split(np.arange(len(dataY)), test_size=test_size)

trainX = dataX[idx_train]
testX = dataX[idx_test]

trainY = dataY[idx_train]
testY = dataY[idx_test]

# binarize the labels - one hot encoding
lb     = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

num_class = len(lb.classes_)

train_labels_raw = trainY.argmax(axis=1)#np.where(trainY==1)[1]
test_labels_raw = testY.argmax(axis=1)#np.where(testY==1)[1]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
# (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
# print(trainX.shape, testX.shape, trainY.shape, testY.shape)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=360, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, vertical_flip=True, fill_mode="nearest")

# initialize the model
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))
# with K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=args["ncores"], 
#                                           inter_op_parallelism_threads=args["ncores"])) as sess:
# with K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=args["ncores"])) as sess:
# K.set_session(sess)

filepath = 'keras_checkpoints/'
checkpoints = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

tensboard = TensorBoard(log_dir='./logs/log-{}'.format(int(time())), histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True,
                     write_grads=False, write_images=False, embeddings_freq=0,
                     embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

# Create the base pre-trained model
# Can't download weights in the kernel
base_model = NASNetLarge(weights = None, include_top=False, input_shape=(IM_SIZE, IM_SIZE, 1))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
if FINE_TUNE: 
	for layer in base_model.layers: layer.trainable = False

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1,
#                         baseline=args["min_val_acc"], mode='max')

# callbacks_list = [tensboard, checkpoints]#, early_stopping]
print(model.summary())

print("[INFO] training network...")
start = time()
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE), epochs=EPOCHS, verbose=1,  
          validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,#callbacks=callbacks_list, 
          shuffle=True)
print('\n\n *** Full TensorFlow Training Took {} minutes'.format((time()-start)//60))

# save the model to disk
print("[INFO] Storing NASNet...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] Storing Label Binarizer...")
joblib.dump(lb, args["labelbin"] + 'joblib.save')

plt.style.use("ggplot")
plt.figure()
N = len(H.history["loss"])
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])

try:
    preds = model.predict(testX, verbose=1)
    
    sub = pd.DataFrame(preds, columns=lb.classes_)
    
    # Insert the column id from the sample_submission at the start of the data frame
    
    # sub.insert(0, 'id', df_test['id'])
    
    print(sub.head(5))
    
    joblib.dump(sub, args["labelbin"].replace('lb', 'pred'))
except Exception as e:
    print('Prediction step failed because', e.message, e.args)

