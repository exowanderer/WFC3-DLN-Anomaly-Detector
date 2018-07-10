# START WITH ARGPARSE TO MINIMIZE IMPORTS with python thingfile.py -h --help
import argparse
from multiprocessing import cpu_count

def val_acc_range_type(astr, min=0.0, max=1.0):
    value = float(astr)
    if min <= value <= max:
        return value
    else:
        raise argparse.ArgumentTypeError('value not in range %s-%s'%(min,max))

args = {}
args["dataset"]       = 'dataset_all'
args["model"]         = 'rename_me_wfc3_vgg16_model'
args["labelbin"]      = 'rename_me_lb'
args["plot"]          = 'rename_me_wfc3_vgg16_model_loss_acc.png'
args["ncores"]        = cpu_count()-1
args["niters"]        = 100
args["l_rate"]        = 1e-3
args["batch_size"]    = 32
args["image_size"]    = 100
args["min_val_acc"]   = 0.65
args["fine_tune"]     = True

# # pip install --upgrade imutils
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d"  , "--dataset"    , type=str  , required=False, 
    help="path to input dataset (i.e., directory of images)", default='dataset_all')
ap.add_argument("-m"  , "--model"      , type=str  , required=False, 
    help="path to output model"        , default='rename_me_wfc3_vgg16_model')
ap.add_argument("-l"  , "--labelbin"   , type=str  , required=False, 
    help="path to output label binarizer"                   , default='rename_me_lb')
ap.add_argument("-p"  , "--plot"       , type=str  , required=False, 
    help="path to output accuracy/loss plot"                , default='rename_me_wfc3_vgg16_model_loss_acc.png')
ap.add_argument("-nc" , "--ncores"     , type=int  , required=False, 
    help="number of cpu cores to use; default == ALL"       , default=cpu_count())
ap.add_argument("-ni" , "--niters"     , type=int  , required=False, 
    help="number of iterations to use; default == 100"      , default=100)
ap.add_argument("-lr" , "--l_rate"     , type=float, required=False, 
    help="initial learning rate"       , default=1e-3)
ap.add_argument("-bs" , "--batch_size" , type=float, required=False, 
    help="batch_size per iteration"    , default=32)
ap.add_argument("-is" , "--image_size" , type=int  , required=False, 
    help="Assuming square images, size of the image"    , default=100)
ap.add_argument('-mva', '--min_val_acc', type=val_acc_range_type, required=False, 
    help="Minimum validation accuracy to begin EarlyStopping criterion", 
    default=100)
ap.add_argument('-ft', '--fine_tune', type=bool, required=False, 
    help="True: Only fit the top layer; False: Fit all layers"    , default=True)

args = vars(ap.parse_args())


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import keras
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten

import os
from tqdm import tqdm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
''' THIS IS SUPPOSED TO GO INTO A NEW FILE AND *CALL* THE ABOVE CLASS '''

from matplotlib import use
use('Agg')

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
# import pickle
import random

# import the necessary packages
from imutils import paths
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from time import time
from tqdm import tqdm

from tensorflow import ConfigProto, Session

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

DATA_DIR    = args["dataset"] # 'dataset_all'
EPOCHS      = args["niters"]#100
INIT_LR     = args["l_rate"]#1e-3
BS          = args["batch_size"] #32
BASELINE    = args["min_val_acc"] # 0.65
N_CORES     = args["ncores"]
SAVE_NAME   = args["model"]
LB_SAVENAME = args["labelbin"]
PLT_SAVENAME= args["plot"]
IM_SIZE     = args["image_size"] # 100
FINE_TUNE   = args["fine_tune"] # True
IMAGE_DIMS  = (IM_SIZE,IM_SIZE,1)

# initialize the data and labels
data    = []
labels  = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths  = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in tqdm(imagePaths, total=len(imagePaths)):
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    image = img_to_array(image)[:,:,:1]
    data.append(image)
    
    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data   = np.array(data, dtype="float16") / 255.0
labels = np.array(labels)

print("[INFO] data  matrix: {}MB".format(data.nbytes // (1024 * 1000)))
print("[INFO] data  shape : {}".format(data.shape))
print("[INFO] label shape : {}".format(labels.shape))

# binarize the labels
lb     = LabelBinarizer()
labels = lb.fit_transform(labels)

num_class = len(lb.classes_)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
print('trainX.shape:{}\ntestX.shape:{}\ntrainY.shape:{}\ntestY.shape'.format(
       trainX.shape,    testX.shape,    trainY.shape,    testY.shape))

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


# Create the base pre-trained model
# Can't download weights in the kernel
base_model = VGG16(weights = None, include_top=False, input_shape=(IM_SIZE, IM_SIZE, 1))

# Add a new top layer
x = base_model.output
x = Flatten()(x)
predictions = Dense(num_class, activation='softmax')(x)

# This is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
if FINE_TUNE:
    for layer in base_model.layers:
        layer.trainable = False

model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

print(model.summary())

early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, mode='max',
                                                verbose=1, baseline=args["min_val_acc"])

tensboard = TensorBoard(log_dir='./logs/log-{}'.format(int(time())), histogram_freq=0, batch_size=BS, write_graph=True,
                     write_grads=False, write_images=False, embeddings_freq=0,
                     embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

testcall = TestCallback((X_test, Y_test))

callbacks_list = [tensboard]#[early_stopping, tensboard, testcall]

print("[INFO] training network...")
start = time()
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS), epochs=EPOCHS, verbose=1, # batch_size=BS, 
          callbacks=callbacks_list, validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
          shuffle=True)
print('\n\n *** Full TensorFlow Training Took {} minutes'.format((time()-start)//60))

# save the model to disk
print("[INFO] Storing VGG16 Net...")
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
    
    sub = pd.DataFrame(preds)
    # Set column names to those generated by the one-hot encoding earlier
    # col_names   = lb.classes_#one_hot.columns.values
    sub.columns = lb.classes_
    # Insert the column id from the sample_submission at the start of the data frame
    
    # sub.insert(0, 'id', df_test['id'])
    
    print(sub.head(5))
    
    joblib.dump(sub, args["labelbin"].replace('lb', 'pred'))
except Exception as e:
    print('Prediction step failed because', e.message, e.args)

