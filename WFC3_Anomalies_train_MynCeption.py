import argparse
from multiprocessing import cpu_count
from functools import partial

def in_range(value, min=0, max=1, dtype=float):
    ivalue = dtype(value)
    if min <= ivalue <= max:
        return ivalue
    
    raise argparse.ArgumentTypeError("{} is invalid; the `dropout_rate` must be either from [0,1] (inclusive).".format(value))
    
def int_greater_than(value, min=0):
    ivalue = int(value)
    if min <= ivalue:
         return ivalue
    
    raise argparse.ArgumentTypeError("{} is invalid; Value must be a positive integer.".format(value))

def float_greater_than(value, min=0):
    fvalue = float(value)
    if min <= fvalue:
         return fvalue
    
    raise argparse.ArgumentTypeError("{} is invalid; Value must be beteween [0, 1]".format(value))

def str2bool(input):
    if input.lower() in ['y', 't', 'true', True, 'yes']:
        return True
    if input.lower() in ['n', 'f', 'false', False, 'no']:
        return False
    return True # if flag exists with no inputs; i.e. `python file.py --flag`

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", type=str, required=False, default='dataset/', help="path to input dataset (i.e., directory of images)")
ap.add_argument("-m", "--model", type=str, required=False, default='rename_me_wfc3_MynCeptionNet_model', help="path to output model")
ap.add_argument("-l", "--labelbin", type=str, required=False, default='rename_me_lb', help="path to output label binarizer")
ap.add_argument("-p", "--plot", type=str, required=False, default="rename_me_wfc3_MynCeptionNet_model_loss_acc.png", help="path to output accuracy/loss plot")
ap.add_argument("-nc", "--ncores", type=int, required=False, default=cpu_count(), help="number of cpu cores to use; default == ALL")
ap.add_argument("-ni", "--niters", type=int, required=False, default=100, help="number of iterations to use; default == 100")
ap.add_argument("-lr", "--l_rate", type=float, required=False, default=1e-3, help="initial learning rate")
ap.add_argument("-bs", "--batch_size", type=int, required=False, default=32, help="batch_size per iteration")
ap.add_argument("-is", "--image_size", type=int, required=False, default=100, help="batch_size per iteration")
ap.add_argument("-id", "--image_depth", type=int, required=False, default=1, help="color depth of image; 1 == monochromatic (default) ")

ap.add_argument('-a', '--activation', type=str, required=False, default='elu', help='Select which activation function to use between each Conv2D layer.')
ap.add_argument('-nl', '--n_layers', type=partial(int_greater_than, min=1), required=False, default=1, help='Select the number of convolutional layers 1, or more')
ap.add_argument('-nt', '--n_towers', type=partial(int_greater_than, min=1), required=False, default=1, help="Number of towers in the inception module; Default=3 (0 == only the 'side layer')")
ap.add_argument('-d0', '--depth0', type=partial(int_greater_than, min=1), required=False, default=64, help='The depth of the Conv2D layers inside the inception module.')
ap.add_argument('-dr', '--dropout_rate', type=in_range, required=False, default=0.5, help='Select the Dropout layer dropout rate.')
ap.add_argument('-ps', '--pool_size', type=partial(int_greater_than, min=2), required=False, default=1, help='The size of the MaxPool2D pool size (symmetric).')
ap.add_argument('-ss', '--stride_size', type=partial(int_greater_than, min=1), required=False, default=1, help='The size of the MaxPool2D stride size (symmetric).')
ap.add_argument('-b', '--use_bias', type=str2bool, nargs='?', required=False, default=False, help='Select whether to activate a bias term for each Conv2D layer (not recomended).')
ap.add_argument('-zp', '--zero_pad', type=str2bool, nargs='?', required=False, default=False, help="Select whether to zero pad between each Conv2D layer (nominally taken care of inside Conv2D(padding='same')).")
ap.add_argument('-zps', '--zero_pad_size', type=partial(int_greater_than, min=1), required=False, default=1, help="Select the kernel size for the zero pad between each Conv2D layer.")
ap.add_argument('-kss', '--kernel_sizes', type=partial(int_greater_than, min=1), nargs='?', required=False, default=[3,5,1], help="Select the kernel sizes per tower: MUST be list of integers, with len == n_towers.")
ap.add_argument('-nsjg', '--n_skip_junc_gap', type=partial(int_greater_than, min=1), required=False, default=0, help="Number of inception layers before a skip junction; 0 = no skip juncitons.")

try:
    args = vars(ap.parse_args())
except:
    args = {}

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions
DATASET = args["dataset"] if 'dataset' in args.keys() else ap.get_default('dataset')
EPOCHS = args["niters"] if 'niters' in args.keys() else ap.get_default('niters')
INIT_LR = args["l_rate"] if 'l_rate' in args.keys() else ap.get_default('l_rate')
BS = args["batch_size"] if 'batch_size' in args.keys() else ap.get_default('batch_size')
IM_SIZE = args['image_size'] if 'image_size' in args.keys() else ap.get_default('image_size')
IM_DEPTH = args['image_depth'] if 'image_depth' in args.keys() else ap.get_default('image_depth')
IMAGE_DIMS = (IM_SIZE,IM_SIZE,IM_DEPTH)

ACTIVATION = args['activation'] if 'activation' in args.keys() else ap.get_default('activation')
N_LAYERS = args['n_layers'] if 'n_layers' in args.keys() else ap.get_default('n_layers')
DEPTH0 = args['depth0'] if 'depth0' in args.keys() else ap.get_default('depth0')
DROPOUT_RATE = args['dropout_rate'] if 'dropout_rate' in args.keys() else ap.get_default('dropout_rate')
POOL_SIZE = args['pool_size'] if 'pool_size' in args.keys() else ap.get_default('pool_size')
STRIDE_SIZE = args['stride_size'] if 'stride_size' in args.keys() else ap.get_default('stride_size')
USE_BIAS = args['use_bias'] if 'use_bias' in args.keys() else ap.get_default('use_bias')
ZERO_PAD = args['zero_pad'] if 'zero_pad' in args.keys() else ap.get_default('zero_pad')
ZERO_PAD_SIZE = args['zero_pad_size'] if 'zero_pad_size' in args.keys() else ap.get_default('zero_pad_size')
N_TOWERS = args['n_towers'] if 'n_towers' in args.keys() else ap.get_default('n_towers')
KERNEL_SIZES = args['kernel_sizes'] if 'kernel_sizes' in args.keys() else ap.get_default('kernel_sizes')
N_SKIP_JUNC_GAP = args['n_skip_junc_gap'] if 'n_skip_junc_gap' in args.keys() else ap.get_default('n_skip_junc_gap')

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
from keras.callbacks import TensorBoard
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from MyBox_O_DNNs.MynCeptionNet import MynCeptionNet
from time import time
from tqdm import tqdm

from tensorflow import ConfigProto, Session

# initialize the data and labels
data    = []
labels  = []

# grab the image paths and randomly shuffle them
print("[INFO] loading images...")
imagePaths  = sorted(list(paths.list_images(DATASET)))
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
data = np.array(data, dtype="float16") / 255.0
labels = np.array(labels)
print("[INFO] data  matrix: {:.2f}MB".format(data.nbytes / (1024 * 1000.0)))
print("[INFO] data  shape : {}".format(data.shape))
print("[INFO] label shape : {}".format(labels.shape))

# binarize the labels
lb     = LabelBinarizer()
labels = lb.fit_transform(labels)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)
print(trainX.shape, testX.shape, trainY.shape, testY.shape)
# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

# initialize the model
# K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)))
# with K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=args["ncores"], 
#                                           inter_op_parallelism_threads=args["ncores"])) as sess:
# with K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=args["ncores"])) as sess:
# K.set_session(sess)


# args["min_val_acc"] = 0.65
# early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=10, mode='max',
#                                                 verbose=1, baseline=args["min_val_acc"])

tensboard = TensorBoard(log_dir='./logs/log-{}'.format(int(time())), histogram_freq=0, batch_size=BS, write_graph=True,
                     write_grads=False, write_images=False, embeddings_freq=0,
                     embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

# testcall = TestCallback((X_test, Y_test))

callbacks_list = [tensboard]#[early_stopping, tensboard, testcall]

print("[INFO] compiling model...")
N_CLASSES = len(lb.classes_)

model = MynCeptionNet.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], 
                            depth=IMAGE_DIMS[2], classes=N_CLASSES,
                            activation=ACTIVATION, n_layers=N_LAYERS, depth0=DEPTH0, 
                            n_towers = N_TOWERS, kernel_sizes = KERNEL_SIZES, 
                            dropout_rate=DROPOUT_RATE, pool_size=POOL_SIZE,
                            stride_size=STRIDE_SIZE, use_bias=USE_BIAS, 
                            n_skip_junc_gap=N_SKIP_JUNC_GAP)

input_img = Input(shape = (32, 32, 3))
model = Model(inputs = input_img, outputs = model)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print(model.summary())

# train the network
print("[INFO] training network...")
start = time()
H = model.fit_generator(
    aug.flow(trainX, trainY, batch_size=BS),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // BS,
    epochs=EPOCHS, verbose=1,
    callbacks=callbacks_list,
    shuffle=True)

print('\n\n *** Full TensorFlow Training Took {} minutes'.format((time()-start)//60))
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# save the label binarizer to disk
print("[INFO] serializing label binarizer...")
joblib.dump(lb, args["labelbin"] + 'joblib.save')
# f = open(args["labelbin"], "wb")
# f.write(pickle.dumps(lb))
# f.close()

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])


''' Expected Terminal Output

$ $ python train.py --dataset dataset --model pokedex.model --labelbin lb.pickle
Using TensorFlow backend.
[INFO] loading images...
[INFO] data matrix: 252.07MB
[INFO] compiling model...
[INFO] training network...
name: GeForce GTX TITAN X
major: 5 minor: 2 memoryClockRate (GHz) 1.076
pciBusID 0000:09:00.0
Total memory: 11.92GiB
Free memory: 11.71GiB
Epoch 1/100
29/29 [
=========================] - 2s - loss: 1.4015 - acc: 0.6088 - val_loss: 1.8745 - val_acc: 0.2134
Epoch 2/100
29/29 [==============================] - 1s - loss: 0.8578 - acc: 0.7285 - val_loss: 1.4539 - val_acc: 0.2971
Epoch 3/100
29/29 [==============================] - 1s - loss: 0.7370 - acc: 0.7809 - val_loss: 2.5955 - val_acc: 0.2008
...
Epoch 98/100
29/29 [==============================] - 1s - loss: 0.0833 - acc: 0.9702 - val_loss: 0.2064 - val_acc: 0.9540
Epoch 99/100
29/29 [==============================] - 1s - loss: 0.0678 - acc: 0.9727 - val_loss: 0.2299 - val_acc: 0.9456
Epoch 100/100
29/29 [==============================] - 1s - loss: 0.0890 - acc: 0.9684 - val_loss: 0.1955 - val_acc: 0.9707
[INFO] serializing network...
[INFO] serializing label binarizer...


'''
