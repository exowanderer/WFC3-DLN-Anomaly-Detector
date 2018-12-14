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

ap.add_argument("-d", "--dataset", type=str, required=False, help="path to input dataset (i.e., directory of images)", default='dataset')
ap.add_argument("-m", "--model", type=str, required=False, help="path to output model", default='rename_me_wfc3_MeGGNet_model')
ap.add_argument("-l", "--labelbin", type=str, required=False, help="path to output label binarizer", default='rename_me_lb')
ap.add_argument("-p", "--plot", type=str, required=False, help="path to output accuracy/loss plot", default="rename_me_wfc3_MeGGNet_model_loss_acc.png")
ap.add_argument("-nc", "--ncores", type=int, required=False, help="number of cpu cores to use; default == ALL", default=1)
ap.add_argument("-ni", "--niters", type=int, required=False, help="number of iterations to use; default == 10", default=10)
ap.add_argument("-lr", "--l_rate", type=float, required=False, help="initial learning rate", default=1e-3)
ap.add_argument("-bs", "--batch_size", type=int, required=False, help="batch_size per iteration", default=32)
ap.add_argument("-is", "--image_size", type=int, required=False, help="batch_size per iteration", default=256)

ap.add_argument('-a', '--activation', type=str, required=False, default='elu', help='Select which activation function to use between each Conv2D layer.')
ap.add_argument('-nl', '--n_layers', type=int, choices=range(1,6), required=False, default=5, help='Select the number of convolutional layers from 1 to 5.')
ap.add_argument('-d0', '--depth0', type=partial(greater_than, min=1), required=False, default=32, help='The depth of the first Conv2D layer; subsequent layers are double in depth, and half in width.')
ap.add_argument('-ks', '--kernel_size', type=partial(greater_than, min=3), required=False, default=3, help='Select the size of the Conv2D kernel (symmetric).')
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

N_CORES = args['ncores'] if inputs_found else ap['ncores'].get_default()

from matplotlib import use
use('Agg')

import keras
import matplotlib.pyplot as plt
import numpy as np
import os
# import pickle
import random
 
# import the necessary packages
# from imutils import paths
from glob import glob

from keras import backend as K
from keras.callbacks import TensorBoard
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


# from:
import os
import tensorflow as tf
from keras.callbacks import TensorBoard

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()

# construct the image generator for data augmentation
train_datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

test_datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

val_datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode="nearest")

tensboard = TrainValTensorBoard(log_dir='./logs/log-{}'.format(int(time())), batch_size=BATCH_SIZE, write_graph=False)
# tensboard = TensorBoard(log_dir='./logs/log-{}'.format(int(time())), histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True,
#                      write_grads=False, write_images=False, embeddings_freq=0,
#                      embeddings_layer_names=None, embeddings_metadata=None)

train_dir = os.environ['HOME'] + "/Research/CHADWICK/Data/train/"
test_dir = os.environ['HOME'] + "/Research/CHADWICK/Data/test/"
val_dir = os.environ['HOME'] + "/Research/CHADWICK/Data/validation/"

train_generator = train_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(IM_SIZE,IM_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(IM_SIZE,IM_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

valid_generator = val_datagen.flow_from_directory(
    directory=val_dir,
    target_size=(IM_SIZE,IM_SIZE),
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

callbacks_list = [tensboard]#[early_stopping, tensboard, testcall]

print("[INFO] compiling model...")
N_CLASSES = len(glob(train_dir + '/*'))

model = MeGGNet16.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], 
                        depth=IMAGE_DIMS[2], classes=N_CLASSES, 
                        activation=ACTIVATION, n_layers=N_LAYERS, depth0=DEPTH0, 
                        kernel_size=KERNEL_SIZE, dropout_rate=[DROPOUT_RATE0,DROPOUT_RATE1], 
                        pool_size=POOL_SIZE, stride_size=STRIDE_SIZE, 
                        use_bias=USE_BIAS, zero_pad=ZERO_PAD, 
                        zero_pad_size=ZERO_PAD_SIZE)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print(model.summary())

# train the network
print("[INFO] training network...")
start = time()

STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n//valid_generator.batch_size

H = model.fit_generator(generator=train_generator,
	                    steps_per_epoch=STEP_SIZE_TRAIN,
	                    validation_data=valid_generator,
	                    validation_steps=STEP_SIZE_VALID,
	                    epochs=EPOCHS,
						callbacks=callbacks_list,
						shuffle=True)


print('\n\n *** Full TensorFlow Training Took {} minutes'.format((time()-start)//60))
# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# H = model.fit_generator(generator=aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
# 					    validation_data=(testX, testY),
# 					    steps_per_epoch=len(trainX) // BATCH_SIZE,
# 					    epochs=EPOCHS, verbose=1,
# 					    callbacks=callbacks_list,
# 					    shuffle=True
# 						)

eval_out = model.evaluate_generator(generator=valid_generator)
print(eval_out)

# save the label binarizer to disk
# print("[INFO] serializing label binarizer...")
# joblib.dump(lb, args["labelbin"] + 'joblib.save')
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