''' THIS IS SUPPOSED TO GO INTO A NEW FILE AND *CALL* THE ABOVE CLASS '''

from matplotlib import use
use('Agg')

import argparse
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
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from MeGGNet16.MeGGNet16 import MeGGNet16
from time import time
from tqdm import tqdm


from tensorflow import ConfigProto, Session
from multiprocessing import cpu_count

# # pip install --upgrade imutils
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()

ap.add_argument("-d" , "--dataset"   , type=str  , required=False, help="path to input dataset (i.e., directory of images)", default='dataset'                                         )
ap.add_argument("-m" , "--model"     , type=str  , required=False, help="path to output model"                             , default='rename_me_wfc3_MeGGNet_model'                      )
ap.add_argument("-l" , "--labelbin"  , type=str  , required=False, help="path to output label binarizer"                   , default='rename_me_lb'                                    )
ap.add_argument("-p" , "--plot"      , type=str  , required=False, help="path to output accuracy/loss plot"                , default="rename_me_wfc3_MeGGNet_model_loss_acc.png")
ap.add_argument("-nc", "--ncores"    , type=int  , required=False, help="number of cpu cores to use; default == ALL"       , default=cpu_count()                                       )
ap.add_argument("-ni", "--niters"    , type=int  , required=False, help="number of iterations to use; default == 100"      , default=100                                               )
ap.add_argument("-lr", "--l_rate"    , type=float, required=False, help="initial learning rate"                            , default=1e-3                                              )
ap.add_argument("-bs", "--batch_size", type=float, required=False, help="batch_size per iteration"                         , default=32                                                )

args = vars(ap.parse_args())

# args = {}
# args["dataset"]       = 'dataset'
# args["model"]         = 'jdf_edits_2nd_test_wfc3_MeGGNet_model'
# args["labelbin"]      = 'jdf_edits_2nd_test_lb'
# args["plot"]          = 'jdf_edits_2nd_test_wfc3_MeGGNet_model_loss_acc.png'
# args["ncores"]        = cpu_count()-1
# args["niters"]        = 100
# args["learning_rate"] = 1e-3
# args["batch_size"]    = 32

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

EPOCHS      = args["niters"]#100
INIT_LR     = args["l_rate"]#1e-3
BS          = args["batch_size"] #32
IMAGE_DIMS  = (100,100,1)

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
model = MeGGNet16.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
    depth=IMAGE_DIMS[2], classes=len(lb.classes_))
    
model = MeGGNet16.build(width=IMAGE_DIMS[1], height=IMAGE_DIMS[0], 
                        depth=IMAGE_DIMS[2], classes=len(lb.classes_), 
                        activation='relu', n_layers=1, depth0=32, 
                        kernel_size=3, dropout_rate=[0.25,0.5], pool_size=2,
                        stride_size=2, use_bias=False, zero_pad=False, 
                        zero_pad_size=1)

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

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
29/29 [==============================] - 2s - loss: 1.4015 - acc: 0.6088 - val_loss: 1.8745 - val_acc: 0.2134
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
