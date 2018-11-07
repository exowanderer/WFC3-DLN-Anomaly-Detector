# from: https://towardsdatascience.com/https-medium-com-manishchablani-useful-keras-features-4bac0724734c

from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from keras.preprocessing import image
from sklearn.preprocessing import LabelBinarizer
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
from keras.datasets import mnist

from time import time
import numpy as np

# Load the data
print('[INFO] Loading MNIST Data Set.')
(trainX, trainY), (testX, testY) = mnist.load_data()
trainX = trainX.reshape(trainX.shape + (1,))
testX = testX.reshape(testX.shape + (1,))

trainX = np.transpose(list(trainX.T)*3)
testX = np.transpose(list(testX.T)*3)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

num_class = len(lb.classes_)

# create the base pre-trained model
print('[INFO] Creating Base Model from Pre-trained Instance.')
base_model = InceptionV3(weights='imagenet', include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)

# let's add a fully-connected layer
n_hidden_units = 1024
x = Dense(n_hidden_units, activation='elu')(x)

# and a logistic layer -- let's say we have 200 classes
predictions = Dense(num_class, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
print('Turning off all layers except top layer for transfer learning')

FINE_TUNE = False
for layer in base_model.layers:
    layer.trainable = FINE_TUNE

# compile the model (should be done *after* setting layers to non-trainable)
print("Compiling Model: optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']")
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

BATCH_SIZE = 32
EPOCHS = 10
VERBOSE = True
SHUFFLE = True
FIT_GEN = False


filepath = 'keras_checkpoints/'
checkpoints = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

tensboard = TensorBoard(log_dir='./logs/log-{}'.format(int(time())), histogram_freq=0, batch_size=BATCH_SIZE, write_graph=True,
                     write_grads=False, write_images=False, embeddings_freq=0,
                     embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)


early_stopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, baseline=0.9, mode='max')

callbacks_list = [tensboard, checkpoints, early_stopping]

print('[INFO] Beginning Network Fitting Process.')
if FIT_GEN:
    aug = image.ImageDataGenerator(rotation_range=360, width_shift_range=0.1,
                                    height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                    horizontal_flip=True, vertical_flip=True, fill_mode="nearest")
    
    # train the model on the new data for a few epochs
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE), epochs=EPOCHS, 
                                    verbose=VERBOSE, callbacks=callbacks_list, 
                                    validation_data=(testX, testY), 
                                    steps_per_epoch=len(trainX) // BATCH_SIZE, 
                                    validation_steps=len(testX) // BATCH_SIZE,
                                    shuffle=SHUFFLE)
else:
    H = model.fit(trainX, trainY, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=VERBOSE, 
                    callbacks=callbacks_list, validation_data=(testX, testY), 
                    # steps_per_epoch=len(trainX) // BATCH_SIZE,
                    # validation_steps=len(testX) // BATCH_SIZE,
                    shuffle=SHUFFLE)

# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.
# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers): print(i, layer.name)
#
# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 172 layers and unfreeze the rest:
# for layer in model.layers[:172]:
#    layer.trainable = False
#
# for layer in model.layers[172:]:
#    layer.trainable = True
#
# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')
#
# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.model.fit_generator(aug.flow(trainX, trainY, batch_size=BATCH_SIZE), epochs=EPOCHS, verbose=VERBOSE,
#           callbacks=callbacks_list, validation_data=(testX, testY), steps_per_epoch=len(trainX) // BATCH_SIZE,
#           shuffle=SHUFFLE)

# # save the model to disk
# print("[INFO] Storing {}...".format(base_model.name))
# model.save(args["model"])
#
# # save the label binarizer to disk
# print("[INFO] Storing Label Binarizer...")
# joblib.dump(lb, args["labelbin"] + 'joblib.save')
#
# plt.style.use("ggplot")
# plt.figure()
# N = len(H.history["loss"])
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="upper left")
# plt.savefig(args["plot"])
#
# try:
#     preds = model.predict(testX, verbose=1)
#
#     sub = pd.DataFrame(preds, columns=lb.classes_)
#
#     # Insert the column id from the sample_submission at the start of the data frame
#
#     # sub.insert(0, 'id', df_test['id'])
#
#     print(sub.head(5))
#
#     joblib.dump(sub, args["labelbin"].replace('lb', 'pred'))
# except Exception as e:
#     print('Prediction step failed because', e.message, e.args)
#
