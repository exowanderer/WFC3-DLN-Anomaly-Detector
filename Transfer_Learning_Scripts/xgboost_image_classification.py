## from https://medium.com/@14prakash/transfer-learning-using-keras-d804b2e04ef8

print("[INFO] Loading necessary libraries.")
from keras.preprocessing import image

from glob import glob
from multiprocessing import cpu_count
from sklearn.preprocessing import LabelBinarizer
from time import time
from tqdm import tqdm

from xgboost import XGBClassifier

import numpy as np
import os
import random

def glob_subdirectories(base_dir, verbose=False):
    list_of_files = []
    if verbose: print('[INFO] Globbing {}'.format('{}/*'.format(base_dir)))
    for subdir in glob('{}/*'.format(base_dir)):
        if verbose: print('[INFO] Globbing {}'.format('{}/*'.format(subdir)))
        list_of_files.extend(glob('{}/*'.format(subdir)))
    
    return list_of_files

def load_data_from_file(filenames, img_size=256):
    
    print('[INFO] Loading images and reshaping to {}x{}'.format(img_size, img_size))
    
    features = []
    labels = []
    
    # loop over the input images
    for kimage, imagePath in tqdm(enumerate(filenames), total=len(filenames)):
        img = image.load_img(imagePath, target_size=(img_size, img_size))
        img = image.img_to_array(img, dtype='uint8')[:,:,0]
        features.append(img[0].flatten())
        
        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2] # /path/to/data/class_name/filename.jpg
        labels.append(label)
        
        del imagePath
    
    return features, labels

print("[INFO] Establishing the location and size of our images.")
img_width, img_height = 256, 256
# base_dir = '/Research/HST_Public_DLN/Data'
# train_data_dir = os.environ['HOME'] + base_dir + "/train"
# validation_data_dir = os.environ['HOME'] + base_dir + "/validation"

base_dir = '/Research/QuickLookDLN/dataset_all'
train_data_dir = os.environ['HOME'] + base_dir + "/train"
validation_data_dir = os.environ['HOME'] + base_dir + "/validation"

# grab the image paths and randomly shuffle them
print("[INFO] loading training images...")
train_filenames = glob_subdirectories(train_data_dir)

print("[INFO] loading validation images...")
validation_filenames = glob_subdirectories(validation_data_dir)

random.seed(42)
random.shuffle(train_filenames)
random.shuffle(validation_filenames)

trainX, trainY = load_data_from_file(train_filenames, img_size=img_width)#, n_jobs=args['ncores'], verbose=True)
testX, testY = load_data_from_file(validation_filenames, img_size=img_width)#, n_jobs=args['ncores'], verbose=True)

trainX = np.array(trainX, dtype='uint8')
testX = np.array(testX, dtype='uint8')

# binarize the labels - one hot encoding
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY).argmax(axis=1)
testY = lb.transform(testY).argmax(axis=1)

num_classes = len(lb.classes_)

max_depth = 3
n_estimators = 100000
learning_rate = 0.05
model = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, n_jobs=cpu_count())

# Train the model 
start = time()
print("[INFO] Fitting the XGBoost")
# eval_set = [testX, testY]
H = model.fit(trainX, trainY, verbose=True) #eval_metric='auc', eval_metric="error", eval_set=eval_set
# H = model.fit(trainX, trainY, eval_metric="error", eval_set=eval_set, verbose=True) #eval_metric='auc'
print("[INFO] Finished full run process in {} minutes".format((time() - start)/60))



# from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#multiclass-settings
from matplotlib import use
use('Agg')
plt.figure()
from matplotlib import pyplot as plt
plt.figure()
lw = 2
plt.plot(fpr[2], tpr[2], color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
plt.savefig('roc_curve_multiclass_xgboost_{}.png'.format(n_estimators))