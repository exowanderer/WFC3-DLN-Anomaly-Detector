from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import os

subdirectories = 'train', 'test', 'validation'
for subdir in subdirectories:
    base_dir = '{}/'.format(subdir)
    class_names = glob('{}/*'.format(base_dir))
    
    n_letters = len(base_dir)
    for kc, class_ in enumerate(class_names):
        class_names[kc] = class_[n_letters:]
    
    class_names = sorted(class_names)
    
    class_filenames = {}
    for class_ in class_names:
        class_filenames[class_] = glob('{}/{}/*'.format(base_dir,class_))
    
    for class_ in class_names:
        for kf, fname in enumerate(class_filenames[class_]):
            class_filenames[class_][kf] = fname.replace('{}/{}/'.format(base_dir,class_),'')
    
    class_filenames_train = {}
    class_filenames_val = {}
    class_filenames_test = {}
    
    for class_ in class_names:
        files_train_val, files_test = train_test_split(class_filenames[class_], test_size=0.1)
        files_train, files_val = train_test_split(files_train_val, test_size=0.2)
        class_filenames_train[class_] = files_train
        class_filenames_val[class_] = files_val
        class_filenames_test[class_] = files_test
    
    n_train_files = sum([len(class_filenames_train[class_]) for class_ in class_names])
    n_test_files = sum([len(class_filenames_test[class_]) for class_ in class_names])
    n_val_files = sum([len(class_filenames_val[class_]) for class_ in class_names])
    
    print()
    print(subdir)
    for class_ in class_names: 
        print('{:22} {:.1f}%'.format(class_, len(class_filenames_train[class_]) / n_train_files*100))