from glob import glob
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import numpy as np
import os

class_names = glob('*')

class_filenames = {}
for class_ in class_names:
    class_filenames[class_] = glob('{}/*'.format(class_))

class_filenames_train = {}
class_filenames_val = {}
class_filenames_test = {}

for class_ in class_names:
    files_train_val, files_test = train_test_split(class_filenames[class_], test_size=0.1)
    files_train, files_val = train_test_split(files_train_val, test_size=0.2)
    class_filenames_train[class_] = files_train
    class_filenames_val[class_] = files_val
    class_filenames_test[class_] = files_test

for class_ in class_names:
    print('{:22} {:6} {}'.format(class_, 'train', len(class_filenames_train[class_])))
    print('{:22} {:6} {}'.format(class_, 'val', len(class_filenames_val[class_])))
    print('{:22} {:6} {}'.format(class_, 'test', len(class_filenames_test[class_])))

sub_directories = ['train', 'test', 'validation']
class_fname_list = [class_filenames_train, class_filenames_test, class_filenames_val]

for class_ in tqdm(class_names):
    for new_dir, idx_class in zip(sub_directories, class_fname_list):
        if not os.path.exists(new_dir): os.mkdir(new_dir)
        if not os.path.exists('{}/{}'.format(new_dir,class_)): os.mkdir('{}/{}'.format(new_dir,class_))
        for fname in tqdm(idx_class[class_]):
            os.rename(fname, '{}/{}'.format(new_dir, fname))

for class_ in tqdm(class_names):
    os.removedirs(class_)
