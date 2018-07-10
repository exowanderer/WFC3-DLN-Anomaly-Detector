# coding: utf-8

from argparse import ArgumentParser
from glob import glob
from numpy import random
from os import environ, remove
from tqdm import tqdm

ap = ArgumentParser()
ap.add_argument('-d', '--directory', type=str, required=False, default=environ['HOME'] + 'Research/STScI/NIRISS/QuickLook/WFC3_QL_DLN/keras_dln/dataset/clean')
ap.add_argument('-n', '--num_keep' , type=int, required=False, default=500)

args = vars(ap.parse_args())

args["directory"] = args["directory"] + '/' if args["directory"][-1] is not '/' else args["directory"]

num_keep = args["num_keep"]

clean_files = glob(args["directory"] + '*jpg')

tb_rm_filenames = random.choice(clean_files, size=len(clean_files)-num_keep, replace=False)

[remove(fname) for fname in tqdm(tb_rm_filenames)]