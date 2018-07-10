from argparse import ArgumentParser
from glob     import glob
from numpy    import random
from os       import environ, path
from shutil   import copyfile
from tqdm     import tqdm

def_in_files_dir    = environ['HOME'] + '/Research/STScI/NIRISS/QuickLook/WFC3_QL_DLN/keras_dln/dataset_all/clean/'
def_out_files_dir   = environ['HOME'] + '/Research/STScI/NIRISS/QuickLook/WFC3_QL_DLN/keras_dln/dataset/clean/'

# construct the argument parse and parse the arguments
ap = ArgumentParser()

ap.add_argument("-n", "--num_to_copy" , type=int , default=10000            , required=False, help="Number of files to copy")
ap.add_argument("-i", "--input_dir"   , type=str , default=def_in_files_dir , required=False, help="Directory to find the JPG files")
ap.add_argument("-o", "--output_dir"  , type=str , default=def_out_files_dir, required=False, help="Directory to put  the JPG files")
ap.add_argument("-r", "--with_replace", type=bool, default=False            , required=False, help="Whether to select 10k unique files or 10k filenames (some overlap)")

args = vars(ap.parse_args())

num_use       = args["num_to_copy"]
in_files_dir  = args["input_dir"] 
out_files_dir = args["output_dir"]
print(in_files_dir, out_files_dir)
clean_filenames = glob(in_files_dir + '/*jpg')

files_use = random.choice(clean_filenames, size=num_use, replace=args["with_replace"])

for fname in tqdm(set(files_use), total=len(files_use)):
    if path.exists(fname):
        copyfile(fname, fname.replace(in_files_dir, out_files_dir))
