from glob import glob
from argparse import ArgumentParser

ap = ArgumentParser()
ap.add_argument("-d", "--directory", type=str, required=False, default='dataset', help="the directory to check subdirectory populations")
args = vars(ap.parse_args())

args["directory"] = args["directory"] + '/' if args["directory"][-1] is not '/' else args["directory"]

flagnames = [fdirnow.replace(args["directory"],'') for fdirnow in glob(args["directory"] + '/*')]
jpg_names = {}

for flgnm in flagnames:
    jpg_names[flgnm] = glob(args["directory"] + flgnm + '/*jpg')
    
for flgnm in jpg_names.keys():
    print(flgnm, len(jpg_names[flgnm]))