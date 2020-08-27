#################################################################################################################
# The Reporting tool from neptus logs
# Implementation of the dune proc
# Author: Andreas Våge
# email: andreas.vage@ntnu.no
#
# Date created: June 2020
#
# Project: AILARON
# Contact
# email: annette.stahl@ntnu.no
# funded by RCN IKTPLUSS program (project number 262701) and supported by NTNU AMOS
# Copyright @NTNU 2020
#######################################

from distutils.dir_util import copy_tree
from tqdm import tqdm as tqdm
from os import walk
import numpy as np
import shutil

def mkdir(dir):
    """Create a directory, ignoring exceptions

    # Arguments:
        dir: Path of directory to create
    """
    try:
        os.mkdir(dir)
    except:
        pass


def rmdir(dir):
    """Recursively remove a directory and contents, ignoring exceptions

   # Arguments:
       dir: Path of directory to recursively remove
   """
    try:
        shutil.rmtree(dir)
    except:
        pass


DATA_PATH = 'data2'
# Clean up folders
rmdir(DATA_PATH + '/whoas/images_background')
rmdir(DATA_PATH + '/whoas/images_evaluation')
mkdir(DATA_PATH + '/whoas/images_background')
mkdir(DATA_PATH + '/whoas/images_evaluation')

classes = []
for _, folders, _ in walk(DATA_PATH + '/whoas/images'):
    for f in folders:
        classes.append(f)


np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:83], classes[83:]

print('Preparing background_data....')
for i in tqdm(range(len(background_classes))):
    folder = background_classes[i]
    src = DATA_PATH + '/whoas/images/' + folder
    dst = DATA_PATH + '/whoas/images_background/' + folder
    copy_tree(src,dst)

print('Preparing evaluation_data....')
for i in tqdm(range(len(evaluation_classes))):
    folder = evaluation_classes[i]
    src = DATA_PATH + '/whoas/images/' + folder
    dst = DATA_PATH + '/whoas/images_evaluation/' + folder
    copy_tree(src,dst)
