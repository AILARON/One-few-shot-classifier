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

from tqdm import tqdm as tqdm
from os import walk
import os
import numpy as np
import shutil
import math

import config

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

def getImageSplit(number_of_examples, train_percent=config.train_percent,
                    val_percent=config.val_percent,test_percent=config.test_percent):
    train_split = int(number_of_examples*train_percent)
    val_split = int(number_of_examples*val_percent)
    test_split = int(number_of_examples*test_percent)
    rem = number_of_examples-train_split-val_split-test_split
    train_split += math.ceil(rem/2)
    test_split += rem-math.ceil(rem/2)
    return train_split, val_split, test_split


DATA_PATH = 'data2'
# Clean up folders
rmdir(DATA_PATH + '/kaggle/traditional_images_background')
rmdir(DATA_PATH + '/kaggle/traditional_images_evaluation')
mkdir(DATA_PATH + '/kaggle/traditional_images_background')
mkdir(DATA_PATH + '/kaggle/traditional_images_evaluation')

species = []
for _, folders, _ in walk(DATA_PATH + '/kaggle/images'):
    for f in folders:
        species.append(f)

images = []
for i in range(len(species)):
    images.append([])
    for _,_,files in walk(DATA_PATH + '/kaggle/images/' + species[i]):
        #print(folders)
        for f in files:
            #print(f)
            images[i].append(f)

image_split_list = []
for specie in images:
    train_split, _, _ = getImageSplit(len(specie))
    image_split_list.append(train_split)

for folderName in species:
    mkdir(DATA_PATH + '/kaggle/traditional_images_background/' + folderName)
for folderName in species:
    mkdir(DATA_PATH + '/kaggle/traditional_images_evaluation/' + folderName)

print('Preparing background data....')
for i in tqdm(range(len(images))):
    folder = species[i]
    for j in range(len(images[i][:image_split_list[i]])):
        src = DATA_PATH + '/kaggle/images/' + folder + '/' + images[i][j]
        dst = dst = DATA_PATH + '/kaggle/traditional_images_background/' + folder + '/'# + images[i][j]
        shutil.copy(src,dst)

print('Preparing evaluation data....')
for i in tqdm(range(len(images))):
    folder = species[i]
    for j in range(len(images[i][image_split_list[i]:])):
        src = DATA_PATH + '/kaggle/images/' + folder + '/' + images[i][j]
        dst = dst = DATA_PATH + '/kaggle/traditional_images_evaluation/' + folder + '/' + images[i][j]
        shutil.copy(src,dst)
'''
np.random.seed(0)
np.random.shuffle(classes)
background_classes, evaluation_classes = classes[:83], classes[83:]

print('Preparing background_data....')
for i in tqdm(range(len(background_classes))):
    folder = background_classes[i]
    src = DATA_PATH + '/whoas/images/' + folder
    dst = DATA_PATH + '/whoas/traditional_images_background/' + folder
    copy_tree(src,dst)

print('Preparing evaluation_data....')
for i in tqdm(range(len(evaluation_classes))):
    folder = evaluation_classes[i]
    src = DATA_PATH + '/whoas/images/' + folder
    dst = DATA_PATH + '/whoas/traiditional_images_evaluation/' + folder
    copy_tree(src,dst)
'''
