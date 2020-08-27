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


import torch
import torchvision
import torchvision.transforms as transforms
import glob
import math
import random
from PIL import Image
import time
import numpy as np
from os import walk
import cv2
import config



def grayspace2RGBspace(x):
    #RGBimg = np.zeros((3,x.shape[1],x.shape[2]))
    #RGBimg[0,:,:] = x
    #RGBimg[1,:,:] = x
    #RGBimg[2,:,:] = x
    RGBimg = cv2.cvtColor(x, cv2.COLOR_GRAY2RGB)
    RGBimg = np.array(cv2.split(RGBimg))
    #print("DONE!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    #print(RGBimg.shape)
    return RGBimg

def getImages():

    #Image from only the first 12 drawers is taken during train time
    if config.dataset == 'kaggle':
        image_path = "data2/kaggle/images"
    elif config.dataset == 'whoas':
        image_path = "data2/whoas/images"
    elif config.dataset == 'silcam':
        image_path = "data2/silcam_classification_database"
    else:
        print('ERROR: Dataset not found/registered')
    images = []
    image_counter = 0
    i = 0
    for (dirpath, dirnames, filenames) in walk(image_path, topdown=True):
        same_species_images = []
        for image_name in filenames:
            if image_name[-3:] == 'png' or image_name[-3:] == 'jpg':
                same_species_images.append(dirpath + "/" + image_name)
                image_counter += 1
        if len(same_species_images) != 0:
            images.append(same_species_images)
        #print(i)
        #print(dirpath)
        #print("images: ", len(same_species_images))
        #print("---------------------------------------------------")
        i += 1
    return images, image_counter

def getOneShotImages(data_type):
    if config.classification_type == 'traditional':
        folder_type = '/traditional_images_'
    else:
        folder_type = '/images_'
    path = config.DATA_PATH + '/' + config.dataset + folder_type + data_type
    train_images = []
    image_counter = 0
    for (dirpath, dirnames, filenames) in walk(path, topdown=True):
        species_images = []
        for image_name in filenames:
            if image_name[-3:] == 'png' or image_name[-3:] == 'jpg':
                species_images.append(dirpath + '/' + image_name)
                image_counter += 1
        if len(species_images) != 0:
            train_images.append(species_images)

    return train_images, image_counter

def getRandomPairedIndices(no, start, end):
    count = 0
    pairs = []
    pair = []
    while (count < no):
        #print("no: ", no)
        #print("start: ", start)
        #print("end: ", end)
        #print("len(pairs)): ", len(pairs))
        pair = []
        x, y = random.randint(start, end), random.randint(start, end)
        pair.append(x)
        pair.append(y)
        if x==y:
            continue
        if x>y:
            x, y = y, x
        if pair in pairs:
            continue
        pairs.append(pair)
        count+=1;
    return pairs

def getImageSplit(number_of_examples, source, train_percent=config.train_percent,
                    val_percent=config.val_percent,test_percent=config.test_percent):

    if config.classification_type == 'one_few_shot' and source == 'train':
        train_percent, val_percent, test_percent = 1, 0, 0
    elif config.classification_type == 'one_few_shot' and source == 'val':
        train_percent, val_percent, test_percent = 0, 1, 0
    elif config.classification_type == 'one_few_shot':
        print('Source: ' + str(source) + ' not recognized')

    train_split = int(number_of_examples*train_percent)
    val_split = int(number_of_examples*val_percent)
    test_split = int(number_of_examples*test_percent)
    rem = number_of_examples-train_split-val_split-test_split
    train_split += math.ceil(rem/2)
    test_split += rem-math.ceil(rem/2)
    return train_split, val_split, test_split

def getTrainingExamples(all_images, num_images, n = 30000):

    number_of_species = len(all_images)

    # For same characters
    n_same = n / 2
    same_images_per_species = math.ceil(n_same / number_of_species)
    n_same = same_images_per_species * number_of_species
    same_species_examples = []
    species_index = 0
    pair = []
    print_every = 1000
    trans = transforms.ToTensor()

#     If there is not enough data in one species to produce the required amount
#     of data, then those extra examples are spread uniformly to the rest
    max_examples = []
    examples_per_species = []
    leftover = 0
    print('Calculating number of examples per species.......')
    for i in range(number_of_species):
        num_same_train_images_of_species, _, _ = getImageSplit(len(all_images[i]), 'train')
        max_same_train_examples_for_specie = num_same_train_images_of_species*(num_same_train_images_of_species-1)
        max_examples.append(max_same_train_examples_for_specie)
        if same_images_per_species - max_same_train_examples_for_specie > 0:
            leftover += same_images_per_species - max_same_train_examples_for_specie
            examples_per_specie = max_same_train_examples_for_specie
        else:
            examples_per_specie = same_images_per_species
        examples_per_species.append(examples_per_specie)

    i = 0
    while leftover > 0:
        if max_examples[i] > same_images_per_species:
            examples_per_species[i] += 1
            leftover -= 1
        i += 1
        if i >= len(max_examples):
            i = 0
            same_images_per_species += 1
    print('-> Done')

#     We need to get n / number_of_char pairs from a single character (12 drawers)
#     So we will get n / number_of_char random pairs of indexes from 0-11
    count = 0
    print('Creating training examples ......')
    for i in range(0, number_of_species):
        #print("Species number: ", i+1)
        train_split, _, _ = getImageSplit(len(all_images[i]), 'train')
        #print('train_split: ', train_split)
        #print(examples_per_species)
        #print("SUUUUUUM: ", sum(examples_per_species))
        indices = getRandomPairedIndices(examples_per_species[i], 0, train_split-1)
        j=0
        for index in indices:
            if(count % print_every == 0):
              print("Creating same examples.... Complete: %d examples"%(count))
            #print(j)
            #print(index)
            j+=1
            pair = []
            img1 = Image.open(all_images[i][index[0]])
            img1 = img1.resize((config.image_size,config.image_size))
            #print("SHAPE1: ", np.array(img1).shape)
            img1 = np.array(img1)
            if config.active_model_name == 'default':
                img1 = img1.reshape((1, config.image_size,config.image_size))
            elif img1.shape != (config.image_size,config.image_size, 3):# and config.active_model_name != 'default':
                img1 = grayspace2RGBspace(np.array(img1))
            #img1 = trans(img1).numpy()
            #if (img1.shape == (1,64,64)):
            #    img1 = grayspace2RGBspace(img1)
            img2 = Image.open(all_images[i][index[1]])
            img2 = img2.resize((config.image_size,config.image_size))
            img2 = np.array(img2)
            if config.active_model_name == 'default':
                img2 = img2.reshape((1, config.image_size,config.image_size))
            elif img2.shape != (config.image_size,config.image_size, 3) and config.active_model_name != 'default':
                img2 = grayspace2RGBspace(np.array(img2))
            #img2 = trans(img2).numpy()
            #if (img2.shape == (1,64,64)):
            #    img2 = grayspace2RGBspace(img2)
            pair.append(img1)
            pair.append(img2)
            same_species_examples.append(pair)
            count += 1
        #print("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print("Done with species nr: ", i)
        #print('Number of species total: ', number_of_species)

#    For different characters
    n_diff = n_same
    count = 0
    diff_species_examples = []
    while(count<n_diff):
        if(count % print_every == 0):
          print("Creating different examples.... Complete: %d examples"%(count))
        #print("count %d"%(count))

        pair = []
        i11,i12,i21,i22 = 0,0,0,0
        #Make sure no images are from the same species
        while i11 == i21:
            i11 = random.randint(0,len(all_images)-1)
            i21 = random.randint(0,len(all_images)-1)
        train_split1, _, _ = getImageSplit(len(all_images[i11]), 'train')
        train_split2, _, _ = getImageSplit(len(all_images[i21]), 'train')
        i12 = random.randint(0,train_split1-1)
        i22 = random.randint(0,train_split2-1)
        img1 = Image.open(all_images[i11][i12])
        img1 = img1.resize((config.image_size,config.image_size))
        img1 = np.array(img1)
        if config.active_model_name == 'default':
            img1 = img1.reshape((1, config.image_size,config.image_size))
        elif img1.shape != (config.image_size,config.image_size, 3) and config.active_model_name != 'default':
            img1 = grayspace2RGBspace(np.array(img1))
        #print(img1.shape)
        #img1 = trans(img1).numpy()
        #if (img1.shape == (1,64,64)):
        #    img1 = grayspace2RGBspace(img1)
        img2 = Image.open(all_images[i21][i22])
        img2 = img2.resize((config.image_size,config.image_size))
        img2 = np.array(img2)
        if config.active_model_name == 'default':
            img2 = img2.reshape((1, config.image_size,config.image_size))
        elif img2.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
            img2 = grayspace2RGBspace(np.array(img2))
        #img2 = trans(img2).numpy()
        #if (img2.shape == (1,64,64)):
        #    img2 = grayspace2RGBspace(img2)
        pair.append(img1)
        pair.append(img2)
        diff_species_examples.append(pair)
        count+=1

    random.shuffle(same_species_examples)
    random.shuffle(diff_species_examples)

    return same_species_examples,diff_species_examples

def getValExamples(all_images, num_images, n = 3000):

    number_of_species = len(all_images)

    # For same characters
    n_same = n / 2
    same_images_per_species = math.ceil(n_same / number_of_species)
    n_same = same_images_per_species * number_of_species
    same_species_examples = []
    species_index = 0
    pair = []
    print_every = 1000
    trans = transforms.ToTensor()

#     If there is not enough data in one species to produce the required amount
#     of data, then those extra examples are spread uniformly to the rest
    max_examples = []
    examples_per_species = []
    leftover = 0

    print('Creating validation examples ......')
    for i in range(number_of_species):
        _, num_same_val_images_of_species, _ = getImageSplit(len(all_images[i]), 'val')
        max_same_val_examples_for_specie = num_same_val_images_of_species*(num_same_val_images_of_species-1)
        max_examples.append(max_same_val_examples_for_specie)
        if same_images_per_species - max_same_val_examples_for_specie > 0:
            leftover += same_images_per_species - max_same_val_examples_for_specie
            examples_per_specie = max_same_val_examples_for_specie
        else:
            examples_per_specie = same_images_per_species
        examples_per_species.append(examples_per_specie)

    i = 0
    while leftover > 0:
        #print(i)
        if max_examples[i] > same_images_per_species:
            examples_per_species[i] += 1
            leftover -= 1
        i += 1
        if i >= len(max_examples):
            i = 0
            same_images_per_species += 1

#     We need to get n / number_of_char pairs from a single character (12 drawers)
#     So we will get n / number_of_char random pairs of indexes from 0-11
    for i in range(0, number_of_species):
        #print("Species number: ", i+1)
        train_split, val_split, _ = getImageSplit(len(all_images[i]), 'val')
        #print('train_split: ', train_split)
        #print(examples_per_species)
        #print("SUUUUUUM: ", sum(examples_per_species))
        indices = getRandomPairedIndices(examples_per_species[i], train_split, train_split+val_split-1)
        j=0
        for index in indices:
            #print(j)
            #print(index)
            j+=1
            pair = []
            img1 = Image.open(all_images[i][index[0]])
            img1 = img1.resize((config.image_size,config.image_size))
            img1 = np.array(img1)
            if config.active_model_name == 'default':
                img1 = img1.reshape((1, config.image_size,config.image_size))
            elif img1.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
                img1 = grayspace2RGBspace(np.array(img1))
            #img1 = trans(img1).numpy()
            #if (img1.shape == (1,64,64)):
            #    img1 = grayspace2RGBspace(img1)
            img2 = Image.open(all_images[i][index[1]])
            img2 = img2.resize((config.image_size,config.image_size))
            img2 = np.array(img2)
            if config.active_model_name == 'default':
                img2 = img2.reshape((1, config.image_size,config.image_size))
            elif img2.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
                img2 = grayspace2RGBspace(np.array(img2))
            #img2 = trans(img2).numpy()
            #if (img2.shape == (1,64,64)):
            #    img2 = grayspace2RGBspace(img2)
            pair.append(img1)
            pair.append(img2)
            same_species_examples.append(pair)
        #print("done!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #print("Done with species nr: ", i)
        #print('Number of species total: ', number_of_species)

#    For different characters
    n_diff = n_same
    count = 0
    diff_species_examples = []
    while(count<n_diff):
        pair = []
        #print("count %d"%(count))
        i11,i12,i21,i22 = 0,0,0,0
        #Make sure no images are from the same species
        while i11 == i21:
            i11 = random.randint(0,len(all_images)-1)
            i21 = random.randint(0,len(all_images)-1)
        train_split1, val_split1, _ = getImageSplit(len(all_images[i11]), 'val')
        train_split2, val_split2, _ = getImageSplit(len(all_images[i21]), 'val')
        #print('train_split1: ', train_split1)
        #print('val_split1 : ', val_split1)
        #print('train_split2: ', train_split2)
        #print('val_split2 : ', val_split2)
        if val_split1 == 0 or val_split2 == 0:
            continue
        i12 = random.randint(train_split1,train_split1+val_split1-1)
        i22 = random.randint(train_split2,train_split2+val_split2-1)
        img1 = Image.open(all_images[i11][i12])
        img1 = img1.resize((config.image_size,config.image_size))
        img1 = np.array(img1)
        if config.active_model_name == 'default':
            img1 = img1.reshape((1, config.image_size,config.image_size))
        elif img1.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
            img1 = grayspace2RGBspace(np.array(img1))
        #img1 = trans(img1).numpy()
        #if (img1.shape == (1,64,64)):
        #    img1 = grayspace2RGBspace(img1)
        img2 = Image.open(all_images[i21][i22])
        img2 = img2.resize((config.image_size,config.image_size))
        img2 = np.array(img2)
        if config.active_model_name == 'default':
            img2 = img2.reshape((1, config.image_size,config.image_size))
        elif img2.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
            img2 = grayspace2RGBspace(np.array(img2))
        #img2 = trans(img2).numpy()
        #if (img2.shape == (1,64,64)):
        #    img2 = grayspace2RGBspace(img2)
        pair.append(img1)
        pair.append(img2)
        diff_species_examples.append(pair)
        count+=1

    random.shuffle(same_species_examples)
    random.shuffle(diff_species_examples)

    return same_species_examples,diff_species_examples

def getFewShotValExamples(images, num_images, n=config.validation_examples):
    val_images = []
    for i in range(len(images)-1):
        if len(images[i]) >= config.n_shot + 1:
            val_images.append(images[i])

    print('Testing on remaining: ', len(val_images), ' classes')
    val_examples = []
    val_example_name = []
    tasks = []
    q_images = []
    q_class_nrs = []
    for l in range(n):
        task = []
        used_classes = []
        random_classes = random.sample(range(0, len(val_images)), config.k_way)
        if config.testing_for_unknwon_species == True and (l % 2) == 0:
            q_class = random.choice([x for x in range(len(val_images)) if x not in random_classes])
            q_class_nrs.append(-1)
            q_images.append(val_images[q_class][random.choice(range(0, len(val_images[q_class])))])
        else:
            q_class = random.choice(random_classes)
        for i in range(config.k_way):
            class_image_selection = []
            random_class = random_classes[i]
            image_sample_list = random.sample(range(0,len(val_images[random_class])), config.n_shot+1)
            if q_class == random_classes[i]:
                if config.testing_for_unknwon_species == True and (l % 2) == 0:
                    print('SOMETHING IS WRONG HERE')
                q_class_nrs.append(i)
                q_images.append(val_images[q_class][image_sample_list[config.n_shot]])
            for j in range(config.n_shot):
                class_image_selection.append(val_images[random_class][image_sample_list[j]])
            task.append(class_image_selection)
        tasks.append(task)

    print_every = 100
    for l in range(n):
        if(l % print_every == 0):
            print("Loading data.... Complete: %d examples"%(l))
        task = tasks[l]
        for i in range(config.k_way):
            class_image_selection = task[i]
            for j in range(config.n_shot):
                pair = []
                pair_name = []
                img1 = Image.open(class_image_selection[j])
                img1 = img1.resize((config.image_size,config.image_size))
                img1 = np.array(img1)
                if config.active_model_name == 'default':
                    img1 = img1.reshape((1, config.image_size,config.image_size))
                elif img1.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
                    img1 = grayspace2RGBspace(np.array(img1))
                img2 = Image.open(q_images[l])
                img2 = img2.resize((config.image_size,config.image_size))
                img2 = np.array(img2)
                if config.active_model_name == 'default':
                    img2 = img2.reshape((1, config.image_size,config.image_size))
                elif img2.shape != (config.image_size,config.image_size,3) and config.active_model_name != 'default':
                    img2 = grayspace2RGBspace(np.array(img2))
                pair.append(img1)
                pair.append(img2)
                pair_name.append(class_image_selection[j])
                pair_name.append(q_images[l])
                val_examples.append(pair)
                val_example_name.append(pair_name)
    return val_examples, q_class_nrs, val_example_name

def getTrainBatches(all_images, num_images, n = 30000, batch_size = 128):
    same_species_examples, diff_species_examples = getTrainingExamples(all_images, num_images, n=n)
    train_batches = []
    current_batch = []
    current_batch_size = 0
    examples_covered = 0
    print_every = 1000
    while examples_covered < n:
        if examples_covered % 2 == 0:
            #print("species_examples: ", len(same_species_examples)+len(diff_species_examples))
            current_batch.append(same_species_examples.pop())
        else:
            current_batch.append(diff_species_examples.pop())
        current_batch_size+=1
        if current_batch_size == batch_size:
            #print("current batch size: ", len(current_batch))
            #for i in range(len(current_batch)):
            #    print(len(current_batch[i]))
            #    print(current_batch[i][0].size)
            #print(np.array(current_batch).shape)
            train_batches.append(torch.tensor(current_batch))
            current_batch = []
            current_batch_size = 0
        if(examples_covered % print_every == 0):
          print("Loading data.... Complete: %d examples"%(examples_covered))
        examples_covered+=1
    if current_batch_size!=0:
        train_batches.append(torch.tensor(current_batch))

    return train_batches

def getValBatch(all_images, num_images, n=config.validation_examples):
    same_character_examples, diff_character_examples = getValExamples(all_images, num_images, n=n)
    val_batches = []
    current_batch = []
    current_batch_size = 0
    examples_covered = 0
    while examples_covered < n:
        if examples_covered % 2 == 0:
            current_batch.append(same_character_examples.pop())
        else:
            current_batch.append(diff_character_examples.pop())
        current_batch_size += 1
        if current_batch_size == config.val_batch_size:
            val_batches.append(torch.tensor(current_batch))
            current_batch = []
            current_batch_size = 0
        examples_covered+=1
    if current_batch_size!=0:
        val_batches.append(torch.tensor(current_batch))

    return val_batches


def getFewShotValBatches(val_images, num_images, n=config.validation_examples, batch_size = config.batch_size):
    val_examples, q_class_nrs, val_examples_name = getFewShotValExamples(val_images, num_images, n=n)
    val_batches = []
    val_batches_name = []
    current_batch = []
    current_batch_name = []
    current_batch_size = 0
    examples_covered = 0
    while examples_covered < n:
        for i in range(config.k_way):
            for j in range(config.n_shot):
                current_batch.append(val_examples[examples_covered*config.k_way*config.n_shot:(examples_covered+1)*config.k_way*config.n_shot][i*config.n_shot+j])
                current_batch_name.append(val_examples_name[examples_covered*config.k_way*config.n_shot:(examples_covered+1)*config.k_way*config.n_shot][i*config.n_shot+j])
        current_batch_size += 1
        if current_batch_size == config.val_batch_size:
            val_batches.append(torch.tensor(current_batch))
            val_batches_name.append(current_batch_name)
            current_batch = []
            current_batch_name = []
            current_batch_size = 0
        examples_covered+=1
    if current_batch_size!=0:
        val_batches.append(torch.tensor(current_batch))
        val_batches_name.append(current_batch_name)

    return val_batches, q_class_nrs, val_batches_name
