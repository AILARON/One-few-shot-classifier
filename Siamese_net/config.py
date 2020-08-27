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

models1 = ['vgg11_bn', 'vgg16_bn', 'vgg19_bn',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
             'customVGG11', 'customResNet18']
models = ['default', 'vgg11_bn', 'vgg16_bn', 'vgg19_bn',
            'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
active_model_name = 'default'
classification_type = 'Not registered'
testing_for_unknwon_species = False
training_examples = 30000
validation_examples = 500
train_percent = 0.80
val_percent = 0.20
test_percent = 0
k_way = 1
n_shot = 1
#image_size = 64
image_size = 105
batch_size = 128
val_batch_size = 32
epochs = 20
learning_rate = 0.0005
weight_decay = 0.01
dataset = 'kaggle'
plot_save_file = 'Results'
error_message = ''
DATA_PATH = 'data2'
MODEL_PATH = 'model'
load_model = False
