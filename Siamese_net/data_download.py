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

def downloadOmniglot(train = True):

	dataset = torchvision.datasets.Omniglot(root = "./data",
		background = train,
		download = True,
		transform = transforms.ToTensor())

#downloadOmniglot()
