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
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import config

class Flatten(nn.Module):
	def forward(self, x):
		N, C, H, W = x.size()
		return x.view(N, -1)

class SiameseNet(nn.Module):
	"""docstring for SiameseNet"""
	def __init__(self):
		super(SiameseNet, self).__init__()
		self.siamese_twin = nn.Sequential(
			nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 10),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 7),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 128, out_channels = 128, kernel_size = 4),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2, stride = 2),
			nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 4),
			nn.ReLU(),
			Flatten(),
			nn.Linear(in_features = 256 * 6 * 6 , out_features = 4096),
			nn.Sigmoid()
			)
		self.final_fc = nn.Linear(in_features = 4096 , out_features = 1)

	def forward(self, x, y):
		x1 = self.siamese_twin(x)
		y1 = self.siamese_twin(y)
		L1 = torch.abs(x1 - y1)
		output = F.sigmoid(self.final_fc(L1))

		return output


class SiameseResNet(nn.Module):
	def __init__(self):
		super(SiameseResNet, self).__init__()
		if config.active_model_name == 'resnet18':
			self.siamese_twin = nn.Sequential(models.resnet18(), nn.Sigmoid())
		elif config.active_model_name == 'resnet34':
			self.siamese_twin = nn.Sequential(models.resnet34(), nn.Sigmoid())
		elif config.active_model_name == 'resnet50':
			self.siamese_twin = nn.Sequential(models.resnet50(), nn.Sigmoid())
		elif config.active_model_name == 'resnet101':
			self.siamese_twin = nn.Sequential(models.resnet101(), nn.Sigmoid())
		elif config.active_model_name == 'resnet152':
			self.siamese_twin = nn.Sequential(models.resnet152(), nn.Sigmoid())
		else:
			error = 'ERROR: Requested ResNet model not found'
			print(error)
			config.error_message += error + '\n'

		self.final_fc = nn.Linear(in_features = 1000, out_features = 1)

	def forward(self, x, y):
		x1 = self.siamese_twin(x)
		y1 = self.siamese_twin(y)
		L1 = torch.abs(x1 - y1)
		output = torch.sigmoid(self.final_fc(L1))

		return output


class SiameseVGGNet(nn.Module):
	def __init__(self):
		super(SiameseVGGNet, self).__init__()
		if config.active_model_name == 'vgg11_bn':
			self.siamese_twin = nn.Sequential(models.vgg11_bn(), nn.Sigmoid())
		elif config.active_model_name == 'vgg16_bn':
			self.siamese_twin = nn.Sequential(models.vgg16_bn(), nn.Sigmoid())
		elif config.active_model_name == 'vgg19_bn':
			self.siamese_twin = nn.Sequential(models.vgg19_bn(), nn.Sigmoid())
		else:
			error = 'ERROR: Requested VGG model not found'
			print(error)
			config.error_message += error + '\n'

		self.final_fc = nn.Linear(in_features = 1000, out_features = 1)

	def forward(self, x, y):
		x1 = self.siamese_twin(x)
		y1 = self.siamese_twin(y)
		L1 = torch.abs(x1 - y1)
		output = torch.sigmoid(self.final_fc(L1))

		return output


class SiameseCustomVGG11Net(nn.Module):
  def __init__(self):
    super(SiameseCustomVGG11Net, self).__init__()
    self.siamese_twin = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
      nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
      nn.ReLU(inplace=True),
      Flatten(),
      nn.Linear(in_features = 512 *4 *4, out_features = 4096),
      nn.Sigmoid()
    )
    self.final_fc = nn.Linear(in_features = 4096, out_features = 1)

  def forward(self, x, y):
    x1 = self.siamese_twin(x)
    y1 = self.siamese_twin(y)
    L1 = torch.abs(x1 - y1)
    output = F.sigmoid(self.final_fc(L1))

    return output


class SiameseCustomRes18Net(nn.Module):
	def __init__(self):
		super(SiameseCustomRes18Net, self).__init__()

		customResNet18 = models.resnet18()
		customResNet18.avgpool = Flatten()
		customResNet18.fc = nn.Linear(in_features=512*int(config.image_size/32)*int(config.image_size/32), out_features=4096)

		self.siamese_twin = nn.Sequential(customResNet18, nn.Sigmoid())
		self.final_fc = nn.Linear(in_features = 4096 , out_features = 1)

	def forward(self, x, y):
		x1 = self.siamese_twin(x)
		y1 = self.siamese_twin(y)
		L1 = torch.abs(x1 - y1)
		output = F.sigmoid(self.final_fc(L1))

		return output


class SiameseCustom2Res18Net(nn.Module):
	def __init__(self):
		super(SiameseCustom2Res18Net, self).__init__()

		customResNet18 = models.resnet18()
		customResNet18.fc = nn.Linear(in_features=512, out_features=4096)

		self.siamese_twin = nn.Sequential(customResNet18, nn.Sigmoid())
		self.final_fc = nn.Linear(in_features = 4096 , out_features = 1)

	def forward(self, x, y):
		x1 = self.siamese_twin(x)
		y1 = self.siamese_twin(y)
		L1 = torch.abs(x1 - y1)
		output = F.sigmoid(self.final_fc(L1))

		return output



class PlanktonNet(nn.Module):

	def __init__(self):
		super(PlanktonNet, self).__init__()
		self.features = nn.Sequential(
			# Pytorch output shapes from Conv2D
			# Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
			# Wout = (Win +2xpadding(1)-dilation(1)x(kernel_size(1)-1)-1)/stride(1) + 1
			# MaxPool2D
			# Hout = (Hin +2xpadding(0)-dilation(0)x(kernel_size(0)-1)-1)/stride(0) + 1
			# Wout = (Win + 2xpadding(1) - dilation(1)x(kernel_size(1) - 1) - 1) / stride(1) + 1
			# -> default padding is 0, default stride = kernel_size dilation=1
			nn.Conv2d(3, 96, kernel_size=13, stride=1, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			#nn.LocalResponseNorm(128),
			nn.Conv2d(96, 256, kernel_size=7, padding=2),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(256, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(384, 384, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
			nn.Conv2d(384, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2),
		)
        #self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512*2*2, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, 4096),
			nn.Sigmoid(),
		)
		self.final_fc = nn.Linear(in_features = 4096 , out_features = 1)
	def forward(self, x,y):
		x1 = self.features(x)
		x1 = x1.view(-1, 512*2*2)
		x1 = self.classifier(x1)
		y1 = self.features(y)
		y1 = y1.view(-1, 512*2*2)
		y1 = self.classifier(y1)
		L1 = torch.abs(x1 - y1)
		output = F.sigmoid(self.final_fc(L1))

		#x = self.avgpool(x)
		#x = torch.flatten(x, 1)
		#x = self.classifier(x)
		return output
