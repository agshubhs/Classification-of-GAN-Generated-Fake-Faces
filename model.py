from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import LinearNorm, ConvNorm
from torchvision import models
from utils import to_gpu 








class GAN_Classifier(nn.Module):
	def __init__(self,hparams):
		super(GAN_Classifier,self).__init__()
		num_labels=hparams.num_labels
    		#For Densenet
		self.model = models.densenet161(pretrained=True)
		for param in self.model.parameters():
			param.requires_grad = False
		classifier_input=self.model.classifier.in_features
		classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
		self.model.classifier = classifier

		#For resnet
		'''
		self.model=models.resnet101(pretrained=True)
		for param in self.model.parameters():
			param.requires_grad = False
		classifier_input=2048
		classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
		self.model.fc = classifier
    		'''
    		#FOR VGG
		'''
		self.model = models.vgg19(pretrained=True)
		for param in self.model.parameters():
			param.requires_grad = False
		classifier_input=25088

		classifier = nn.Sequential(nn.Linear(classifier_input, 1024),
                           nn.ReLU(),
                           nn.Linear(1024, 512),
                           nn.ReLU(),
                           nn.Linear(512, num_labels),
                           nn.LogSoftmax(dim=1))
		self.model.classifier = classifier
		'''

	def parse_batch(self, batch):
		a,b=batch
		a=to_gpu(a).float()
		b=b.cuda()
		return (a,b)

	def forward(self,inputs):
		out=self.model(inputs)
		return (out)

	def inference(self,inputs):
		out=self.model(inputs)
		return out
