import random
import numpy as numpy
import torch
import torch.utils.data
from PIL import Image
import layers
import torchvision.transforms as transforms
from utils import load_filepath_and_identity
from hparams import create_hparams
import argparse

class TextMeLoader(torch.utils.data.Dataset):
	""" Load image and corresponding type splti by |"""

	def __init__(self,imagepath_and_type,hparams):
		self.imagepath_and_type=load_filepath_and_identity(imagepath_and_type)
		#self.transform=transform
		random.seed(1234)
		random.shuffle(self.imagepath_and_type)


	def get_image_type_pair(self, imagepath_and_type):
		image,srm_filter,identity=imagepath_and_type[0],imagepath_and_type[1],int(imagepath_and_type[2])
		image=self.get_image(image)
		identity=self.get_identity(identity)
		# return (image, srm_filter,identity)
		return image,identity

	def get_image(self,img):
		transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485,0.456,0.406],std= [0.229,0.224,0.225])])
		image=Image.open(img)
		image=transform(image)
		return image
	def get_identity(self,idx):
		return torch.tensor(idx)




	def __getitem__(self,index):
		return self.get_image_type_pair(self.imagepath_and_type[index])

	def __len__(self):
		return len(self.imagepath_and_type)


