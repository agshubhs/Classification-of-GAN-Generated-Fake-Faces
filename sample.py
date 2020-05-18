import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

transformations = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datadir= '/mnt/e/B. Tech IV Year/II Semester/BTP/exmp_code/Image-Classification-with-PyTorch/flowers'
train_set = datasets.ImageFolder(datadir+'/train', transform = transformations)
val_set = datasets.ImageFolder(datadir+'/valid', transform = transformations)
print(train_set[1])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)

for batch in train_loader:
	print(batch)
	break