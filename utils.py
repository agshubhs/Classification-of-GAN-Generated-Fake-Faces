import numpy as np 
import torch

def load_filepath_and_identity(filename,split="|"):
	with open(filename, encoding='utf-8') as f:
		filepath_and_identity=[line.strip().split(split) for line in f]

	return filepath_and_identity

# def load_image_to_torch(full_path):
	
def to_gpu(x):
	x=x.contiguous()

	if torch.cuda.is_available():
		x=x.cuda(non_blocking=True)

	return torch.autograd.Variable(x)
