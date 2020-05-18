import os
import time
import argparse
import math
from numpy import finfo

import torch
from torch.utils.data import DataLoader
from torch import nn
from model import GAN_Classifier
from data_utils import TextMeLoader
from logger import GANClassifierlogger
from hparams import create_hparams

def prepare_dataloaders(hparams):
	trainset=TextMeLoader(hparams.training_files,hparams)
	valset=TextMeLoader(hparams.validation_files,hparams)
	#collate_fn

	train_sampler=None
	shuffle=True

	train_loader=DataLoader(trainset, num_workers=0,shuffle=shuffle,
		sampler=train_sampler,batch_size=hparams.batch_size,pin_memory=False,drop_last=True,#colla
		)

	return train_loader	,valset,#collate


def prepare_directories_and_logger(output_directory, log_directory,rank):
	if rank==0 :
		if not 	os.path.isdir(output_directory):
			os.makedirs(output_directory)
			os.chmod(output_directory,0o775)
		logger=GANClassifierlogger(os.path.join(output_directory,log_directory))
	
	else :
		logger=None
	return logger 

def load_model(hparams):
	model= GAN_Classifier(hparams).cuda()
	#fp16 run

	return model

def warm_start_model(checkpoint_path,model,ignore_layers):
	assert os.path.isfile(checkpoint_path)
	print("Warm starting modelfrom checkpoint '{}'".format(checkpoint_path))
	checkpoint_dict=torch.load(checkpoint_path,map_location='cpu')
	model_dict=checkpoint_dict['state_dict']
	if len(ignore_layers) > 0:
        	model_dict = {k: v for k, v in model_dict.items()
                      if k not in ignore_layers}
        	dummy_dict = model.state_dict()
        	dummy_dict.update(model_dict)
        	model_dict = dummy_dict
	model.load_state_dict(model_dict)
	return model

def load_checkpoint(checkpoint_path,model,optimizer):
	assert	os.path.isfile(checkpoint_path)
	print("Loading checkpoint  '{}'".format(checkpoint_path))
	checkpoint_dict=torch.load(checkpoint_path,map_location='cpu')
	model.load_state_dict(checkpoint_dict['state_dict'])
	optimizer.load_state_dict(checkpoint_dict['optimizer'])
	learning_rate=checkpoint_dict['learning rate']
	iteration=checkpoint_dict['iteration']
	print("Loaded checkpoint '{}' from iteration'{}'".format(checkpoint_path,iteration))

	return model,optimizer,learning_rate, iteration

def save_checkpoint(model,optimizer,learning_rate,iteration,filepath):
	print("Saving model and optimizer state at iteration {} to {}".format(iteration,filepath))
	torch.save({'iteration':iteration,
				'state_dict':model.state_dict(),
				'optimizer':optimizer.state_dict(),
				'learning_rate':learning_rate},filepath)

def validate(model,criterion, valset,iteration,batch_size,n_gpus,logger,rank):
	model.eval()
	with torch.no_grad():
		val_sampler=None
		val_loader=DataLoader(valset,sampler=val_sampler,num_workers=0,
								shuffle=False,batch_size=batch_size,pin_memory=False,#collate_fn=collate_fn
								)

		val_loss=0.0
		accuracy=0.0
		total=0
		for i, batch in enumerate(val_loader):
			x,y=model.parse_batch(batch)
			y_pred=model(x)
			_,preds=torch.max(y_pred,1)
			#print(y)
			#print(y_pred)
			#print(preds)
			loss=criterion(y_pred,y)
			reduced_val_loss=loss.item()
			val_loss +=reduced_val_loss
			accuracy +=torch.sum(preds==y)
			total +=len(preds)
			#print(total)
			#print(accuracy)
			#print(len(val_loader))

		val_loss=val_loss/(i+1)
		
		accuracy=accuracy/total

	model.train()
	if rank ==0 :
		print("Validation loss {} :{:9f} , Accuracy:{:9f} :".format(iteration,val_loss,accuracy))
		logger.log_validation(val_loss,model,y,y_pred,iteration,accuracy)


def train(output_directory,log_directory,checkpoint_path,warm_start,n_gpus,rank,group_name,hparams):
	torch.manual_seed(hparams.seed)
	torch.cuda.manual_seed(hparams.seed)

	model=load_model(hparams)
	learning_rate =hparams.learning_rate
	optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate,weight_decay=hparams.weight_decay)

	criterion=nn.CrossEntropyLoss()

	logger= prepare_directories_and_logger(output_directory,log_directory,rank)
	##collate
	train_loader,valset=prepare_dataloaders(hparams)

	iteration=0;
	epoch_offset=0;
	if checkpoint_path is not None:
		if warm_start :
			model=warm_start_model(checkpoint_path,model,hparams.ignore_layers)

		else:
			model,optimizer,learning_rate,iteration=load_checkpoint(checkpoint_path,model,optimizer)

			if hparams.use_saved_learning_rate:
				learning_rate=_learning_rate
			iteration +=1
			epoch_offset=max(0,int(iteration/len(train_loader)))

	model.train()
	is_overflow= False

	for epoch in range(epoch_offset,hparams.epochs):
		print("Epoch:{}".format(epoch))
		for i ,batch in enumerate(train_loader):
			#print(batch)
			start= time.perf_counter()
			for param_group in optimizer.param_groups:
				param_group['lr']=learning_rate
			model.zero_grad()
			x,y=model.parse_batch(batch)
			y_pred=model(x)
			#print('ypred',y_pred)
			#print('y',y)
			#print(y.type())
			#print(y_pred.type())
			loss=criterion(y_pred,y)
			reduced_loss=loss.item()

			loss.backward()
			grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), hparams.grad_clip_thresh)

			optimizer.step()

			if not is_overflow and  rank==0 :
				duration=time.perf_counter()-start
				print("Train Loss {} {:.6f} Grad Norm {:.6f} {:.2f}s/it".format(iteration,reduced_loss,grad_norm,duration))
				logger.log_training(reduced_loss,grad_norm,learning_rate,duration,iteration)

			if not is_overflow and (iteration % hparams.iters_per_checkpoint==0):
				validate(model,criterion,valset,iteration,hparams.batch_size,
					n_gpus,logger,rank)

				if rank==0 :
					checkpoint_path	=os.path.join(output_directory,"checkpoint_{}".format(iteration))
					save_checkpoint(model,optimizer,learning_rate,iteration,checkpoint_path)


			iteration +=1


if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output_directory', type=str,help='directory to save checkpoints')
	parser.add_argument('-l', '--log_directory', type=str,
                        help='directory to save tensorboard logs')
	parser.add_argument('-c', '--checkpoint_path', type=str, default=None,
                        required=False, help='checkpoint path')
	parser.add_argument('--warm_start', action='store_true',
                        help='load model weights only, ignore specified layers')
	parser.add_argument('--n_gpus', type=int, default=1,
                        required=False, help='number of gpus')
	parser.add_argument('--rank', type=int, default=0,
                        required=False, help='rank of current gpu')
	parser.add_argument('--group_name', type=str, default='group_name',
                        required=False, help='Distributed group name')
	parser.add_argument('--hparams', type=str,
                        required=False, help='comma separated name=value pairs')
	args = parser.parse_args()
	hparams = create_hparams(args.hparams)
	torch.backends.cudnn.enabled = hparams.cudnn_enabled
	torch.backends.cudnn.benchmark = hparams.cudnn_benchmark
	print("FP16 Run:", hparams.fp16_run)
	print("Dynamic Loss Scaling:", hparams.dynamic_loss_scaling)
	print("Distributed Run:", hparams.distributed_run)
	print("cuDNN Enabled:", hparams.cudnn_enabled)
	print("cuDNN Benchmark:", hparams.cudnn_benchmark)
	train(args.output_directory, args.log_directory, args.checkpoint_path,
          args.warm_start, args.n_gpus, args.rank, args.group_name, hparams)
