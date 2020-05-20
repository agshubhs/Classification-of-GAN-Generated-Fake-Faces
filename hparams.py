import tensorflow as tf

def create_hparams(hparam_string=None,verbose=None):
	"""Create model hyperparametes. PArse non default from the string"""
	hparams=tf.contrib.training.HParams(
		epochs=100,
		iters_per_checkpoint=200,
		seed=1234,
		dynamic_loss_scaling=True,
		fp16_run=False,
		dist_backend="tcp://localhsot:54321",
		cudnn_enabled=True,
		cudnn_benchmark=False,
		ignore_layers=['embedding.weight'],

		# data Parameters
		training_files='filelists/gan_images_train_list.txt',
		validation_files='filelists/gan_images_val_list.txt',

		# Optimization Hyperparameters
		num_labels=4,
		used_aved_learning_rate=False,
		learning_rate=1e-5,
		weight_decay=1e-7,
		grad_clip_thresh=1.0,
		batch_size=16,
		mask_padding=True,
		distributed_run=False		
		)

	if hparam_string:
		tf.logging.info('Parsing command line haparams %s',hparams.values())


	return hparams

