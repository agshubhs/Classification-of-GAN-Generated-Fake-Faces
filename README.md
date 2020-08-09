# Classification of GAN generated Fake Faces

PyTorch implementation for Classification of GAN generated Fake Faces using SRM features. 


## Pre-requisites
1. NVIDIA GPU + CUDA cuDNN

##Dataset
For the dataset we used 4 dataset:
1. FFHQ DAtAset
2. Synthetic imagges generated from [PROGAN](https://github.com/tkarras/progressive_growing_of_gans) Network.
3. Synthetic imagges generated from [StyleGAN 1](https://github.com/NVlabs/stylegan) Network.
4. Synthetic imagges generated from [StyleGAN 2](https://github.com/NVlabs/stylegan2) Network.

Use this model to generate the dataset or you can use the availalbe from their github repos.

Direectory Structure  for dataset
``` dataset-|---- /ffhq_/ -|- ffhq__0000.png
		  |			     |- ffhq__0001.png
		  |			     .
		  |			     .
		  |
		  |-----/pgan_/ -|- pgan__0001.png
		  | 			 |- pgan__0002.png
		  |			     |- pgan__0003.png
		  |			     .
		  |			     .
		  |
		  |-----/sgan1/ -|- sgan1_0001.png
		  |			     |- sgan1_0002.png
		  |			     .
		  |			     .
		  |
		  |-----/sgan2_/-|- sgan2_0001.png
	  	  |			     |- sgan2_0002.png
		  |			     .
		  |			     .
		  |
		  |-----/ffhq__filter/ -|- ffhq__0001.png
		  |			     		|- ffhq__0002.png
		  |			     		.
		  |			     		.
		  |
		  |-----/pgan__filter/ -|- pgan__0001.png
		  |			     		|- pgan__0002.png
		  |			     		.
		  |			     		.
		  |
		  |-----/sgan1_filter/ -|- sgan1_0001.png
		  |			     		|- sgan1_0002.png
		  |			   			.
		  |			     		.
		  |
		  |-----/sgan2_filter/ -|- sgan1_0001.png
		  			     		|- sgan2_0002.png
	   			     			.
	   			     			.
	   			     			
```

![Sample imaaage from the dataset](/dataset/Slide1.PNG)

Rename and sace the files according to the above directory structure.

Run `python filter.py` to generate the SDRM filter images.

## Setup
1. Clone this repo: `git clone https://github.com/Shubhanshu07/Classification-of-GAN-Generated-Fake-Faces.git`
2. CD into this repo: `cd Classification-of-GAN-Generated-Fake-Faces.git`
3. Install [PyTorch 1.0]
4. Install python requirements or build docker image 
    - Install python requirements: `pip install -r requirements.txt`

## Training
1. `python train.py --output_directory=outdir --log_directory=logdir` or `python train.py -o outdir -l logdir`
 2. (OPTIONAL) `tensorboard --logdir=outdir/logdir`

## Training using a pre-trained model
Training using a pre-trained model can lead to faster convergence  
By default, the dataset dependent text embedding layers are [ignored]

1. Download our published [Tacotron 2] model
2. `python train.py --output_directory=outdir --log_directory=logdir -c tacotron2_statedict.pt `


## Inference demo
1. `jupyter notebook --ip=127.0.0.1 --port=31337`
2. Load inference.ipynb 


## Acknowledgements
This project is done as a part of B. Tech. Project under the guidance of Professor Vinod Panjakshan, Department of Electronics and Communication Engineering, Indian Institute of Technology, Roorkee. 
We want to extend our heartiest thanks to our supervisor Professor Vinod Pankajakshan for providing us an opportunity to work on this challenging project. His support and continued motivation helped us to gain insight into various aspects of research and development. We would never have been able to complete our work without our supervisors’ guidance, help from friends, and support from our families and loved ones. Last but not least, we thank God/Nature from the inner part of the soul who made all the things possible.


