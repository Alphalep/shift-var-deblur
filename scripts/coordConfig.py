import os
from scripts.forward import *
import numpy as np 
# path to saving models
models_dir = 'checkpoint/cgan_Setup3'

# path to saving loss plots
losses_dir = 'losses/cgan_Setup3'

# path to the data directories
blur_train = "data/Urban100/blur_train"
blur_val  = "data/Urban100/blur_val"
root_train="data/Urban100/train"
root_val = "data/Urban100/val"

#data_dir = 'data/CoordConv'
##val_dir = 'val/CoordConv'
#imgs_dir = 'imgs/CoordConv'
#noisy_dir = 'noisy/CoordConv'
#debug_dir = 'debug/CoordConv'


# maximun number of synthetic words to generate
#num_synthetic_imgs = 18000
#rain_percentage = 0.8

"""CoordConv Setup"""
switch_disc =  True  #Switches coordconv layers in the discriminator
switch_CGAN_encoder  = True #Switches coordconv layers in the encoder leg of UNET
switch_cGAN_decoder =  True #Switches coordconv layers in the decoder leg of UNET


resume = True # False for trainig from scratch, True for loading a previously saved weight
ckpt='model50.pth' # model file path to load the weights from, only useful when resume is True
lrG =  1e-5          # learning rate
lrD  = 1e-5
beta1 = 0.5     # beta1 for Adam Optimizer
beta2 = 0.999   # beta2 for Adam Optimizer
include_reblurLoss = True
epochs = 200    # epochs to train for 

# batch size for train and val loaders
batch_size = 32 # try decreasing the batch_size if there is a memory error

# log interval for training and validation
log_interval = 50

#test_dir = os.path.join(data_dir, val_dir, noisy_dir)
res_dir = 'results/cgan_Setup1'
test_bs = 4

#Blur Model Parameters

#root_dir= "data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR"
optics_setup = np.load("psf_info.npz")
psf_arr = optics_setup['psf_arr']
windows  = optics_setup['windows']
setup =optics_setup['setup']

k_size = setup[0]
step_size = k_size//2

#Training Parameters
L1_lambda = 50
L2_lambda = 50


#test_dir = os.path.join(data_dir, val_dir, noisy_dir)
res_dir = 'results/cgan_pl_no_cycle_111'
test_bs = 4

#Blur Model Parameters

root_dir= "data/DIV2K/DIV2K_valid_HR/DIV2K_valid_HR"
optics_setup = np.load("psf_info.npz")
psf_arr = optics_setup['psf_arr']
windows  = optics_setup['windows']
setup =optics_setup['setup']

k_size = setup[0]
step_size = k_size//2

#Training Parameters
L1_lambda = 50
L2_lambda = 50

