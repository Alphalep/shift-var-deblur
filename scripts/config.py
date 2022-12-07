import os

# path to saving models
models_dir = 'ShiVaNet/models'

# path to saving loss plots
losses_dir = 'ShiVaNet/losses'

# path to the data directories
data_dir = 'ShiVaNet/data'
train_dir = 'ShiVaNet/train'
val_dir = 'ShiVaNet/val'
imgs_dir = 'ShiVaNet/imgs'
noisy_dir = 'ShiVaNet/noisy'
debug_dir = 'ShiVaNet/debug'

# depth of UNet 
depth = 4 # try decreasing the depth value if there is a memory error

# text file to get text from
txt_file_dir = 'shitty_text.txt'
# maximun number of synthetic words to generate
num_synthetic_imgs = 18000
train_percentage = 0.8

resume = False  # False for trainig from scratch, True for loading a previously saved weight
ckpt='model60.pth' # model file path to load the weights from, only useful when resume is True
lr = 1e-5          # learning rate
epochs = 100     # epochs to train for 

# batch size for train and val loaders
batch_size = 16 # try decreasing the batch_size if there is a memory error

# log interval for training and validation
log_interval = 25

test_dir = os.path.join(data_dir, val_dir, noisy_dir)
res_dir = 'results'
test_bs = 16
