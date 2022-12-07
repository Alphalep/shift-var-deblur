import sys, os, time, glob, time, pdb, cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from scripts.forward import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts.trackers import *
plt.switch_backend('agg') # for servers not supporting display

# import neccesary libraries for defining the optimizers
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms

from models.coordunet import CoordConvUNet
from models.network import cGAN, Discriminator
import scripts.coordConfig as cfg
def main():
    script_time = time.time()
    """CoordConv Setup"""
    switch_disc =  True
    switch_CGAN_encoder  = True
    switch_cGAN_decoder =  True

    """Declare the Device Cluster for training"""
    #Needs modification for DDP Parallel
    #How to change device id?

    print(os.path.abspath("."))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print('device: ', device)

    """Load and Transform dataset for training"""

    transform = transforms.Compose([transforms.Resize((512,512)),
                                        transforms.ToTensor()])
    blur_train = "data/DIV2K/blur_train"
    blur_val  = "data/DIV2K/blur_val"
    batch_size = cfg.batch_size
    models_dir = cfg.models_dir
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)
    


    train_dataset = Custom_Dataset(root_dir="data/DIV2K/train",blur_dir = blur_train,transform=transform)
    val_dataset = Custom_Dataset(root_dir="data/DIV2K/val",blur_dir = blur_val,transform=transform)
    #Loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = batch_size, shuffle = not True)
    
    #Information about loading
    print('\nlen(train_loader): {}  @bs={}'.format(len(train_loader), batch_size))
    print('len(val_loader)  : {}  @bs={}'.format(len(val_loader), batch_size))

    """Model Definition"""

    gen = cGAN(in_channels=1,out_channels=1,depth=64,padding = 1, encodeCoordConv=switch_CGAN_encoder,decodeCoordConv=switch_cGAN_decoder).to(device)
    disc = Discriminator(in_channels = 1 , d = 64, coordConv  = switch_disc).to(device)
    gen.weight_init(mean=0.0, std=0.02)
    disc.weight_init(mean=0.0, std=0.02)
    
    """ RESUME AND RELOAD"""
    resume = cfg.resume
    if not resume:
        print('\nfrom scratch')
        train_epoch_loss = []
        val_epoch_loss = []
        running_disc_loss = []
        running_train_loss =[]
        running_val_loss = []
        epochs_till_now = 0
    else:
        ckpt_path = os.path.join(models_dir, cfg.ckpt)
        ckpt = torch.load(ckpt_path)
        print(f'\nckpt loaded: {ckpt_path}')
        gen_state_dict = ckpt['gen_state_dict']
        gen.load_state_dict(gen_state_dict)
        gen.to(device)
        disc_state_dict = ckpt['disc_state_dict']
        disc.load_state_dict(disc_state_dict)
        disc.to(device)
        

        losses = ckpt['losses']
        running_disc_loss=losses['running_disc_loss']
        running_train_loss = losses['running_train_loss']
        running_val_loss = losses['running_val_loss']
        train_epoch_loss = losses['train_epoch_loss']
        val_epoch_loss = losses['val_epoch_loss']
        epochs_till_now = ckpt['epochs_till_now']


    """Initialize Forward Model"""
    svblur = svBlur(psfs = cfg.psf_arr, windows = cfg.windows , step_size = cfg.step_size,device = 'cuda')
    
    """LOSS CRITERION"""
    BCE_loss = nn.BCELoss()
    L1_loss = nn.L1Loss()
    """Adam Optimizer"""
    G_optimizer = optim.Adam(gen.parameters(), lr=cfg.lrG, betas=(cfg.beta1, cfg.beta2))
    D_optimizer = optim.Adam(disc.parameters(), lr=cfg.lrD, betas=(cfg.beta1, cfg.beta2))
    
    log_interval = cfg.log_interval
    epochs = cfg.epochs

    

    ###
    print('\nmodel has {} M parameters'.format(count_parameters(disc)+count_parameters(gen)))
    print(f'epochs_till_now: {epochs_till_now}')
    print(f'epochs from now: {epochs}')
    ###

    for epoch in range(epochs_till_now,epochs_till_now+epochs):
        print('\n===== EPOCH {}/{} ====='.format(epoch + 1, epochs_till_now + epochs))    
        print('\nTRAINING...')
        epoch_train_start_time = time.time()
        #Put the models into training 

        gen.train()
        disc.train()

        for batch_idx, (imgs,noisy_imgs) in enumerate(train_loader):
            
            batch_start_time = time.time()
            imgs = imgs.to(device)
            noisy_imgs = noisy_imgs.to(device)

            """Train Discriminator""" 
            D_optimizer.zero_grad()
            #train with real
            d_result =  disc(noisy_imgs,imgs) ####PROBLEM
            d_real_loss = BCE_loss(d_result,torch.ones_like(d_result).to(device))
            #train with fake
            img_gen  = gen(imgs)
            d_result = disc(noisy_imgs,img_gen).squeeze()
            d_fake_loss = BCE_loss(d_result,torch.zeros_like(d_result).to(device))
            d_train_loss =  (d_fake_loss + d_real_loss)*0.5
            running_disc_loss.append(d_train_loss.item())
            d_train_loss.backward()
            D_optimizer.step()
            
            """Train Generator"""

            G_optimizer.zero_grad()
            out = gen(noisy_imgs)
            d_result = disc(noisy_imgs, out).squeeze()
            g_train_loss = BCE_loss(d_result, torch.ones_like(d_result).to(device)) + cfg.L1_lambda * L1_loss(out,imgs)  #+ cfg.L2_lambda * L1_loss(svblur(out),noisy_imgs)
            running_train_loss.append(g_train_loss.item())
            g_train_loss.backward()
            G_optimizer.step()

            if (batch_idx + 1)%log_interval == 0:
                batch_time = time.time() - batch_start_time
                m,s = divmod(batch_time, 60)
                print('train loss @batch_idx {}/{}: {} in {} mins {} secs (per batch)'.format(str(batch_idx+1).zfill(len(str(len(train_loader)))), len(train_loader), g_train_loss.item(), int(m), round(s, 2)))


        train_epoch_loss.append(np.array(running_train_loss).mean())
        epoch_train_time = time.time() - epoch_train_start_time
        m,s = divmod(epoch_train_time, 60)
        h,m = divmod(m, 60)
        print('\nepoch train time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))

        print('\nVALIDATION...')
        epoch_val_start_time = time.time()
        disc.eval()
        gen.eval()
        with torch.no_grad():
            for batch_idx, (imgs, noisy_imgs) in enumerate(val_loader):
                imgs = imgs.to(device)
                noisy_imgs = noisy_imgs.to(device) 
                out = gen(noisy_imgs)
                d_result = disc(noisy_imgs, out).squeeze()
                loss = BCE_loss(d_result, torch.ones_like(d_result).to(device)) + cfg.L1_lambda * L1_loss(out,imgs)  #+ cfg.L2_lambda * L1_loss(svblur(out),noisy_imgs)
                running_val_loss.append(loss.item())

                if (batch_idx + 1)%log_interval == 0:
                    print('val loss   @batch_idx {}/{}: {}'.format(str(batch_idx+1).zfill(len(str(len(val_loader)))), len(val_loader), loss.item()))
        val_epoch_loss.append(np.array(running_val_loss).mean())

        epoch_val_time = time.time() - epoch_val_start_time
        m,s = divmod(epoch_val_time, 60)
        h,m = divmod(m, 60)
        print('\nepoch val   time: {} hrs {} mins {} secs'.format(int(h), int(m), int(s)))        
        if (epoch+1)%log_interval == 0:
           
            #Plot Loss-Epoch and Loss-Batch curve
            plot_losses(running_train_loss, running_disc_loss, running_val_loss, val_epoch_loss,  epoch)   
             #Store Model Dict
            torch.save({'gen_state_dict': gen.state_dict(),
                        'disc_state_dict': disc.state_dict(),
                        'losses': {'running_train_loss': running_train_loss,
                                    'running_disc_loss': running_disc_loss, 
                                    'running_val_loss': running_val_loss, 
                                    'train_epoch_loss': train_epoch_loss, 
                                    'val_epoch_loss': val_epoch_loss}, 
                        'epochs_till_now': epoch+1}, 
                    os.path.join(models_dir, 'model{}.pth'.format(str(epoch + 1).zfill(2))))
                
    total_script_time = time.time() - script_time
    m, s = divmod(total_script_time, 60)
    h, m = divmod(m, 60)

    print(f'\ntotal time taken for running this script: {int(h)} hrs {int(m)} mins {int(s)} secs')
  
    print('\nHasta La Vista.')


if __name__ == "__main__":
    main()