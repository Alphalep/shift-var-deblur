import torch
import math
import cv2
from tqdm import tqdm
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal,misc
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft2,fftshift,fft,ifft2,ifftshift
from empatches import EMPatches
import numpy as np
from skimage import color, data, restoration
from helper import *

N = 129# Kernel Size
img_size = (512,512)
radius = N//2 #np.sqrt(N//np.pi) #Radius of the aperture
center = [N//2,N//2]
lam = 0.55 * 10**-6
wave_number = 2*np.pi/lam
W_d = 1*lam  
W_sa = 0
W_coma= 10 * lam
W_astig = 5*lam 
W_field = 0
W_distort  = 5*lam 
coeffs = [W_d,W_sa,W_coma,W_astig,W_field,W_distort]
#Shift of point in test image
side =2
x_shift = 10
y_shift  = 0
#For patching
patch_size = 129
step_size = N//2
overlapPercent = 1- (step_size/patch_size)
amp = amplitude_aperture(N,radius,center)
phase_mask = phase_aperture(N,center,radius,wave_number,1,0,coeffs)
pupil = amp*(phase_mask)
A,B,C,D = expand_complex_image(pupil)
setup = np.array([N,patch_size])

#------------------------------#
# ---------PATCHING -----------#
#------------------------------#
emp = EMPatches()
img = np.zeros(img_size)
img_p = np.pad(img,step_size)
img_patches, indices = emp.extract_patches(img_p, patchsize=patch_size, overlap=overlapPercent)
img_R = np.sqrt((img.shape[0]//2)**2 +(img.shape[1]//2)**2)
#np.save("indices.npy",indices)

psf_arr = []
for ind in tqdm(indices):
    X = (ind[0] + ind[1])/2
    Y = (ind[2]+ind[3])/2
    p0 = (X-img.shape[0]/2)/img_R
    q0 = (Y-img.shape[1]/2)/img_R
    G = amp*(phase_aperture(N,center,radius,wave_number,p0,q0,coeffs))
    p_norm = get_mag(fft_image(G))**2
    p_norm = p_norm/np.sum(p_norm)
    psf_arr.append(p_norm)

psf_stack = np.stack(psf_arr,axis=2)
psf_stack = np.expand_dims(psf_stack,axis=3)
print(psf_stack.shape)
#saved_psf = np.save('MultiWienerNet/psfs.npy',psf_stack)
#print(psf_arr.shape)

window_arr = []


for n in range(len(psf_arr)):  
    windows = np.zeros(img_p.shape,float)
    ind = indices[n]
    windows[ind[0]:ind[1],ind[2]:ind[3]] = bartlett(N)
    window_arr.append(windows) 
windows = np.array(window_arr)
print(windows.shape)
np.savez("psf_info",psf_arr = psf_arr,windows = windows , setup = setup )
