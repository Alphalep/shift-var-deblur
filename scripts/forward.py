from scripts.helper import *
from torch.utils.data import Dataset
from torchvision import datasets,models,transforms
from torchvision.utils import make_grid
import os
from os import listdir
from PIL import Image
#Forward Model with Seidel Aberrations
#Load the PSF arrays 

class Urban100(Dataset):
    def __init__(self,root_dir,transform=None):
        self.root_dir  = root_dir
        self.transform = transform
        self.tensor_transform = transforms.Compose([transforms.CenterCrop((512,512)),transforms.ToTensor()])

    def __len__(self):
        return(len(listdir(self.root_dir))) 


    def __getitem__(self,idx):
        file_name = os.listdir(self.root_dir)[idx]
        file_path = os.path.join(self.root_dir,file_name)
        image = Image.open(file_path).convert('L')
        img_hr = self.tensor_transform(image)
        img_tensor  = torch.unsqueeze(img_hr,0)
        if self.transform is not None:
            img_blur = self.transform(img_tensor)
            
        return (img_hr,img_blur,file_name)
   
   
class Custom_Dataset(Dataset):
    def __init__(self,root_dir,blur_dir,transform=None):
        self.root_dir  = root_dir
        self.blur_dir  = blur_dir
        self.transform = transform

    def __len__(self):
        return(len(listdir(self.root_dir))) 


    def __getitem__(self,idx):
        file_name = os.listdir(self.root_dir)[idx]

        file_path_true = os.path.join(self.root_dir,file_name)
        file_path_blur = os.path.join(self.blur_dir,file_name)
        image_true = Image.open(file_path_true).convert('L')
        image_blur = Image.open(file_path_blur).convert('L')
        
        if self.transform is not None:
            img_hr = self.transform(image_true)
            img_blur = self.transform(image_blur)
            
        return (img_hr,img_blur)


class shiftVariantBlur(object):

    def __init__(self,psf_arr,step_size,indices):
        self.psf_arr = psf_arr
        self.indices = indices
        self.step_size = step_size
        self.len = len(indices)

    def __call__(self,x):
        img = np.array(x)
        img_p = np.pad(img,self.step_size)
        window = np.zeros(img_p.shape,float)
        sum = window
        for id in range(self.len):
            ind = self.indices[id]
            window[ind[0]:ind[1],ind[2]:ind[3]]= tri(2*self.step_size)
            temp = cv2.filter2D((window*img_p),-1,np.flip(self.psf_arr[id],-1),cv2.BORDER_CONSTANT)
            sum = sum + temp
            window[ind[0]:ind[1],ind[2]:ind[3]]= 0
        sum= sum[self.step_size:sum.shape[0]-self.step_size,self.step_size:sum.shape[1]-self.step_size]
        return Image.fromarray(sum)


class svBlur(object):
    def __init__(self,psfs,windows,step_size,device):
        self.psfs = torch.from_numpy(psfs).unsqueeze(1) 
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device

    def __call__(self,imgs):
        imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        #imgs = imgs.expand(imgs.shape[0],self.psfs.size(0),imgs.shape[2],imgs.shape[3])
        #patched_imgs = imgs * self.windows.expand(imgs.shape).to(self.device)
        output = torch.sum(F.conv2d(imgs.expand(imgs.shape[0],self.psfs.size(0),imgs.shape[2],imgs.shape[3]) * self.windows.expand(imgs.shape[0],self.psfs.size(0),-1,-1).to(self.device),self.psfs.to(self.device),groups = self.psfs.size(0)),dim=1,keepdim=True)
        return output


class svBlur_tx(object):
    def __init__(self,psfs,windows,step_size,device):
        self.psfs = torch.from_numpy(psfs).unsqueeze(1) 
        self.step_size =  step_size
        self.windows = torch.from_numpy(windows)
        self.device = device

    def __call__(self,imgs):
        imgs = F.pad(imgs,(self.step_size,self.step_size,self.step_size,self.step_size),"constant",0)
        #patched_imgs = imgs.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device)
        output = torch.sum(F.conv2d(imgs.expand(-1,self.psfs.size(0),-1,-1).to(self.device) * self.windows.expand(imgs.size(0),self.psfs.size(0),-1,-1).to(self.device),self.psfs.to(self.device),groups = self.psfs.size(0)),dim=1,keepdim=True)
        return output
