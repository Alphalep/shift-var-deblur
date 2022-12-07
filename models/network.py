from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.functional as F
from coordconv import CoordConv2d,CoordConvTranspose2d


class cGAN(nn.Module):
    def __init__(
        self,
        in_channels = 1,
        out_channels = 1,
        depth = 64,
        padding = 1,
        encodeCoordConv = False,
        decodeCoordConv = False
    ):
        super(cGAN,self).__init__()
        self.padding = padding
        self.depth = depth
        prev_channels = in_channels
        
        if encodeCoordConv:
            # U-NET ENCODER
            
            self.down_path = nn.ModuleList()
            self.down_path.append(CoordConv2d(in_channels,depth,4,2,1))
            self.down_path.append(UNetCoordConvBlock(depth,depth*2,4,2,1))
            self.down_path.append(UNetCoordConvBlock(depth*2,depth*4,4,2,1))
            self.down_path.append(UNetCoordConvBlock(depth*4,depth*8,4,2,1))
            self.down_path.append(UNetCoordConvBlock(depth*8,depth*8,4,2,1))
            self.down_path.append(UNetCoordConvBlock(depth*8,depth*8,4,2,1))
            self.down_path.append(UNetCoordConvBlock(depth*8,depth*8,4,2,1))
            self.down_path.append(CoordConv2d(depth*8,depth*8,4,2,1))
    
        else:
            self.down_path = nn.ModuleList()
            self.down_path.append(nn.Conv2d(in_channels,depth,4,2,1))
            self.down_path.append(UNetConvBlock(depth,depth*2,4,2,1))
            self.down_path.append(UNetConvBlock(depth*2,depth*4,4,2,1))
            self.down_path.append(UNetConvBlock(depth*4,depth*8,4,2,1))
            self.down_path.append(UNetConvBlock(depth*8,depth*8,4,2,1))
            self.down_path.append(UNetConvBlock(depth*8,depth*8,4,2,1))
            self.down_path.append(UNetConvBlock(depth*8,depth*8,4,2,1))
            self.down_path.append(nn.Conv2d(depth*8,depth*8,4,2,1))


            # U-NET DECODER
        if decodeCoordConv:

            self.up_path = nn.ModuleList()
            
            self.up_path.append(UNetCoordUpBlock(depth*8,depth*8,4,2,1))
            self.up_path.append(UNetCoordUpBlock(depth*8*2,depth*8,4,2,1))
            self.up_path.append(UNetCoordUpBlock(depth*8*2,depth*8,4,2,1))
            self.up_path.append(UNetCoordUpBlock(depth*8*2,depth*8,4,2,1))
            self.up_path.append(UNetCoordUpBlock(depth*8*2,depth*4,4,2,1))
            self.up_path.append(UNetCoordUpBlock(depth*4*2,depth*2,4,2,1))
            self.up_path.append(UNetCoordUpBlock(depth*2*2,depth,4,2,1))


            self.last = CoordConvTranspose2d(depth*2,out_channels,4,2,1)

        else :
            
            self.up_path = nn.ModuleList()

            self.up_path.append(UNetUpBlock(depth*8,depth*8,4,2,1))
            self.up_path.append(UNetUpBlock(depth*8*2,depth*8,4,2,1))
            self.up_path.append(UNetUpBlock(depth*8*2,depth*8,4,2,1))
            self.up_path.append(UNetUpBlock(depth*8*2,depth*8,4,2,1))
            self.up_path.append(UNetUpBlock(depth*8*2,depth*4,4,2,1))
            self.up_path.append(UNetUpBlock(depth*4*2,depth*2,4,2,1))
            self.up_path.append(UNetUpBlock(depth*2*2,depth,4,2,1))

            self.last  = nn.ConvTranspose2d(depth*2,out_channels,4,2,1)
    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward (self,x):
        bridge = []
        for i,down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path)-1:
                bridge.append(x)
        for i,up in enumerate(self.up_path):
            if i < 3:
                x = F.dropout(up(x,bridge[-i-1]),0.5,training=True)
            else:
                x = up(x,bridge[-i-1])
        
        out = self.last(x)
        out = F.tanh(out) # Using a tanh function
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size,kernel_size,stride, padding, batch_norm=True):
        super(UNetUpBlock, self).__init__()
        block =[]
        block.append(nn.ReLU())
        block.append(nn.ConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride,padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)
           
    def forward(self,x,bridge):
        out = self.block(x)
        out =  torch.cat([out,bridge],1)
        return out


class UNetCoordUpBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size,stride,padding, batch_norm=True):
        super(UNetCoordUpBlock, self).__init__()
        block =[]
        block.append(nn.ReLU())
        block.append(CoordConvTranspose2d(in_size, out_size, kernel_size=kernel_size, stride=stride,padding=int(padding)))
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        self.block = nn.Sequential(*block)
     
    def forward(self, x,bridge):
        out = self.block(x)
        out =  torch.cat([out,bridge],1)
        return out 


class UNetCoordConvBlock(nn.Module):
    def __init__(self, in_size, out_size, kernel_size,stride,padding, batch_norm=True):
        super(UNetCoordConvBlock, self).__init__()
        block = []
        block.append(nn.LeakyReLU(negative_slope=0.2))
        block.append(CoordConv2d(in_size, out_size, kernel_size=kernel_size,stride = stride, padding=int(padding),with_r=False))#Changed from Conv2d to CoordConv2d
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size,kernel_size,stride,padding, batch_norm=True):
        super(UNetConvBlock, self).__init__()
        block = []
        block.append(nn.LeakyReLU(negative_slope=0.2))
        block.append(nn.Conv2d(in_size, out_size, kernel_size=kernel_size,stride =stride,padding=int(padding)))
        
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Discriminator(nn.Module): 
    # initializers
    def __init__(self,in_channels=1,d=64, coordConv=False):
        super(Discriminator, self).__init__()
        if coordConv:
            self.conv1 = CoordConv2d(in_channels*2, d, 4, 2, 1)
            self.conv2 = CoordConv2d(d, d * 2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(d * 2)
            self.conv3 = CoordConv2d(d * 2, d * 4, 4, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(d * 4)
            self.conv4 = CoordConv2d(d * 4, d * 8, 4, 1, 1)
            self.conv4_bn = nn.BatchNorm2d(d * 8)
            self.conv5 = CoordConv2d(d * 8, 1, 4, 1, 1)
        else:
            self.conv1 = nn.Conv2d(in_channels*2, d, 4, 2, 1)
            self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
            self.conv2_bn = nn.BatchNorm2d(d * 2)
            self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
            self.conv3_bn = nn.BatchNorm2d(d * 4)
            self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 1, 1)
            self.conv4_bn = nn.BatchNorm2d(d * 8)
            self.conv5 = nn.Conv2d(d * 8, 1, 4, 1, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = F.sigmoid(self.conv5(x))

        return x
 
