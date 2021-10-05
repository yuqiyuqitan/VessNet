import torch
import torch.nn as nn 
import numpy as np

class VessNet(nn.Module):
    
    # initiate model
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1,
                 ker_size_encode=[(1,3,3),(1,3,3),(3,3,3)],
                 pad_encode=[(0,1,1),(0,1,1),(1,1,1)],
                 ker_size_decode=[(3,3,3),(1,3,3),(1,3,3)],
                 pad_decode=[(1,1,1),(0,1,1),(0,1,1)],
                 down_factor=[(1,2,2),(1,2,2),(2,2,2)],
                 up_factor=[(2,2,2),(1,2,2),(1,2,2)],
                 fnumb=8,
                 final_activation=None):
        super().__init__()
        # encode path
        current_channels = in_channels
        self.enconv = []
        self.down = []
        for i in range(len(ker_size_encode)):
            self.enconv.append(self._conv(current_channels, fnumb, ker_size_encode[i], pad_encode[i]))     
            current_channels = fnumb
            fnumb *= 2
            self.down.append(self._down(down_factor[i]))
            
        self.enconv = nn.ModuleList(self.enconv)
        self.down=nn.ModuleList(self.down)
        # base 
        self.base = self._conv(current_channels,fnumb,ker_size=(3,3,3), pad=(1,1,1))

        # decode path
        self.deconv = []
        self.up = []
        for j in range(len(ker_size_decode)):
            self.up.append(self._up(fnumb, fnumb, up_factor[j]))
            self.deconv.append(self._conv(int(fnumb/2)+fnumb, int(fnumb/2), ker_size_decode[j], pad_decode[j]))
            fnumb = int(fnumb/2)

       
        self.up = nn.ModuleList(self.up)
        self.deconv = nn.ModuleList(self.deconv)   

        # output convolution

        self.last = nn.Sequential(
                            nn.Conv3d(fnumb, out_channels, kernel_size=(1,1,1)),
                            nn.Sigmoid())
                
                                
    # helper function for convolution layers
    def _conv(self, in_channels, out_channels, ker_size, pad):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size= ker_size, padding= pad),
                             nn.ReLU(),
                             nn.Conv3d(out_channels, out_channels, kernel_size= ker_size, padding= pad),
                             nn.ReLU()) 
    
    # helper function for 3D downsampling with MaxPooling
    def _down(self, ker_size):
        return nn.MaxPool3d(kernel_size=ker_size, stride=ker_size)
    
    # helper function for 3D upsampling with transposed 3D convolution
    def _up(self, in_channels, out_channels, ker_size):
        return nn.ConvTranspose3d(in_channels, out_channels, kernel_size=ker_size, stride=ker_size)
    
    def forward(self, input):
        x = input
        
        # apply encoder path
        enconv_out = []
        for level in range(3):
            x = self.enconv[level](x)
            enconv_out.append(x)
            x = self.down[level](x)
            
        # apply base
        x = self.base(x)
        
        # apply decoder path
        enconv_out = enconv_out[::-1]
        for level in range(3):
            x = self.up[level](x)
            x = torch.cat((x, enconv_out[level]), dim=1)
            x = self.deconv[level](x)
            
        # apply output conv and activation (if given)
        x = self.last(x)
        return x