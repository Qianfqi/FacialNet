# FacialNet.py

# This file defines the FacialNet model architecture.
# This file should include:

# 1. Import necessary libraries
#    - PyTorch and its neural network modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
class D_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(D_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.branch = nn.Sequential(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(out_channels, out_channels, kernel_size=1))
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # print(x.shape)
        x1 = self.bn1(self.conv1(x))
        x2 = self.bn2(self.branch(x))
        # print(x1.shape, x2.shape)
        x = self.relu(x1 + x2)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = D_block(in_channels, in_channels)
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
    def forward(self, x, skip_connection=None):
        # print(x.shape)
        if skip_connection is not None:  
            # print(skip_connection.shape)
            x = torch.cat((x, skip_connection), dim=1)
        # print(x.shape)
        x = self.conv(x)
        # print(x.shape)
        x = self.upsample(x)
        # print(x.shape)
        return x
        

class FacialNet(nn.Module):
    def __init__(self):
        super(FacialNet, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.encoder = self.mobilenet.features
        self.stage1 = self.encoder[:2]   # first stage
        self.stage2 = self.encoder[2:4]  # second stage
        self.stage3 = self.encoder[4:7]  # third stage
        self.stage4 = self.encoder[7:14] # fourth stage
        self.stage5 = self.encoder[14:]  # fifth stage
        self.decoder1 = Decoder(1280, 96)
        self.decoder2 = Decoder(192, 32)
        self.decoder3 = Decoder(64, 24)
        self.decoder4 = Decoder(48, 16)
        self.decoder5 = Decoder(32, 3)
        self.final_conv1 = nn.Conv2d(3, 9, kernel_size=1)
    def forward(self, x):
        enc1 = self.stage1(x)
        enc2 = self.stage2(enc1)
        enc3 = self.stage3(enc2)
        enc4 = self.stage4(enc3)
        enc5 = self.stage5(enc4)
        # print(enc1.shape, enc2.shape, enc3.shape, enc4.shape, enc5.shape)
        dec1 = self.decoder1(enc5, None)
        dec2 = self.decoder2(dec1, enc4)
        dec3 = self.decoder3(dec2, enc3)
        dec4 = self.decoder4(dec3, enc2)
        dec5 = self.decoder5(dec4, enc1)
        out = self.final_conv1(dec5)
        return out