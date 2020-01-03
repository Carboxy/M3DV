# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:14:49 2019

@author: Carboxy
"""

import torch.nn as nn
#import torch

class CarboxyNet(nn.Module):
    def __init__(self):
        super(CarboxyNet,self).__init__()
        self.features = nn.Sequential(
                nn.Conv3d(1,32,kernel_size=3,padding=1),
                nn.BatchNorm3d(32),
                
                nn.ReLU(inplace=True),
                nn.Conv3d(32,64,kernel_size=5,padding=2),
                nn.BatchNorm3d(64),
                
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2,stride=2),
                nn.Conv3d(64,192,kernel_size=3,padding=1),
                nn.BatchNorm3d(192),
                
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2,stride=2),
                nn.Conv3d(192,128,kernel_size=3,padding=1),
                nn.BatchNorm3d(128),
                
                nn.ReLU(inplace=True),
                nn.MaxPool3d(kernel_size=2,stride=2)
                )
        
        self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(128*4*4*4,2048),
                
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(2048,1024),
                
                nn.ReLU(inplace=True),
                nn.Linear(1024,2),
                nn.Sigmoid()  
                )
    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x