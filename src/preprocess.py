# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 08:45:36 2019

@author: Carboxy
"""

#import torch.nn as nn
#import torch
#from sklearn.model_selection import KFold
import os
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#读取test
file =open('test.csv','r')
lines=file.readlines()
file.close()
row=[]
names=[]
labels = []
for line in lines:
    row.append(line.split(','))
for col in row:
    names.append(col[0])
    labels.append(np.double(col[1]))
    
labels = np.array(labels)
x,y,z=32,32,32
Xtest = np.zeros((117,x,y,z))
index = 0
x_begin = int((100-x)/2)
x_end = x_begin + x
y_begin = int((100-y)/2)
y_end = y_begin + y
z_begin = int((100-z)/2)
z_end = z_begin + z
for name in names:
    data_name = 'test/'+name+'.npz'
    npz_readin = np.load(data_name)      
    temp =npz_readin['voxel']
    Xtest[index,:,:,:] = temp[x_begin:x_end,y_begin:y_end,z_begin:z_end]
    index = index+1
    
    
#读取train_dev
file =open('train_val.csv','r')
lines=file.readlines()
file.close()
row=[]
names=[]
labels = []
for line in lines:
    row.append(line.split(','))
for col in row:
    names.append(col[0])
    labels.append(np.double(col[1]))
    
    
labels = np.array(labels)
Xtrain = np.zeros((465,x,y,z))
index = 0
x_begin = int((100-x)/2)
x_end = x_begin + x
y_begin = int((100-y)/2)
y_end = y_begin + y
z_begin = int((100-z)/2)
z_end = z_begin + z
for name in names:
    data_name = 'train_val/'+name+'.npz'
    npz_readin = np.load(data_name)      
    temp =npz_readin['voxel']
    Xtrain[index,:,:,:] = temp[x_begin:x_end,y_begin:y_end,z_begin:z_end]
    index = index+1
    
Xtrain = np.reshape(Xtrain,[465,32*32*32])
Xtest = np.reshape(Xtest,[117,32*32*32])

ss = StandardScaler()
Xtrain = ss.fit_transform(Xtrain)
Xtest = ss.transform(Xtest)

Xtrain  = np.reshape(Xtrain,[465,1,32,32,32])
Xtest = np.reshape(Xtest,[117,1,32,32,32])


np.save('data_zscore/Xtrain_dev.npy',Xtrain)
np.save('data_zscore/Xtest.npy',Xtest)
   