# -*- coding: utf-8 -*-

from densenet import DenseNet
import torch

# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 00:08:13 2019

@author: Carboxy
"""

'''
考虑到内存开销，该版本将数据增强过程放在每个批次中
'''

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataHelper1 import DealTrainset,DealDevset,DealTestset, preprocess
from model import CarboxyNet
from sklearn import metrics
import time
import os

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 超参数设置
BATCH_SIZE = 16
NUM_EPOCHS = 40
LR = 0.0001

#数据预处理
#preprocess()
#定义训练批处理数据
dealTrainset = DealTrainset()
train_loader = DataLoader(dataset=dealTrainset,
                          batch_size=BATCH_SIZE,
                          shuffle=True
                          )

# 定义验证批处理数据
dealDevset = DealDevset()
dev_loader = DataLoader(dataset=dealDevset,
                          batch_size=BATCH_SIZE,
                          shuffle=False
                          )
#定义损失函数和优化方式
net = CarboxyNet().to(device)
criterion = nn.CrossEntropyLoss()
#criterion = nn.BCELoss()
#criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)

#保存模型的一些量
scores_perepoch=[]
dev_auc = []
compare_dev_auc = 0
time = str(time.time())
save_path = 'parameter2/' + time 
isExist = os.path.exists(save_path)
if not isExist:
    os.makedirs(save_path)
    print('创建保存目录成功')
else:
    print('目录已存在')

for epoch in range(NUM_EPOCHS):
    sum_loss = 0.0
    #数据读取
    for i,data in enumerate(train_loader):
        Xtrain,Ytrain = data
        Xtrain = Xtrain.type(torch.FloatTensor)
        Ytrain = Ytrain.type(torch.LongTensor)
        Xtrain,Ytrain = Xtrain.to(device), Ytrain.to(device)
        
        #梯度清零
        optimizer.zero_grad()
        
        #前后传播+后向传播
        outputs = net(Xtrain)        
        loss = criterion(outputs,Ytrain)
        loss.backward()
        optimizer.step()
        
        # 每训练3个batch打印一次平均loss
        sum_loss += loss.item()
        if i % 3 == 2:
            print('[%d, %d] loss: %.08f'
                  %((epoch+1),i+1,sum_loss/3))
            sum_loss = 0.0
            
    # 每跑完一次epoch测试一下准确率(测试集)
    with torch.no_grad():
        correct = 0
        total = 0
        save_score = np.zeros([1,1])
        save_label = np.zeros([1,1])
        for data in dev_loader:
            Xdev, Ydev = data
            Xdev = Xdev.type(torch.FloatTensor)
            Ydev = Ydev.type(torch.LongTensor)
            Xdev, Ydev = Xdev.to(device), Ydev.to(device)
            outputs = net(Xdev)
                
            #计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += Ydev.size(0)
            correct += (predicted==Ydev).sum()
            #计算AUROC
            score = np.array(outputs.data.cpu())
            score_1 = score[:,1]
            score_1 = score_1[:,np.newaxis]
            label = np.array(Ydev.cpu())
            label = label[:,np.newaxis]
            #保存score和label
            if save_score.shape[0] == 1:
                save_score = np.copy(score_1)
                save_label = np.copy(label)
            else:
                save_score = np.concatenate((save_score,score_1),axis=0)
                save_label = np.concatenate((save_label,label),axis=0)
                    
     
        ROC = metrics.roc_auc_score(save_label,save_score)
        print('第%d个epoch的识别准确率为(验证集)：%.2f%%,AUROC:%.2f%%' % (epoch + 1, (100.0 * correct / total),
            (100*ROC)))
        dev_auc.append(ROC)
        scores_perepoch.append(save_score)
        if ROC>compare_dev_auc:
            compare_dev_auc = ROC
            file_path = save_path + '/' + str(ROC) + '.pkl'
            torch.save(net.state_dict(), file_path)
            print('save model done!')

    
       # 每跑完一次epoch测试一下准确率(训练集)    
    with torch.no_grad():
        correct = 0
        total = 0
        for data in train_loader:
            Xtrain, Ytrain = data
            Xtrain = Xtrain.type(torch.FloatTensor)
            Ytrain = Ytrain.type(torch.LongTensor)
            Xtrain, Ytrain = Xtrain.to(device), Ytrain.to(device)
            outputs = net(Xtrain)
            
            #计算准确率
            _, predicted = torch.max(outputs.data, 1)
            total += Ytrain.size(0)
            correct += (predicted==Ytrain).sum()
        print('第%d个epoch的识别准确率为(训练集)：%.2f%%' % (epoch + 1, (100.0 * correct / total)))

#dev_auc = np.array(dev_auc)
#file_path = save_path + '/auc.npy'
#np.save(file_path,dev_auc)
#scores = np.array(scores_perepoch)   
#file_path = save_path + '/scores.npy'     
#np.save(file_path,scores)
