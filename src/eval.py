# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset
from model1 import CarboxyNet
from model2 import DenseNet
from torch.utils.data import DataLoader
import argparse
#from sklearn.preprocessing import StandardScaler

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-tp',
        '--TestSet_Path',
        help='测试集的路径，如test/',
    )
    parser.add_argument(
        '-sp',
        '--Save_Score_Path',
        help='保存score的路径，默认为score.npy',
        default = 'score.npy',
    )



    return parser.parse_args()

class DealTestset(Dataset):

    def __init__(self):
        x_test = np.load('Xtest.npy')
        self.x_data = torch.from_numpy(x_test)
        self.len = x_test.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len

def preprocess(test_path):
    print('开始数据预处理')
    file =open('test.csv','r')
    lines=file.readlines()
    file.close()
    row=[]
    names=[]
    for line in lines:
        row.append(line.split(','))
    for col in row:
        names.append(col[0])
        
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
        data_name = test_path + name+'.npz'
        npz_readin = np.load(data_name)      
        temp =npz_readin['voxel']
        Xtest[index,:,:,:] = temp[x_begin:x_end,y_begin:y_end,z_begin:z_end]
        index = index+1
        
    Xtest = np.reshape(Xtest,[117,1,32,32,32])    
        
#    #读取train_dev
#    file =open('train_val.csv','r')
#    lines=file.readlines()
#    file.close()
#    row=[]
#    names=[]
#    for line in lines:
#        row.append(line.split(','))
#    for col in row:
#        names.append(col[0])
        
#    Xtrain = np.zeros((465,x,y,z))
#    index = 0
#    x_begin = int((100-x)/2)
#    x_end = x_begin + x
#    y_begin = int((100-y)/2)
#    y_end = y_begin + y
#    z_begin = int((100-z)/2)
#    z_end = z_begin + z
#    for name in names:
#        data_name = 'train_val/'+name+'.npz'
#        npz_readin = np.load(data_name)      
#        temp =npz_readin['voxel']
#        Xtrain[index,:,:,:] = temp[x_begin:x_end,y_begin:y_end,z_begin:z_end]
#        index = index+1
#        
#    Xtrain = np.reshape(Xtrain,[465,32*32*32])
#    Xtest = np.reshape(Xtest,[117,32*32*32])
#    
#    ss = StandardScaler()
#    Xtrain = ss.fit_transform(Xtrain)
#    Xtest = ss.transform(Xtest)
#    
#    Xtrain  = np.reshape(Xtrain,[465,1,32,32,32])
#    Xtest = np.reshape(Xtest,[117,1,32,32,32])
    
    
#    np.save('data_zscore/Xtrain_dev.npy',Xtrain)
    np.save('Xtest.npy',Xtest)
    
    print('数据预处理完成')


def main():
    #获取输入参数
    args = parse_args()
    preprocess(args.TestSet_Path)
    # 定义是否使用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义测试批处理数据
    dealTestset = DealTestset()
    test_loader = DataLoader(dataset=dealTestset,
                              batch_size=8,
                              shuffle=False
                              )
    print('开始加载训练好的模型')
    net1 = CarboxyNet().to(device)
    net2 = DenseNet().to(device)
    net1.load_state_dict(torch.load('para1.pkl',map_location=device))
    net2.load_state_dict(torch.load('para2.pkl',map_location=device))
    print('模型加载完成')
    
    x_test = np.load('Xtest.npy')
    x_test = torch.from_numpy(x_test)
    x_test = x_test.type(torch.FloatTensor)
    
    torch.manual_seed(11)
    save_score1 = np.zeros((1,1))
    print('正在推理')
    with torch.no_grad():
        for data in test_loader:
            Xtest = data
            Xtest = Xtest.type(torch.FloatTensor)
            Xtest = Xtest.to(device)
            outputs = net1(Xtest)
                
            #预测
            _, predicted = torch.max(outputs.data, 1)
            #保存score
            score = np.array(outputs.data.cpu())
            score_1 = score[:,1]
            score_1 = score_1[:,np.newaxis]

            if save_score1.shape[0] == 1:
                save_score1 = np.copy(score_1)
            else:
                save_score1 = np.concatenate((save_score1,score_1),axis=0)
                
    
    save_score2 = np.zeros((1,1))
    with torch.no_grad():
        for data in test_loader:
            Xtest = data
            Xtest = Xtest.type(torch.FloatTensor)
            Xtest = Xtest.to(device)
            outputs = net2(Xtest)
                
            #预测
            _, predicted = torch.max(outputs.data, 1)
            #保存score
            score = np.array(outputs.data.cpu())
            score_1 = score[:,1]
            score_1 = score_1[:,np.newaxis]

            if save_score2.shape[0] == 1:
                save_score2 = np.copy(score_1)
            else:
                save_score2 = np.concatenate((save_score2,score_1),axis=0)
    
    merge_score = save_score1*0.5 + save_score2*0.5
    print(merge_score)
    np.save(args.Save_Score_Path,merge_score)
    message = '已将score保存至' + args.Save_Score_Path
    print(message)
    return merge_score

if __name__ == '__main__':
    score = main()