import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
import torch
#from sklearn.preprocessing import StandardScaler
import time

def Transform(Xtrain,Ytrain,uprate=1.5,option = 'mixup'):
    '''
    数据增强的函数
    将数据扩大到原先的uprate倍，
    包括原数据+(uprate-1)倍的transform后的数据
    option:数据增强的方式
    '''
    a = 0.25 #mixup的系数
    bn = Xtrain.shape[0]
    uprate_bn = int(bn*(uprate-1))
    shuffle_idx = np.array(range(0,bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:,np.newaxis]
    #选出要进行transform的数据
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn,1]).astype('float64')
    for i in range(uprate_bn):
        #选择前后两个数据进行mixup
        curr_idx = shuffle_idx[i]
        b_idx = (curr_idx-1) % bn
        a_idx = (curr_idx+1) % bn
        Xtrain_trans[i] = a*Xtrain[b_idx] + (1-a)*Xtrain[a_idx]
        Ytrain_trans[i] = a*Ytrain[b_idx] + (1-a)*Ytrain[a_idx]
        
    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain,Xtrain_trans),0)
    Ytrain = np.concatenate((Ytrain,Ytrain_trans),0)
    
    return Xtrain,Ytrain

def Transform2(Xtrain,Ytrain,uprate=1.5):
    '''
    旋转90度
    '''
    bn = Xtrain.shape[0]
    uprate_bn = int(bn*(uprate-1))
    shuffle_idx = np.array(range(0,bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:,np.newaxis]
    #选出要进行transform的数据
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn,1]).astype('float64')
    for i in range(uprate_bn):
       
        curr_idx = shuffle_idx[i]
        Xtrain_trans[i] = np.rot90(Xtrain[curr_idx],k=1,axes=(2,3))
        Ytrain_trans[i] = Ytrain[curr_idx]
        
    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain,Xtrain_trans),0)
    Ytrain = np.concatenate((Ytrain,Ytrain_trans),0)
    
    return Xtrain,Ytrain[:,0]

def Transform3(Xtrain,Ytrain,uprate=1.5):
    '''
    旋转180度
    '''
    bn = Xtrain.shape[0]
    uprate_bn = int(bn*(uprate-1))
    shuffle_idx = np.array(range(0,bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:,np.newaxis]
    #选出要进行transform的数据
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn,1]).astype('float64')
    for i in range(uprate_bn):
       
        curr_idx = shuffle_idx[i]
        Xtrain_trans[i] = np.rot90(Xtrain[curr_idx],k=1,axes=(1,2))
        Ytrain_trans[i] = Ytrain[curr_idx]
        
    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain,Xtrain_trans),0)
    Ytrain = np.concatenate((Ytrain,Ytrain_trans),0)
    
    return Xtrain,Ytrain[:,0]   

def Transform4(Xtrain,Ytrain,uprate=1.5):
    '''
    镜像
    '''
    bn = Xtrain.shape[0]
    uprate_bn = int(bn*(uprate-1))
    shuffle_idx = np.array(range(0,bn))
    np.random.shuffle(shuffle_idx)
    Ytrain = Ytrain[:,np.newaxis]
    #选出要进行transform的数据
    Xtrain_trans = np.zeros([uprate_bn,
                             1,
                             Xtrain.shape[2],
                             Xtrain.shape[3],
                             Xtrain.shape[4]]).astype('float64')
    Ytrain_trans = np.zeros([uprate_bn,1]).astype('float64')
    for i in range(uprate_bn):
       
        curr_idx = shuffle_idx[i]
        tmp = Xtrain[curr_idx]
        tmp = tmp[np.newaxis,:,:,:,:]
        for j in range(32):
            Xtrain_trans[i,0,j,:,:] = tmp[0,0,31-j,:,:]
        Ytrain_trans[i] = Ytrain[curr_idx]
        
    Ytrain = Ytrain.astype('float64')
    Xtrain = np.concatenate((Xtrain,Xtrain_trans),0)
    Ytrain = np.concatenate((Ytrain,Ytrain_trans),0)
    
    return Xtrain,Ytrain[:,0]   

def preprocess():
    '''
    划分验证集和训练集
    将标准化后的训练集、验证集、测试集保存在本地
    '''
    print('Preprocess Begin!')
    
    Xtrain_dev = np.load('data_origin/Xtrain_dev.npy')
    Ytrain_dev = np.load('data_origin/Ytrain_dev.npy')
    Xtest = np.load('data_origin/Xtest.npy')

    
    #划分验证集和训练集
    kf = KFold(n_splits=5,shuffle=True,random_state=int(time.time()))
    for train_idx,dev_idx in kf.split(Ytrain_dev):
        tmp1 = train_idx
        tmp2 = dev_idx
    Xtrain = Xtrain_dev[tmp1]
    Ytrain = Ytrain_dev[tmp1]
    Xdev = Xtrain_dev[tmp2]
    Ydev = Ytrain_dev[tmp2]
        
    #标准化
#    ss = StandardScaler()
#    Xtrain = ss.fit_transform(Xtrain)
#    Xdev = ss.transform(Xdev)
#    Xtest = ss.transform(Xtest)
    
    
    #保存在本地
    np.save('data_origin/Xtrain.npy',Xtrain)
    np.save('data_origin/Ytrain.npy',Ytrain)
    np.save('data_origin/Xdev.npy',Xdev)
    np.save('data_origin/Ydev.npy',Ydev)
    np.save('data_origin/Xtest.npy',Xtest)
    
    print('Preprocess Done!')
    

    
    
class DealTrainset(Dataset):

    def __init__(self):
        x_train = np.load('data_zscore/Xtrain.npy')
        y_train = np.load('data_zscore/Ytrain.npy')
        #数据增强
        #x_train, y_train = Transform4(x_train,y_train,1.3)
        
        self.x_data = torch.from_numpy(x_train)
        self.y_data = torch.from_numpy(y_train)
        self.len = y_train.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

class DealDevset(Dataset):

    def __init__(self):
        #_, _, x_dev,y_dev = loadTrainDev()
        x_dev = np.load('data_zscore/Xdev.npy')
        tmp_y_dev = np.load('data_zscore/Ydev.npy')
        tmp_y_dev = tmp_y_dev[:,np.newaxis]
        y_dev = np.zeros([tmp_y_dev.shape[0],2])
        y_dev[:,0] = tmp_y_dev[:,0]
        y_dev[:,1] = 1-tmp_y_dev[:,0]
        self.x_data = torch.from_numpy(x_dev)
        self.y_data = torch.from_numpy(y_dev)
#        self.y_data = self.y_data.type(torch.FloatTensor)  # 转Float
        self.len = y_dev.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
class DealTestset(Dataset):

    def __init__(self):

        x_test = np.load('data_zscore/Xtest.npy')
        self.x_data = torch.from_numpy(x_test)
        self.len = x_test.shape[0]
    
    def __getitem__(self, index):
        return self.x_data[index]

    def __len__(self):
        return self.len