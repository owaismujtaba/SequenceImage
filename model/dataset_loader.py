import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import WeightedRandomSampler

import config
import pdb
import os
from skimage import img_as_float, img_as_ubyte
from skimage import io
from sklearn.preprocessing import OneHotEncoder
import torch

def weighted_sampler(train):
    '''
    Args:
      train: train dataset
    return:
      weighted_sampler: sampeler to draw equal proportion of samples as we have imblanced dataset 18000 from class 1
      and 42000 from class 0
    '''
    targets = train.labels
    class_count = np.unique(targets, return_counts=True)[1]
    weight = 1. / class_count
    samples_weight = weight[targets]
    samples_weight = torch.from_numpy(samples_weight)
    weighted_sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    return weighted_sampler



def data_loader():
    
    train_data = TumorDataset(kind='train')
    test_data = TumorDataset(kind='test')
        
    '''
    train_size = len(train_data)
    val_size = int(0.1*train_size)
    val_data = Subset(train_data, range(val_size))
    train_data = Subset(train_data, range(val_size, train_size))
    '''
    sampler = weighted_sampler(train_data)
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, sampler=sampler)
    #validation_loader = DataLoader(val_data, batch_size= len(val_data))
    #test_loader = DataLoader(test_data, batch_size=len(test_data))
    validation_loader = DataLoader(test_data, batch_size= config.BATCH_SIZE)
    test_loader = DataLoader(test_data, batch_size=config.BATCH_SIZE)
    
    return train_loader, validation_loader, test_loader



def load_labels_file(kind='train'):
    
    if kind == 'train':
        labels_path = config.TRAIN_DATA_LABELS
        data_path = config.TRAIN_DATA_FOLDER
    else:
        labels_path = config.TEST_DATA_LABELS
        data_path = config.TEST_DATA_FOLDER
    
    labels = pd.read_csv(labels_path)['0'].values
    labels = labels-1
    #pdb.set_trace()
    return labels, data_path
    
    

class TumorDataset(Dataset):
    
    def __init__(self, kind='train'):
        
        self.labels, self.data_dir = load_labels_file(kind)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        img_name = str(idx) + '.png'
        img_path = os.path.join(self.data_dir, img_name)
        
        
        #img = np.empty(shape=(1, 102, 102))
        
        #pdb.set_trace()
        #img = np.empty(shape=(3, 224, 224))
        img = (img_as_float(io.imread(img_path))-0.5)/0.5
        #img = img_as_ubyte(io.imread(img_path))
        #img = img/255
        
        #img[0, :, :] = (img_as_float(io.imread(img_path)) - 0.5)/0.5
        img = torch.from_numpy(img)
        img = img.type(torch.FloatTensor)
        label = self.labels[idx]
        
        return img, label
