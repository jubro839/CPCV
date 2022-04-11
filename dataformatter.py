# Dataloader 
    # Scaler 부분 추가 

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Dataset
import numpy as np 


## Mostly adapted from original TFT Github, data_formatters
class Datasetformatter(Dataset):
    
    def __init__(self, data_iter):
        '''
        data_iter : data iterator that holds (x,y) 
        '''
        
        self.X =  data_iter['X']
        self.Y = data_iter['Y']       
        self.X_date = data_iter['X_date']
        self.Y_date = data_iter['Y_date']
        # date to numpy index 
        # a,b = self.inputs_date.shape # a: data point, b : input seq
        # self.inputs_date_arr = np.arange(a*b).reshape(a,b)
        
        # a,b = self.outputs_date.shape # a: data point, b : output seq
        # self.outputs_date_arr = np.arange(a*b).reshape(a,b)
        
        self.sampled_data = {
        'X': self.X,
        'Y': self.Y,
        'X_date': self.X_date,
        'Y_date': self.Y_date
        }
                
    # def __getitem__(self, index):
    #     s = {
    #     'inputs': self.inputs[index],
    #     'outputs': self.outputs[index],
    #     'inputs_date': self.inputs_date[index],
    #     'outputs_date': self.outputs_date[index]
    #     } 
    # Dataloader 에 들어갈 때, datetime 은 넣을수가 없다. 
    
    def __getitem__(self, index):
        
        s = {
        'X': torch.from_numpy(self.X[index]).float(),
        'Y': torch.from_numpy(self.Y[index]).float(),
        'X_date': torch.from_numpy(self.X_date[index]),
        'Y_date': torch.from_numpy(self.Y_date[index])
        }
            
        return s
    
    def __len__(self):
        return self.X.shape[0]



def get_dataloader(X, Y, X_date, Y_date, batch_size, workers_num = 2, shuffle = False):
    '''
    Args:
        X (numpy) : Feature values
        Y (numpy) : Label values
        num_workers : number of multiprocessing for CPU
    '''    
    data_iter = {"X": X,
                 "Y": Y,
                 "X_date": X_date,
                 "Y_date": Y_date}
    data_iter_out = Datasetformatter(data_iter)
    dataloader = DataLoader(data_iter_out, batch_size = batch_size, num_workers = workers_num, shuffle = shuffle, drop_last=True) 
    return dataloader
    









