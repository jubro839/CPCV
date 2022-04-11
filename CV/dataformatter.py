from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import torch
from torch import nn 
import numpy as np 

import util

class CustomDataLoader(object):
    def __init__(self, xs, ys, x_date, y_date, batch_size, device, shuffle = True, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.device = device
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0) # padding 하고자 하는 만큼 뒤에 추가 
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            
            x_date_padding = np.repeat(x_date[-1:], num_padding, axis = 0)
            y_date_padding = np.repeat(y_date[-1:], num_padding, axis = 0)
            
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
            
            x_date = np.concatenate([x_date, x_date_padding], axis = 0)
            y_date = np.concatenate([y_date, y_date_padding], axis = 0)
            
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        
        self.x_date = x_date
        self.y_date = y_date
        
        if shuffle:
            permutation = np.random.permutation(self.size)
            # print ("permutation : ", len(permutation))
            xs, ys = self.xs[permutation], self.ys[permutation]
            # print ("x_date len:", x_date.shape)
            # print ("y_date len:", y_date.shape)
            
            x_date, y_date = self.x_date[permutation], self.y_date[permutation]
            self.xs = xs
            self.ys = ys
            self.x_date = x_date
            self.y_date = y_date
            
    def get_iterator(self):
        """ Building Iterator
        Returns:
            (iterator): x, y, x_date, y_date 

        Yields:
            (x_i, y_i, (x_date_i, y_date_i))
        """
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = torch.Tensor(self.xs[start_ind: end_ind, ...]).to(self.device)
                y_i = torch.Tensor(self.ys[start_ind: end_ind, ...]).to(self.device)
                
                x_date_i = self.x_date[start_ind: end_ind]
                y_date_i = self.y_date[start_ind: end_ind]
                yield (x_i, y_i, (x_date_i, y_date_i))
                self.current_ind += 1
        return _wrapper()

class DataLoaderSet:
    
    def __init__(self, X, Y, X_date, Y_date, path_ind_ls, train_ind, valid_ind_set, loader_params):
        """ Building train/validation dataloader 
        Args:
            X (array): R^(N, seq, M), where N: data number/ seq: sequence length/ M: number of asset 
            Y (array): R^(N, seq, M)
            X_date (array with datetime)
            Y_date ()
            path_ind_ls (tuple): set of path number corresponding to valid data set
                ex
                    (5, 2, 1)
            train_ind (array): train index array set 
                ex
                    array([1199, 1200, 1201, ..., 6295, 6296, 6297])
            valid_ind_set (_type_): set of N number of validations index 
                ex
                    [array([   0,    1,    2, ..., 1047, 1048, 1049]),
                    array([2100, 2101, 2102, ..., 3147, 3148, 3149]),
                    array([4200, 4201, 4202, ..., 5246, 5247, 5248])]
            loader_params
        Return 
            dataloader dictionary 
        """
        self.X = X
        self.Y = Y 
        self.X_date = X_date
        self.Y_date = Y_date
        
        self.path_ind_ls = path_ind_ls
        self.train_ind = train_ind
        self.valid_ind_set = valid_ind_set
        self.loader_params = loader_params

        self.dataloader = {}
        self._trainloader()
        self._validloader()

    
    def _trainloader(self):

        # trainig
        self.train_X, self.train_Y = self.X[self.train_ind], self.Y[self.train_ind]
        self.train_x_date, self.train_y_date = self.X_date[self.train_ind], self.Y_date[self.train_ind]
        
        # scaling 
        scaling_value = self.train_X.reshape(-1, self.train_X.shape[-1])
        mean, std = scaling_value.mean(axis = 0), scaling_value.std(axis = 0)
        self.scaler = util.StandardScaler(mean, std)
        
        self.train_X = self.scaler.transform(self.train_X)
        # TRAIN Dataloader generation 
        # print ("Train loader generation")
        # train_loader = get_dataloader(train_X, train_Y, batch_size=batch_num, shuffle = True)
        train_loader = CustomDataLoader(self.train_X, self.train_Y, self.train_x_date, self.train_y_date, self.loader_params['batch_size'], self.loader_params['device'], self.loader_params['shuffle'])
        self.dataloader['train_loader'] = train_loader 

    def _validloader(self):

        valid_loader_set = {} # key: path / value: valid loader

        for ix, valid_ind in enumerate(self.valid_ind_set):
            valid_X, valid_Y = self.X[valid_ind], self.Y[valid_ind]
            x_date, y_date = self.X_date[valid_ind], self.Y_date[valid_ind]
            valid_X = self.scaler.transform(valid_X)
            # Custom Validation Loader 
            valid_loader = CustomDataLoader(valid_X, valid_Y, x_date, y_date, self.loader_params['batch_size'],  self.loader_params['device'], self.loader_params['shuffle'])
            path_num = self.path_ind_ls[ix] # path number for each validation set 
            valid_loader_set[path_num] = valid_loader

        self.dataloader['valid_loader'] = valid_loader_set
    