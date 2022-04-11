import torch.optim as optim 

from model import *
from dataformatter import DataLoaderSet
from util import StandardScaler


class trainer():
    """
    trainer for every iteration of CV split set
        for each train/validation set by fold
    """
    def __init__(self, X, Y, X_date, Y_date, path_ind_ls, train_ind, valid_ind_set, path_num, device, model_params, loader_params):
        
        # Model Init
        self.model = LSTM_Model(model_params) # Model 부분 수정 
        self.model.to(device)
        self.criterion = nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.01)
        
        # DataLoader 
        LoaderGenerator = DataLoaderSet(X, Y, X_date, Y_date, path_ind_ls, train_ind, valid_ind_set, loader_params)
        self.loader_set = LoaderGenerator.dataloader
        
        # scaler2 for prediction during training
        mean2 = torch.from_numpy(LoaderGenerator.scaler.mean).to(device)
        std2 = torch.from_numpy(LoaderGenerator.scaler.std).to(device)

        self.scaler2 = StandardScaler(mean2, std2)
    
    def train(self, train_x, train_y):
        self.model.train()
        
        train_pred = self.model(train_x)
        train_pred = self.scaler2.inverse_transform(train_pred)
        loss_train = self.criterion(train_pred, train_y)
        loss_train.backward()
        
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def eval(self, val_x, val_y):
        self.model.eval()
        val_pred = self.model(val_x)
        val_pred = self.scaler2.inverse_transform(val_pred)
        
        val_loss = self.criterion(val_pred, val_y)
        
        return val_loss
        
        
            