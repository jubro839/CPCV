import pickle
import pandas as pd
import numpy as np
import os
import scipy.sparse as sp
import torch
from torch import nn
from scipy.sparse import linalg
from torch.utils.data import Dataset
from torch.utils.data import DataLoader,Dataset, Subset



class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data): # series 예측이라면 필요 or 필요 없음 
        return (data * self.std) + self.mean


def generate_xy_seq(df: pd.DataFrame, x_seq = 66, y_seq = 22):
    """
    Generate samples from
    :param df:
    :param x_seq:
    :param y_seq:
    :param scaler:
    :return:
    # x: (epoch_size, input_length, num_nodes, input_dim)
    # y: (epoch_size, output_length, num_nodes, output_dim)
    """
    num_samples, num_nodes = df.shape
    dates_arr = np.array(df.index)
    data = np.expand_dims(df.values, axis = -1) # df -> array [N, F, 1]

    x_offsets = np.arange(-x_seq+1, 1)  
    y_offsets = np.arange(1, y_seq+1)

    # feature_list = [data]

    x, y = [], []
    x_date, y_date = [],[]

    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))

    for t in range(min_t, max_t):
        # value seperation
        x.append(data[t+x_offsets, ...])
        y.append(data[t+y_offsets, ...])
        # date seperation
        x_date.append(dates_arr[t+x_offsets])
        y_date.append(dates_arr[t+y_offsets])
        
    x = np.squeeze(np.stack(x, axis = 0))
    y = np.squeeze(np.stack(y, axis = 0))

    x_date = np.stack(x_date, axis = 0)
    y_date = np.stack(y_date, axis = 0)

    return x, y, x_date, y_date