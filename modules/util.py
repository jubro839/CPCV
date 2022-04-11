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


