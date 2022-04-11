
import pandas as pd
import numpy as np 
from itertools import combinations

# combination 
# total_split_num = 10
# val_split_num = 2

class TrainValidPathNum():
    
    def __init__(self,
                 total_split_num: int = 10,
                 val_split_num = 2):
        
        self.total_split_num = total_split_num
        self.val_split_num = val_split_num
        self.path_fold_num, self.path_num = self.path_generation()
        self.total_path_train = self.path_train_generation()
        self.total_path_valid = self.path_valid_generation()
        
    # Path Generation 
    def path_generation(self):
        splits = [i for i in range(self.total_split_num)]
        val_comb = list(combinations(splits, self.val_split_num))

        self.train_split_num = self.total_split_num - self.val_split_num # train 에 사용되는 block 개수 

        path_fold_num  = self.train_split_num + 1 # 한 path 에 존재하는 fold 의 개수 / train_split_num + 1  = 5
        path_num = int(len(val_comb) * self.val_split_num / self.total_split_num) # 전체 path 의 개수 = path_fold_num 

        return path_fold_num, path_num
    
    # Path 별 train number 
    def path_train_generation(self):
        '''
        Path 별 Train model index

        1. Path 별 Train model index 정하기 
            1) path 별 block 개수를 정한다. (전체 split 개수 - test split 개수)
            2) 전체 path 개수를 정한다. 
            3) path 별 model train 의 index 를 결정해준다. 
                - i-th path 의 first value : i 
                - 각 path 의 j-th value : (j-1)th value + train_split_num - (j-1)
                - i-th path 의 (i+1)-th value 부터는 path_i[i] + 1

        2. Path 별 하나의 Train 별 test set index 

            하나의 train 에 대해서 test set 두개 존재 
            두개중 어떤 것이 해당 path 에 해당되는 것인지 확인 

        return 
            {0: [0, 1, 2, 3, 4, 5, 6, 7, 8],
            1: [1, 9, 10, 11, 12, 13, 14, 15, 16],
            2: [2, 10, 17, 18, 19, 20, 21, 22, 23],
            3: [3, 11, 18, 24, 25, 26, 27, 28, 29],
            4: [4, 12, 19, 25, 30, 31, 32, 33, 34],
            5: [5, 13, 20, 26, 31, 35, 36, 37, 38],
            6: [6, 14, 21, 27, 32, 36, 39, 40, 41],
            7: [7, 15, 22, 28, 33, 37, 40, 42, 43],
            8: [8, 16, 23, 29, 34, 38, 41, 43, 44]}

        '''
        total_path_train = {}
        for path_ind in range(self.path_num): # 
            
            path_ls = []
            if path_ind == 0 :   
                firstfold = [i for i in range(self.path_fold_num)]
                firstfold = [x  for x in firstfold]
                total_path_train[path_ind] = (firstfold)
                
            else:
                path_ls = [0 for _ in range(self.path_fold_num)] # [0, 0, 0, 0, 0]
                path_ls[0] = path_ind
            
                for minus in range(self.train_split_num):
                    ind = minus + 1
                    if ind <= path_ind: # ind : 채워 넣고자 하는 위치
                        path_ls[ind] = path_ls[ind-1] + self.train_split_num - minus
                    else: # ind > path_ind 
                        path_ls[ind] = path_ls[ind-1] + 1
                path_ls = [x for x in path_ls]
                total_path_train[path_ind] = path_ls
        return total_path_train
            
    def path_valid_generation(self):
    # Path 별 Valid set 
        """
        Find Validation index number(0 or 1) per Path Number.
        There are two validation data set per each training data set.
        Each Test data set belongs to different Path.
        Clearify Path number that each test set(0 or 1) belongs to for every independent training set. 
        
        key: Path number 
        value: 0-th or 1-th test set for each training dataset belonged to Path
        return 
            {0: [[0, 1], 1, 1, 1, 1, 1, 1, 1, 1],
            1: [0, [0, 1], 1, 1, 1, 1, 1, 1, 1],
            2: [0, 0, [0, 1], 1, 1, 1, 1, 1, 1],
            3: [0, 0, 0, [0, 1], 1, 1, 1, 1, 1],
            4: [0, 0, 0, 0, [0, 1], 1, 1, 1, 1],
            5: [0, 0, 0, 0, 0, [0, 1], 1, 1, 1],
            6: [0, 0, 0, 0, 0, 0, [0, 1], 1, 1],
            7: [0, 0, 0, 0, 0, 0, 0, [0, 1], 1],
            8: [0, 0, 0, 0, 0, 0, 0, 0, [0, 1]]}
        """
        total_path_valid = {}
        for path_ind in range(self.path_num):
            a = [0 for _ in range(self.total_split_num-1)] # 각 path 에 대해서 test set 의 default index 0 저장 -> 추후 값 변경 됨
            for fold_ind in range(self.total_split_num-1): # 각 path 에 대한 fold index
                if fold_ind < path_ind: # fold index 가 path index 보다 아래라면 test set 은 0 번째 것으로 사용
                    a[fold_ind] = 0        
                elif path_ind == fold_ind: # fold index 가 path index 와 같다면 -> 두개의 test set 모두 해당 됨
                    a[fold_ind] = [0,1]
                else: # fold_ind > path_ind: 1번째 것
                    a[fold_ind] = 1
            total_path_valid[path_ind] = a
        return total_path_valid
    
    def train_valid_path(self):
        # search_path_num    
            # train, valdation index 별 path 번호 
        """
        Path number by training, validation index number

        return 
        dcit {(0, 0): 0,
        (0, 1): 0,
        (1, 1): 0,
        (2, 1): 0,
        (3, 1): 0,... }
        
        """
        tr_val_ind_path = {} 
        for path_ind in range(self.path_num):
            for fold_ind in range(self.total_split_num - 1):
                tr_ind = list(self.total_path_train.values())[path_ind][fold_ind]
                ts_ind = list(self.total_path_valid.values())[path_ind][fold_ind]
                if type(ts_ind) == list:
                    for ts in ts_ind:
                        tr_val_ind_path[tr_ind, ts] = path_ind
                else:
                    tr_val_ind_path[tr_ind, ts_ind] = path_ind
        return tr_val_ind_path