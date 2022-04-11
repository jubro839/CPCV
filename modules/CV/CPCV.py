"""
Implements the Combinatorial Purged Cross-Validation class from Chapter 12
"""
from itertools import combinations
from typing import List

import pandas as pd
import numpy as np

from scipy.special import comb
from sklearn.model_selection import KFold
from CV.cross_validation import ml_get_train_times

def _get_number_of_backtest_paths(n_train_splits: int, n_test_splits: int) :
    """
    Number of combinatorial paths for CPCV(N,K)
    :param n_train_splits: (int) number of train splits
    :param n_test_splits: (int) number of test splits
    :return: (int) number of backtest paths for CPCV(N,k)
    """
    return int(comb(n_train_splits, n_train_splits - n_test_splits) * n_test_splits / n_train_splits)


class CombinatorialPurgedKFold(KFold):
    """
    Advances in Financial Machine Learning, Chapter 12.

    Implements Combinatial Purged Cross Validation (CPCV)

    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in between

    :param n_splits: (int) The number of splits. Default to 3
    :param samples_info_sets: (pd.Series, date value) The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param pct_embargo: (float) Percent that determines the embargo size.
    """

    def __init__(self,
                 n_splits: int = 3,
                 n_valid_splits: int = 2,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.,
                 purge_behind: bool = False):

        if not isinstance(samples_info_sets, pd.Series):
            raise ValueError('The samples_info_sets param must be a pd.Series')
        super(CombinatorialPurgedKFold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.samples_info_sets = samples_info_sets
        self.pct_embargo = pct_embargo
        self.n_test_splits = n_valid_splits
        self.num_backtest_paths = _get_number_of_backtest_paths(self.n_splits, self.n_test_splits) # self. 정의가 안 되어 있는데/./? 
        self.backtest_paths = []  # Array of backtest paths
        self.valid_comb = list(combinations([x for x in range(self.n_splits)], self.n_test_splits))
        self._set_path_indexs() # train 별 path index mapping 
        self.purge_behind = purge_behind
        
    def _set_path_indexs(self):
        train_path_count = [0 for _ in range(self.n_splits)]
        self.val_path_pair = {}
        for val_group in self.valid_comb: 
            path_comb = []
            for split_ind in val_group:
                path_comb.append(train_path_count[split_ind])
                train_path_count[split_ind] += 1
            self.val_path_pair[val_group] = tuple(path_comb)
        

    def _generate_combinatorial_test_ranges(self, splits_indices: dict):
        """
        Using start and end indices of test splits from KFolds and number of test_splits (self.n_test_splits),
        generates combinatorial test ranges splits

        :param splits_indices: (dict) Test fold integer index: [start test index, end test index]
        :return: (list) Combinatorial test splits ([start index, end index])
        """

        # Possible test splits for each fold
        combinatorial_splits = list(combinations(list(splits_indices.keys()), self.n_test_splits))
        combinatorial_test_ranges = []  # List of test indices formed from combinatorial splits
        for combination in combinatorial_splits:
            temp_test_indices = []  # Array of test indices for current split combination
            for int_index in combination:
                temp_test_indices.append(splits_indices[int_index])
            combinatorial_test_ranges.append(temp_test_indices)
        return combinatorial_test_ranges

    def _fill_backtest_paths(self, train_indices: list, test_splits: list):
        """
        Using start and end indices of test splits and purged/embargoed train indices from CPCV, find backtest path and
        place in the path where these indices should be used.

        :param test_splits: (list) of lists with first element corresponding to test start index and second - test end
        """
        # Fill backtest paths using train/test splits from CPCV
        for split in test_splits:
            found = False  # Flag indicating that split was found and filled in one of backtest paths
            for path in self.backtest_paths:
                for path_el in path:
                    if path_el['train'] is None and split == path_el['test'] and found is False:
                        path_el['train'] = np.array(train_indices)
                        path_el['test'] = list(range(split[0], split[-1]))
                        found = True

    # noinspection PyPep8Naming
    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None):
        """
        The main method to call for the PurgedKFold class

        :param X: (pd.DataFrame) Samples dataset that is to be split
        :param y: (pd.Series) Sample labels series
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices]
        """
        if X.shape[0] != self.samples_info_sets.shape[0]:
            raise ValueError("X and the 'samples_info_sets' series param must be the same length")

        test_ranges: [(int, int)] = [(ix[0], ix[-1] + 1) for ix in np.array_split(np.arange(X.shape[0]), self.n_splits)]
            # test_ranges : (st, end ind) for each split in the fold (e.g, fold set)
               
        
        splits_indices = {}
        for index, [start_ix, end_ix] in enumerate(test_ranges):
            splits_indices[index] = [start_ix, end_ix]

        combinatorial_test_ranges = self._generate_combinatorial_test_ranges(splits_indices)
        
        valid_order_comb_map = {} # 0: (0,1), 
        for i in range(len(self.valid_comb)):
            valid_order_comb_map[i] = self.valid_comb[i]
        
        spilt_ind_map = {}
        for i in range(len(self.valid_comb)): 
            spilt_ind_map[i] = self.valid_comb[i] 
        
        # Prepare backtest paths
        for _ in range(self.num_backtest_paths):
            path = []
            for split_idx in splits_indices.values():
                path.append({'train': None, 'test': split_idx})
            self.backtest_paths.append(path)

        embargo: int = int(X.shape[0] * self.pct_embargo)
        self.embargo = embargo
        
        for comb_order, test_splits in enumerate(combinatorial_test_ranges):
                # test_splits : [[0,10], [10, 21], [21, 31]]  test set 만 가지고 있음 
            
            if self.purge_behind:
                cv_set = {}
                val_comb_ind_set = valid_order_comb_map[comb_order]
                valid_set = {}
                train_set = {}
                
                for split_ind, _ in enumerate(test_ranges):
                    if split_ind in val_comb_ind_set: # validation
                        split = test_ranges[split_ind]
                        for st_ind, end_ind in [split]:
                            valid_ind = np.arange(st_ind,end_ind)
                            valid_set[split_ind] = (valid_ind)
                    else: # train
                        split = test_ranges[split_ind]
                        for st_ind, end_ind in [split]:
                            train_ind = np.arange(st_ind, end_ind)
                            train_set[split_ind] = (train_ind)
                cv_set['val'] = valid_set
                cv_set['train'] = train_set    
                
                # Purging & Embargo 
                train_indices, test_indices = self.purging_behind(cv_set)
            else:
                # Embargo
                test_times = pd.Series(index=[self.samples_info_sets[ix[0]] for ix in test_splits], data=[
                    self.samples_info_sets[ix[1] - 1] if ix[1] - 1 + embargo >= X.shape[0] else self.samples_info_sets[
                        ix[1] - 1 + embargo]
                    for ix in test_splits])

                test_indices = []
                for [start_ix, end_ix] in test_splits:
                    test_indices.append(list(range(start_ix, end_ix)))

                # Purge
                train_times = ml_get_train_times(self.samples_info_sets, test_times)

                # Get indices
                train_indices = []
                for train_ix in train_times.index:
                    train_indices.append(self.samples_info_sets.index.get_loc(train_ix))
            
            valid_group_indices = spilt_ind_map[comb_order]
            path_ls = self.val_path_pair[valid_group_indices] # path_ls : [path num1, path num2,..path numN]

            self._fill_backtest_paths(train_indices, test_splits)

            yield path_ls, np.array(train_indices), [np.array(x) for x in test_indices]

    def purging_behind(self, cv_set):
        """_summary_

        Args:
            cv_set (double nested dictionary)
                - keys : ['train', 'val']
                - value : 2nd dict
                    - keys: fold num [0, 1]
                    - values (np) 
        Return: 
            purged train or validation data set
        """
        purging_num = self.embargo
        embargo_num = self.embargo

        val_keys = list(cv_set['val'].keys())
        train_keys = list(cv_set['train'].keys())
        # print ("val_keys ", val_keys)
 
        for val_k in val_keys:
            if val_k-1 in train_keys: # train | val -> val 앞 버리기 
                cv_set['val'][val_k] = cv_set['val'][val_k][purging_num: ]
                if val_k+1 in train_keys: # val | train -> train 앞 버리기 
                    cv_set['train'][val_k+1] = cv_set['train'][val_k+1][embargo_num: ]

            elif val_k+1 in train_keys: # val | train -> train 앞 버리기 
                cv_set['train'][val_k+1] = cv_set['train'][val_k+1][embargo_num: ]
                if val_k-1 in train_keys: # train | val -> val 앞 버리기 
                    cv_set['val'][val_k] = cv_set['val'][val_k][purging_num: ]
        
        train_indices = [x for y in list(cv_set['train'].values()) for x in y]
        test_indices = cv_set['val'].values()

        return train_indices, test_indices
    

def LoadingCPCV(X,Y,sample_info_sets,params):
    """
    X (dataframe): data for X input 
    Y (dataframe): data for Y output 
    sample_info_sets (series): from start date time of X input to end date time of Y output 
    params (dict): parameters dictionary 
    """    

    cv_gen = CombinatorialPurgedKFold(n_splits = params["total_split_num"],
                            n_valid_splits = params["val_split_num"], 
                            samples_info_sets = sample_info_sets, 
                            pct_embargo= params["pct_embargo"],
                            purge_behind= params["purge_behind"])

    cv_split = cv_gen.split(X,Y)
    
    return cv_split