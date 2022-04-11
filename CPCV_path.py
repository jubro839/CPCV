
from itertools import combinations
# CPCV Path 
    # block 별로 path counting 하기 

class CPCVPath:
    """
        
    """
    def __init__(self, n_splits, n_valid):
        self.n_splits = n_splits 
        self.n_valid = n_valid
        self.valid_comb = list(combinations([x for x in range(self.n_splits)], self.n_valid))
        self._set_path_indexs()
        
    def _set_path_indexs(self):
        train_path_count = [0 for _ in range(self.n_splits)]
        self.train_path_pair = {}
        
        for val_group in self.valid_comb: 
            path_comb = []
            for split_ind in val_group:
                path_comb.append(train_path_count[split_ind])
                train_path_count[split_ind] += 1
            self.train_path_pair[val_group] = tuple(path_comb)
    
    def get_path(self, valid_group_indices):
        return self.train_path_pair[valid_group_indices]
    
        