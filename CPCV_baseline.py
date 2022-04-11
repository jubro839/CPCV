def get_splits(n_groups, n_tests):
    return 15

def get_paths(n_groups, n_tests):
    return 5

def trainNum_to_foldNum(train_num):
    


class CPCV:
    def __init__(self, n_groups, n_tests):
        pass


    def search_path_num(tr_ind, ts_ind):
        """ Search path numbers based on training index(fold number)

        Args:
            tr_ind (int): fold index among entire fold number    
                from enumerate
            ts_ind (int): 0 or 1 
        
            path_tr_ts (dict) : path_tr_ts['train'][tr_ind]['test'][ts_ind]
        return: path_num
        """
        
        pathNum = path_tr_ts['train'][tr_ind]['test'][ts_ind]
        
        
        

    def split(self, n_observations):
        """yield
            return: (path_index1, path_index2), (train_set, test_set)
        """
        pass

#class AMOMDataLoader

# Trainset 기준으로 combination 이 짜져야 한다. 


if __name__ == '__main__':
    n_groups = 6
    n_tests = 2
    n_splits = get_splits(6, 2)
    n_paths = get_paths(6, 2)
    results = [0 for _ in range(n_paths)] # [0, 0, 0, ..., 0]

    total_loss = [0 for i in range(path_num)]

    # for idx, path, data in enumerate(CPCV.split()):
    #     """
    #     paths: [path_index1, path_index2]
    #     data: [train_set, test_set]
    #         train_set: [train_indexes]
    #         test_sets: [[group1_indexes], [group2_idnexs]]
    #     """
    for train_ind, (train, valid_set) in enumerate(gen): # 결국 순서대로 -> 순서 자체가 train model number
        i +=1

        print ('i:', i)
        # MODEL INIT

        # TRAIN     
            # TRAIN Dataloader generation 
        pred_tr = Model(TRAIN)
        loss_tr = criterion(pred_tr, label)
        
        # TEST 
            # TEST Dataloader generation 
        for val_ind, valid in enumerate(valid_set):
            loss_val = Model(valid)
            path_ind = train_valid_path(train_ind, val_ind)
            
            total_loss[path_ind] += loss_val

    # GET RESULTS            
    for i in range(len(total_loss)):
        total_loss[i] /= n_test


        
