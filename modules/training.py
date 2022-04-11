import argparse
import pandas as pd
from itertools import combinations

import CV.CPCV as CPCV
from engine import trainer

from CV.generate_training_data import generate_xy_seq



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "./data/data_input_demo.csv", help = "data loading path")
    parser.add_argument("--seq_length_x", type=int, default=12, help="Past Sequence Length.",)
    parser.add_argument("--seq_length_y", type=int, default=12, help="Future Sequence Length.",)
    
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    parser.add_argument("--total_split_num", type=int, default=6, help="total split number for each data fold")
    parser.add_argument("--val_split_num", type=int, default=2, help="validation split number for each fold")
    parser.add_argument("--pct_embargo", type=float, default=0.01, help="embargo ratio")
    
    parser.add_argument("--num_layers", type=int, default=1, help="number layers")
    parser.add_argument("--device", type=str, default="cuda:0", help= "device name")
    
    parser.add_argument("--epoch_num", type=int, default=10, help="epoch number")
    args = parser.parse_args()

    # Data Loading  
    df = pd.read_csv(args.data_dir, index_col = [0])
    df = df.set_index(['date'])[['13ty_index', 'interty_index', 'lty_index', 'mbs_index',\
        '13cy_index', 'intercy_index', 'lcy_index', 'ty_index', 'cy_index','agg_index']]
    df.index = pd.to_datetime(df.index, format = '%Y-%m-%d')
    
    data_params = {"seq_length_x": args.seq_length_x,
                 "seq_length_y": args.seq_length_y,
                 "date_from": "1997-05-19", 
                 "date_to":  "2021-11-05"}
       
    dataloader_params = {'batch_size': args.batch_size,
                 'shuffle':True,
                 'device':args.device}
    
    model_params = {
        "num_assets": df.shape[-1], 
        "past_seq": args.seq_length_x, 
        "future_seq": args.seq_length_y, 
        "num_layers": args.num_layers,
        "device": args.device
    }
    
    cpcv_parms = {"total_split_num": args.total_split_num, 
                 "val_split_num": args.val_split_num, 
                 "pct_embargo": args.pct_embargo,
                 "purge_behind": False}

    # if os.path.exists(args.output_dir):
    #     reply = str(input(f'{args.output_dir} exists. Do you want to overwrite it? (y/n)')).lower().strip()
    #     if reply[0] != 'y': exit
    # else:
    #     os.makedirs(args.output_dir)
    
    # date slicing 
    df = df.loc[data_params['date_from'] : data_params['date_to']]
    
    # X, Y generation
    X, Y, x_date, y_date = generate_xy_seq(df, x_seq = data_params["seq_length_x"], y_seq = data_params["seq_length_y"])
    
    
    # CPCV Loading
        # CPCV Params 
    sample_info_sets = pd.Series(index=df[:-(data_params["seq_length_x"] + data_params["seq_length_y"]-1)].index, data=df[(data_params["seq_length_x"] + data_params["seq_length_y"]-1):].index)
    
    folds = [i for i in range(args.total_split_num)]
    val_comb = list(combinations(folds, args.val_split_num))
    train_split_num = args.total_split_num - args.val_split_num
    path_num = int(len(val_comb) * args.val_split_num / args.total_split_num) # 전체 path 의 개수 = path_fold_num 
    
    print ("total split number: {} , total path number: {}".format(len(val_comb), len(path_num)))
    
    # Loading Splitting iterator
    cv_split_iterator = CPCV.LoadingCPCV(X,Y, sample_info_sets, cpcv_parms)
    
    # Training 
        # for loop for each cv_split
    total_loss = [0 for _ in range(path_num)]

    for idx, (path_ind_ls, train_ind, valid_ind_set) in enumerate(cv_split_iterator): # 결국 순서대로 -> 순서 자체가 train model number
        
        print ('Split Number:', idx)
        
        # Model Loading 
        # Loss, optimizer, Dataloader(with scaler)
        
        print ("DataLoader generation...")
        engine = trainer(X, Y, x_date, y_date, path_ind_ls, train_ind, valid_ind_set, path_num, args.device, model_params, dataloader_params)
        
        # Training
        print ("Training...")
        for epoch in range(args.epoch_num):
            train_loader = engine.loader_set['train_loader']
            for train_x, train_y, dates in train_loader.get_iterator():
                engine.train(train_x, train_y)
    
        # Validation 
        print ("Validation...")
        valid_loader_set = engine.loader_set['valid_loader']
        for path_num in list(valid_loader_set.keys()): 
            n_valid_loader = valid_loader_set[path_num]
            for valid_x, valid_y, dates in n_valid_loader.get_iterator():
                val_loss = engine.eval(valid_x, valid_y)
                total_loss[path_num] += val_loss
    
    for i in range(len(total_loss)):
        total_loss[i] /= args.total_split_num
    