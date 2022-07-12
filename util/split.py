import os
from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def group_split(df, groups, train_size=0.7, seed=42):
    """
    Split a dataframe in train and test.
    
    Args:
        df (pandas.DataFrame): dataframe with dataset info and values
        groups (pandas.Series): series with group values
        train_size (float): size o train split
        seed (int): seed for split random state

    Returns:
        train_idx (list): The training set indices for that split.
        test_idx (list): The testing set indices for that split.
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    gss.get_n_splits()
    train_idx, test_idx  = next(gss.split(df, df, groups))
    
    return train_idx, test_idx

def save_dataframes(dfs_dict, save_folder):
    for name, df in dfs_dict.items():
        fname = os.path.join(save_folder, name +'.csv')
        df.to_csv(fname, index=False)

def main():
    parser = argparse.ArgumentParser(description='Spliting CheXpert Dataset')
    parser.add_argument('--csv-path',default='data/CheXpert-v1.0-small/train.csv', type=str, help='path to dataset')
    parser.add_argument('--save-dir',default='data', type=str, help='dir to save csvs')
    parser.add_argument('--train-size', type=float, default=0.7)
    parser.add_argument('--test-size', type=float, default=0.2)

    args = parser.parse_args()

    assert os.path.exists(args.csv_path), args.csv_path
    df = pd.read_csv(args.csv_path)

    # get patient ID
    df['id'] = df['Path'].str.findall('[0-9]{5}').str.get(0).astype(int)

    # train + (test+val) split
    split1 = args.train_size

    # test val split
    split2 = args.test_size/(1-args.train_size) 
    
    # spliting data indexes
    train_idx, tmp_idx = group_split(df, df['id'], train_size=split1)
    df_tmp = df.loc[tmp_idx].reset_index(drop=True)
    test_idx, val_idx = group_split(df_tmp,df_tmp['id'], train_size=split2)

    dfs_dict = {'train':df.loc[train_idx].reset_index(drop=True), 
        'test': df_tmp.loc[test_idx].reset_index(drop=True),
        'val': df_tmp.loc[val_idx].reset_index(drop=True)}

    Path(args.save_dir).mkdir(exist_ok=True)
    save_dataframes(dfs_dict, args.save_dir)

if __name__ == '__main__':
    main()
