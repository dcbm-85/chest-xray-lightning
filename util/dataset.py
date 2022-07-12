import os
from pathlib import Path
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from util.imgaug import GetTransforms

class CheXpertDataset(Dataset):
    def __init__(self, csv_path, args, mode='train'):
        self.csv_path = csv_path 
        self.args = args
        self.mode = mode
        self.labels, self.img_paths = self.load_df()
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # Get and transform the label
        label = self.labels.loc[idx].values
        label = torch.FloatTensor(label)

        # Get and transform the images
        img_path = self.img_paths.loc[idx]
        assert os.path.exists(img_path), img_path
        img = Image.open(img_path).convert('RGB')
        img = GetTransforms(img, self.args, self.mode)

        return img, label

    def load_df(self):
        df = pd.read_csv(self.csv_path)
        labels = df[self.args.tasks].fillna(value=0) # only tasks
        labels = labels.replace(-1,1) # U-Ones 
        img_paths = df.iloc[:,0].apply(lambda x: Path(self.args.data_dir) / x)
        return labels, img_paths

class CheXpertDM(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        train_set = CheXpertDataset(self.args.train_csv, self.args, mode='train')
        train_loader = DataLoader(train_set, batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers, 
                                             drop_last=True, shuffle=True,
                                             pin_memory=True)
        return train_loader

    def test_dataloader(self):
        test_set = CheXpertDataset(self.args.test_csv, self.args, mode='test')
        test_loader = DataLoader(test_set, batch_size=self.args.test_batch_size,
                                             num_workers=self.args.num_workers, 
                                             drop_last=False, shuffle=False)
        return test_loader

    def val_dataloader(self):
        val_set = CheXpertDataset(self.args.val_csv, self.args, mode='val')
        val_loader = DataLoader(val_set, batch_size=self.args.batch_size,
                                             num_workers=self.args.num_workers, 
                                             drop_last=False, shuffle=False)
        return val_loader
