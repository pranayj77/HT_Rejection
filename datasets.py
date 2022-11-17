import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import wfdb
import os
from time import time as time
from tqdm import tqdm
import copy
from scipy import signal
import pickle



class PTBDataset(Dataset):
  def __init__(
      self,
      csv = '/content/drive/MyDrive/Pranay - ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2/ptbxl_database_repeat_paired.csv',
      root_dir = '/content/drive/MyDrive/Pranay - ECG/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.2'):
    df = pd.read_csv(csv)
    df['recording_date_y'] = pd.to_datetime(df['recording_date_y']).astype(int)/ 10**9
    df['recording_date_x'] = pd.to_datetime(df['recording_date_x']).astype(int)/ 10**9
    df['recording_date_x'] = (df['recording_date_x']-df['recording_date_x'].min())/(df['recording_date_x'].max()-df['recording_date_x'].min())
    df['recording_date_y'] = (df['recording_date_y']-df['recording_date_y'].min())/(df['recording_date_y'].max()-df['recording_date_y'].min())
    self.data = df
    self.root_dir = root_dir
    
    index = np.where(self.data.strat_fold_x.isin([0,1,2,3,4,5,6,7,8]))
    self.train = self.data.iloc[index]
    index = np.where(self.data.strat_fold_x.isin([9]))
    self.val = self.data.iloc[index]
    index = np.where(self.data.strat_fold_x.isin([10]))
    self.test = self.data.iloc[index]

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.toint()
    fname_1 = self.root_dir+os.sep+self.data.iloc[idx].filename_lr_x
    fname_2 = self.root_dir+os.sep+self.data.iloc[idx].filename_lr_y
    sample_1,head = wfdb.rdsamp(str(fname_1))
    sample_2,head = wfdb.rdsamp(str(fname_2))
    label_1 = self.data["ST-DEPR-MI_x"].iloc[idx]
    label_2 = self.data["ST-DEPR-MI_y"].iloc[idx]
    # sample = sample[np.newaxis,:,:]
    # sample = signal.resample(sample,2500,axis=1)
    sample_1 = np.swapaxes(sample_1,0,1)
    sample_2 = np.swapaxes(sample_2,0,1)
    time = self.data["recording_date_y"].iloc[idx] - self.data["recording_date_x"].iloc[idx]
    return np.squeeze(sample_1),np.squeeze(sample_2),label_1,label_2,time

  def set_fold(self,fold):
    if fold == "Train":
      self.data = self.train
    elif fold == "Val":
      self.data = self.val
    elif fold == "Test":
      self.data = self.test
    else:
      raise Exception(f"No fold named: {fold}")
    return self

