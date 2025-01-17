import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr'][idx], f['hr'][idx] # 归一化处理还需要修改
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len(f['lr'])

class EvalDataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr'][str(idx)][ : , : ], f['hr'][str(idx)][ : , : ] # 归一化处理还需要修改
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len(f['lr'])