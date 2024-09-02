import h5py
import numpy
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr'][idx] / 255., f['hr'][idx] / 255.
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len[f['lr']]

class EvalDataset(Dataset):
    def __init__(self,h5_file):
        super().__init__()
        self.h5_file = h5_file
    def __getitem__(self,idx):
        with h5py.File(self.h5_file,'r') as f:
            return f['lr'][idx] / 255., f['hr'][idx] / 255.
    def __len__(self):
        with h5py.File(self.h5_file,'r') as f:
            return len[f['lr']]