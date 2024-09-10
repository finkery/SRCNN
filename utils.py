import numpy as np
import torch
# 归一化
def normalize_dem(dem):
    min_value = np.min(dem)
    max_value = np.max(dem)
    normalized_dem = (dem - min_value) / (max_value - min_value)
    return normalized_dem, min_value, max_value

# 反归一化
def denormalize_dem(normalized_dem, min_value, max_value):
    original_dem = normalized_dem * (max_value - min_value) + min_value
    return original_dem

# 坡度和坡向的计算
def calculate_slope_aspect(dem, cellsize):

    dzdx = (torch.roll(dem, -1, axis=1) - torch.roll(dem, 1, axis=1)) / (2 * cellsize)
    dzdy = (torch.roll(dem, -1, axis=0) - torch.roll(dem, 1, axis=0)) / (2 * cellsize)
    
    # Calculate slope
    slope = torch.arctan(torch.sqrt(dzdx**2 + dzdy**2))

    # Calculate aspect
    aspect = dzdy / dzdx
    
    return slope, aspect

def calc_rmse(ground_truth,preds):
    err = np.sum((ground_truth.astype(float) - preds.astype(float)) ** 2)
    err /= preds.shape[0] * preds.shape[1]
    return err

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count