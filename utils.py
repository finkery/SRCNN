import torch
# 坡度和坡向的计算
def calculate_slope_aspect(dem, cellsize = 30,eval = False):

    dzdy = (torch.roll(dem, -1, dims=1) - torch.roll(dem, 1, dims=1))
    dzdx = (torch.roll(dem, -1, dims=0) - torch.roll(dem, 1, dims=0))
    # Calculate slope
    slope = torch.tan(torch.sqrt((dzdx / (2 * cellsize))**2 + (dzdy / (2 * cellsize))**2))

    # Calculate aspect
    aspect = dzdy / dzdx
    slope = torch.where(torch.isnan(slope) | torch.isinf(slope), torch.zeros_like(slope), slope)  
    aspect = torch.where(torch.isnan(aspect) | torch.isinf(aspect), torch.zeros_like(aspect), aspect)  
    return slope, aspect

def calc_rmse(ground_truth,preds):
    preds = torch.where(torch.isnan(preds) | torch.isinf(preds), torch.zeros_like(preds), preds)  
    ground_truth = torch.where(torch.isnan(ground_truth) | torch.isinf(ground_truth), torch.zeros_like(ground_truth), ground_truth)  
    err = torch.sum((ground_truth - preds) ** 2)
    err /= (preds.shape[0] + 1) * (preds.shape[1] + 1)
    return torch.sqrt(err)

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