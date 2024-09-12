import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *

# 梯度损失
def gradient_loss(y_true, y_pred, alpha=0.5, beta=0.5):

    true_slope, true_aspect = calculate_slope_aspect(y_true)
    pred_slope, pred_aspect = calculate_slope_aspect(y_pred)
    slope_diff = nn.L1Loss()(pred_slope, true_slope)
    aspect_diff = nn.L1Loss()(pred_aspect, true_aspect)
    return alpha * slope_diff + beta * aspect_diff


# 综合损失
def combined_loss(y_true, y_pred):
    y_pred = torch.where(torch.isnan(y_pred) | torch.isinf(y_pred), torch.zeros_like(y_pred), y_pred)  
    y_true = torch.where(torch.isnan(y_true) | torch.isinf(y_true), torch.zeros_like(y_true), y_true)  
    grad_loss = gradient_loss(y_true, y_pred)  # 梯度损失
    mse_loss = nn.L1Loss()(y_true, y_pred)  # MSE损失
    return mse_loss + grad_loss