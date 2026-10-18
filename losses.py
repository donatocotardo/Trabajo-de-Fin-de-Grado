import torch
import torch.nn as nn
import torch.nn.functional as F

def get_criterion(loss_type):
    if loss_type == "MSE": 
        return nn.MSELoss()
    elif loss_type == "L1": 
        return nn.L1Loss()
    elif loss_type == "SSIM":
        def ssim_loss(pred, target, C1=0.01**2, C2=0.03**2):
            pred = pred.float()
            target = target.float()

            mu_x = F.avg_pool2d(pred, 3, 1, 0)
            mu_y = F.avg_pool2d(target, 3, 1, 0)

            sigma_x  = F.avg_pool2d(pred * pred, 3, 1, 0) - mu_x ** 2
            sigma_y  = F.avg_pool2d(target * target, 3, 1, 0) - mu_y ** 2
            sigma_xy = F.avg_pool2d(pred * target, 3, 1, 0) - mu_x * mu_y

            ssim_num = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            ssim_den = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
            ssim_map = ssim_num / (ssim_den + 1e-7)

            return 1 - ssim_map.mean()
        return ssim_loss
    else: 
        raise ValueError(f"loss_type desconocido: {loss_type}")