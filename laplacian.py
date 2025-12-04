import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch

def gauss_kernel(size=5, channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current-up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target, occu_mask_backward):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        occu_mask_backward_list = []
        for i in range(len(pyr_input)):
            occu_mask_backward_list.append(F.interpolate(occu_mask_backward, size=(pyr_input[i].shape[2], pyr_input[i].shape[3]), mode='bilinear', align_corners=True))
        # print("shape of pyr_input: ", pyr_input[0].shape)
        # print("shape of pyr_input: ", pyr_input[1].shape)
        # # len of pyr_input 
        # print("len of pyr_input: ", len(pyr_input))
        return sum(torch.nn.functional.l1_loss(c*a, c*b) for i, (a, b, c) in enumerate(zip(pyr_input, pyr_target, occu_mask_backward_list)))/len(pyr_input)

class LapLossConf(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLossConf, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)
        
    def forward(self, input, target, conf, log_conf, occu_mask_backward):
        pyr_input  = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        # reshape conf Bx1xHxW and log_conf Bx1xHxW to the same H and W as pyr_input
        conf_list = []
        log_conf_list = []
        for i in range(len(pyr_input)):
            # reshape height and width of conf and log_conf to the same as pyr_input
            reshaped_conf = F.interpolate(conf, size=(pyr_input[i].shape[2], pyr_input[i].shape[3]), mode='bilinear', align_corners=True)
            reshaped_log_conf = F.interpolate(log_conf, size=(pyr_input[i].shape[2], pyr_input[i].shape[3]), mode='bilinear', align_corners=True)
            conf_list.append(reshaped_conf)
            log_conf_list.append(reshaped_log_conf)
        
        # multiply pyr_input with conf and log_conf
        pyr_input_conf = [c*torch.abs(a-b)-0.2*d for a, b, c, d in zip(pyr_input, pyr_target, conf_list, log_conf_list)]
        pyr_conf_losses = []
        for i in range(len(pyr_input_conf)):
            avg_conf_loss = (pyr_input_conf[i].mean(1, True))
            pyr_conf_losses.append(avg_conf_loss.sum()/occu_mask_backward.sum())
            
        
        return sum(pyr_conf_losses)/len(pyr_conf_losses)