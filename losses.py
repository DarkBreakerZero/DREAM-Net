import torch
from torchvision import models
import os
import scipy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse
from datetime import datetime
from math import exp
from torch.autograd import Variable

class vgg_feature_extractor(nn.Module):

    # VGG(
    #   (features): Sequential(
    #     (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (1): ReLU(inplace=True)
    #     (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (3): ReLU(inplace=True)
    #     (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #     (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (6): ReLU(inplace=True)
    #     (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (8): ReLU(inplace=True)
    #     (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #     (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (11): ReLU(inplace=True)
    #     (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (13): ReLU(inplace=True)
    #     (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (15): ReLU(inplace=True)
    #     (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #     (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (18): ReLU(inplace=True)
    #     (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (20): ReLU(inplace=True)
    #     (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (22): ReLU(inplace=True)
    #     (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #     (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (25): ReLU(inplace=True)
    #     (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (27): ReLU(inplace=True)
    #     (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    #     (29): ReLU(inplace=True)
    #     (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    #   )
    #   (avgpool): AdaptiveAvgPool2d(output_size=(7, 7))
    #   (classifier): Sequential(
    #     (0): Linear(in_features=25088, out_features=4096, bias=True)
    #     (1): ReLU(inplace=True)
    #     (2): Dropout(p=0.5, inplace=False)
    #     (3): Linear(in_features=4096, out_features=4096, bias=True)
    #     (4): ReLU(inplace=True)
    #     (5): Dropout(p=0.5, inplace=False)
    #     (6): Linear(in_features=4096, out_features=1000, bias=True)
    #   )
    # )

    ''' NOTE: multi-gpu is not perfect at this time, model in cuda 0, vgg in cuda 1'''

    def __init__(self):
        super(vgg_feature_extractor, self).__init__()
        vgg_model = models.vgg16(pretrained=True)
        self.vgg_layers = nn.Sequential(*list(vgg_model.features.children())[:27])
        # self.vgg_layers.cuda()

    def forward(self, img):

        out = [None] * (len(self.vgg_layers)+1)

        if img.shape[1] != 3:

            img = img.repeat(1, 3, 1, 1)

        out[0] = img

        for i in range(len(self.vgg_layers)):

            out[i+1] = self.vgg_layers[i](out[i])

        # return out[3], out[8], out[13], out[20], out[27]
        return out[2], out[7], out[12], out[19], out[26]

def vgg_loss_calc(x, y, vgg_feature_extractor):

    xf1, xf2, xf3, xf4, xf5 = vgg_feature_extractor(x)
    yf1, yf2, yf3, yf4, yf5 = vgg_feature_extractor(y)

    # vgg_loss = F.mse_loss(xf1, yf1) + \
    #            F.mse_loss(xf2, yf2) + \
    #            F.mse_loss(xf3, yf3) + \
    #            F.mse_loss(xf4, yf4) + \
    #            F.mse_loss(xf5, yf5)

    vgg_loss = F.l1_loss(xf1, yf1) + \
               F.l1_loss(xf2, yf2) + \
               F.l1_loss(xf3, yf3) + \
               F.l1_loss(xf4, yf4) + \
               F.l1_loss(xf5, yf5)

    return vgg_loss

# class vgg_loss_calc(nn.Module):
#
#     ''' NOTE: not support multi-gpu '''
#
#     def __init__(self):
#         super(vgg_loss_calc, self).__init__()
#         vgg_model = models.vgg16(pretrained=True)
#         self.vgg_layers = nn.Sequential(*list(vgg_model.features.children())[:27])
#         self.vgg_layers.cuda()
#
#     def forward(self, input, target, loss_layers=[2, 7, 12, 19, 26]):
#
#         if input.shape[1] != 3:
#             input = input.repeat(1, 3, 1, 1)
#             target = target.repeat(1, 3, 1, 1)
#
#         x = input
#         y = target
#
#         loss = 0.0
#
#         for i, layer in enumerate(self.vgg_layers):
#             x = layer(x)
#             y = layer(y)
#             if i in loss_layers:
#                 # loss = loss + torch.nn.functional.mse_loss(x, y)
#                 loss = loss + torch.nn.functional.l1_loss(x, y)
#         return loss

def tv_loss(x, beta = 0.5, reg_coeff = 5):
    '''Calculates TV loss for an image `x`.

    Args:
        x: image, torch.Variable of torch.Tensor
        beta: See https://arxiv.org/abs/1412.0035 (fig. 2) to see effect of `beta`
    '''
    dh = torch.pow(x[:,:,:,1:] - x[:,:,:,:-1], 2)
    dw = torch.pow(x[:,:,1:,:] - x[:,:,:-1,:], 2)
    a,b,c,d=x.shape
    return reg_coeff*(torch.sum(torch.pow(dh[:, :, :-1] + dw[:, :, :, :-1], beta))/(a*b*c*d))

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps / 0.02 * 1024

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):

    min_val = torch.min(torch.cat([img1, img2], 1))
    max_val = torch.max(torch.cat([img1, img2], 1))

    img1 = (img1 - min_val) / (max_val - min_val)
    img2 = (img2 - min_val) / (max_val - min_val)

    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)
