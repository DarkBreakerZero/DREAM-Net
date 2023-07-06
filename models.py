# -*- coding: utf-8 -*-
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import torch
import math
import torch.nn.functional as F

class ConvBnRelu2d(nn.Module):
    def __init__(self, in_chl, out_chl, kernel_size=3, dilation=1, stride=1, groups=1, is_bn=False, is_relu=True):
        super(ConvBnRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_chl, out_chl, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=stride,
                              dilation=dilation, groups=groups, bias=True)
        self.bn = None
        self.relu = None

        if is_bn is True:
            self.bn = nn.BatchNorm2d(out_chl, eps=1e-4)
        if is_relu is True:
            # self.relu = nn.ReLU(inplace=True)
            self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(ResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            )

        self.conv_x = None

        if in_chl != out_chl:

            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv_out = self.encode(x)

        if self.conv_x is None:

            res_out = F.leaky_relu(conv_out + x)

        else: 

            res_out = F.leaky_relu(conv_out + self.conv_x(x))

        return res_out


class EncodeResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(EncodeResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            )

        self.conv_x = None

        if in_chl != out_chl:

            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x):

        conv_out = self.encode(x)

        if self.conv_x is None:

            res_out = F.leaky_relu(conv_out + x)

        else: 

            res_out = F.leaky_relu(conv_out + self.conv_x(x))

        down_out = F.max_pool2d(res_out, kernel_size=2, stride=2, padding=0, ceil_mode=True)

        return res_out, down_out

class DecodeResBlock2D(nn.Module):

    def __init__(self, in_chl, out_chl, kernel_size=3):
        super(DecodeResBlock2D, self).__init__()

        self.encode = nn.Sequential(
            ConvBnRelu2d(in_chl+out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            ConvBnRelu2d(out_chl, out_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1),
            )

        self.conv_x = None

        if in_chl != out_chl:

            self.conv_x = ConvBnRelu2d(in_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_bn=False, is_relu=False)

    def forward(self, x, skip_x):

        _, _, H, W = skip_x.size()
        
        up_out = F.upsample(x, size=(H, W), mode='bilinear')
        
        conv_out = self.encode(torch.cat([up_out, skip_x], 1))

        if self.conv_x is None:

            res_out = F.leaky_relu(conv_out + up_out)

        else: 

            res_out = F.leaky_relu(conv_out + self.conv_x(up_out))


        return res_out

class DREAM_UNetL3(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(DREAM_UNetL3, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.encoder1 = EncodeResBlock2D(model_chl, model_chl)
        self.encoder2 = EncodeResBlock2D(model_chl, model_chl*2)
        self.encoder3 = EncodeResBlock2D(model_chl*2, model_chl*4)

        self.conv_center = ResBlock2D(model_chl*4, model_chl*8)

        self.decoder3 = DecodeResBlock2D(model_chl*8, model_chl*4)
        self.decoder2 = DecodeResBlock2D(model_chl*4, model_chl*2)
        self.decoder1 = DecodeResBlock2D(model_chl*2, model_chl)

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):

        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)
        res_x2, down_x2 = self.encoder2(down_x1)
        res_x3, down_x3 = self.encoder3(down_x2)

        x_center = self.conv_center(down_x3)

        out_x3 = self.decoder3(x_center, res_x3)
        out_x2 = self.decoder2(out_x3, res_x2)
        out_x1 = self.decoder1(out_x2, res_x1)

        # out = F.leaky_relu(self.conv_end(out_x1) + x)
        out = F.leaky_relu(self.conv_end(out_x1) + torch.unsqueeze(x[:, -1, :, :], 1))

        return out

class DREAM_UNetL2(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(DREAM_UNetL2, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.encoder1 = EncodeResBlock2D(model_chl, model_chl)
        self.encoder2 = EncodeResBlock2D(model_chl, model_chl*2)

        self.conv_center = ResBlock2D(model_chl*2, model_chl*4)

        self.decoder2 = DecodeResBlock2D(model_chl*4, model_chl*2)
        self.decoder1 = DecodeResBlock2D(model_chl*2, model_chl)

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):

        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)
        res_x2, down_x2 = self.encoder2(down_x1)

        x_center = self.conv_center(down_x2)

        out_x2 = self.decoder2(x_center, res_x2)
        out_x1 = self.decoder1(out_x2, res_x1)

        out = F.leaky_relu(self.conv_end(out_x1) + torch.unsqueeze(x[:, -1, :, :], 1))

        return out


class DREAM_UNetL1(nn.Module):
    def __init__(self, in_chl=1, out_chl=1, kernel_size=3, model_chl=32):
        super(DREAM_UNetL1, self).__init__()
        self.out_chl = out_chl
        self.model_chl = model_chl
        self.in_chl = in_chl

        self.conv_start = ConvBnRelu2d(in_chl, model_chl, kernel_size=kernel_size, dilation=1, stride=1, groups=1)
        self.encoder1 = EncodeResBlock2D(model_chl, model_chl)

        self.conv_center = ResBlock2D(model_chl*1, model_chl*2)

        self.decoder1 = DecodeResBlock2D(model_chl*2, model_chl)

        self.conv_end = ConvBnRelu2d(model_chl, out_chl, kernel_size=1, dilation=1, stride=1, groups=1, is_relu=False)

    def forward(self, x):

        x_start = self.conv_start(x)

        res_x1, down_x1 = self.encoder1(x_start)

        x_center = self.conv_center(down_x1)

        out_x1 = self.decoder1(x_center, res_x1)

        out = F.leaky_relu(self.conv_end(out_x1) + x)

        return out

class DreamNetDense(nn.Module):

    def __init__(self, recon_op, iter_block=3, net_chl=32):
        super(DreamNetDense, self).__init__()
        self.iter_block = iter_block
        self.net_chl = net_chl
        self.recon_op = recon_op

        self.net = nn.ModuleList()

        self.net = self.net.append(DREAM_UNetL3(in_chl=1, out_chl=1, kernel_size=3, model_chl=self.net_chl))

        for i in range(self.iter_block):
            self.net = self.net.append(DREAM_UNetL3(in_chl=1, out_chl=1, kernel_size=3, model_chl=self.net_chl))
            self.net = self.net.append(DREAM_UNetL1(in_chl=1, out_chl=1, kernel_size=3, model_chl=self.net_chl))
            self.net = self.net.append(DREAM_UNetL2(in_chl=i+2, out_chl=1, kernel_size=3, model_chl=self.net_chl))

    def forward(self, proj, ldct, mask):

        img_current = self.net[0](ldct)

        img_dense = img_current

        for i in range(self.iter_block):
            proj_current = self.recon_op.forward(img_current / 1024)

            proj_net = self.net[3 * i + 1](proj_current)
            proj_wrt = proj_net * (1 - mask) + proj * mask

            img_error = self.recon_op.backprojection(self.recon_op.filter_sinogram(proj_current - proj_wrt)) * 1024
            img_error_net = self.net[3 * i + 2](img_error)

            img_current = self.net[3 * i + 3](torch.cat([img_dense, img_current + img_error_net], 1))

            img_dense = torch.cat([img_dense, img_current], 1)

        return proj_net, img_current