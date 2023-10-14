# -*- coding: utf-8 -*-
# @Time : 2023/9/29 19:32
# @Author : zhemglee
# @FileName: plots.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn

import torch
import piq
from skimage.io import imread

i = 6

x1 = torch.tensor(imread('./newpic/5s/input_{}s.png'.format(i))).permute(2, 0, 1)[None, ...] / 255.
y1 = torch.tensor(imread('./newpic/5s/target_{}s.png'.format(i))).permute(2, 0, 1)[None, ...] / 255.
x2 = torch.tensor(imread('./newpic/5s/output_{}s.png'.format(i))).permute(2, 0, 1)[None, ...] / 255.
y2 = torch.tensor(imread('./newpic/5s/target_{}s.png'.format(i))).permute(2, 0, 1)[None, ...] / 255.

# input & tar
X = x1
Y = y1
psnr_values = piq.psnr(X, Y, data_range=1.0)
ssim_values = piq.ssim(X, Y, data_range=1.0)
# lpips_criterion = piq.LPIPS(reduction='none')
# lpips_values = lpips_criterion(X, Y)
# print(lpips_values)
train_log = f'{"Input_{} & Target_{}".format(i,i)}:  psnr: {"{:.4f}".format(psnr_values.item())}, ssim: {"{:.4f}".format(ssim_values)}'
print(train_log,'\n')

# output & tar
X = x2
Y = y2
psnr_values = piq.psnr(X, Y, data_range=1.0)
ssim_values = piq.ssim(X, Y, data_range=1.0)
# lpips_criterion = piq.LPIPS(reduction='none')
# lpips_values = lpips_criterion
train_log = f'{"Output_{} & Target_{}".format(i,i)}:  psnr: {"{:.4f}".format(psnr_values.item())}, ssim: {"{:.4f}".format(ssim_values)}'
print(train_log,'\n')

