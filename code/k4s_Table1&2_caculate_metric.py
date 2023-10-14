# -*- coding: utf-8 -*-
# @Time : 2023/9/29 19:32
# @Author : zhemglee
# @FileName: plots.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn
import torch
import torch.nn.functional as F
import numpy as np
import piq
import os
import sys
import logging


if not os.path.exists('newpic'):
    os.makedirs('newpic')

logging.basicConfig(filename='./newpic/output_index_record.txt', level=logging.INFO, format='%(message)s')

data = np.load('./output_oam_0.0/figure2_data.npz')
array_names = data.files
print(array_names)
logging.info(f'data lable: ,{array_names}')
# Train
a = data[array_names[0]]
b = data[array_names[1]]
c = data[array_names[2]]

print('Train_Image_Shape:',a.shape)
logging.info(f'Train_Image_Shape: ,{a.shape}')
logging.info('-----------------------------Train-----------------------------')
print('-----------------------------Train-----------------------------')
# ori & tar
X = torch.tensor(a)
Y = torch.tensor(b)
X = (X - X.min()) / (X.max() - X.min())  # 【20, 1, 256, 256】
Y = (Y - Y.min()) / (Y.max() - Y.min())
npccl1 = torch.mean(torch.sqrt(torch.sum(torch.abs(X - Y), dim=(2, 3), keepdim=True))) / 50
npccl2 = torch.mean(torch.sqrt(torch.sum((X - Y) ** 2, dim=(2, 3), keepdim=True))) / 30
ssim = 1 - piq.ssim(X, Y, data_range=1.0)
cosine_similarity = F.cosine_similarity(X.view(X.size(0), -1), Y.view(Y.size(0), -1), dim=1).mean()
cosine = 1 - (cosine_similarity + 1) / 2
harrpsi = 1 - piq.haarpsi(X, Y, data_range=1.0)

psnr_values = piq.psnr(X, Y, data_range=1.0)
ssim_values = piq.ssim(X, Y, data_range=1.0)
lpips_criterion = piq.LPIPS(reduction='none')
lpips_values = torch.mean(lpips_criterion(X, Y,))
# print('npccl1:',"{:.4f}".format(npccl1.item()),'npccl2:',"{:.4f}".format(npccl2.item()),\
#       'cosine_similarity:',"{:.4f}".format(cosine_similarity.item()),'harrpsi:',"{:.4f}".format((1-harrpsi).item()),\
#       'psnr:',"{:.4f}".format(psnr_values.item()),'ssim:',"{:.4f}".format(ssim_values),\
#       'lpips:',"{:.4f}".format(lpips_values.item()))
train_log = f'npccl1: {"{:.4f}".format(npccl1.item())}, npccl2: {"{:.4f}".format(npccl2.item())}, cosine_similarity: {"{:.4f}".format(cosine_similarity.item())}, harrpsi: {"{:.4f}".format((1-harrpsi).item())}, psnr: {"{:.4f}".format(psnr_values.item())}, ssim: {"{:.4f}".format(ssim_values)}, lpips: {"{:.4f}".format(lpips_values.item())}'
logging.info(train_log)
print(train_log,'\n')
# output & tar
X = torch.tensor(c)
Y = torch.tensor(b)
X = (X - X.min()) / (X.max() - X.min())  # 【20, 1, 256, 256】
Y = (Y - Y.min()) / (Y.max() - Y.min())
npccl1 = torch.mean(torch.sqrt(torch.sum(torch.abs(X - Y), dim=(2, 3), keepdim=True))) / 50
npccl2 = torch.mean(torch.sqrt(torch.sum((X - Y) ** 2, dim=(2, 3), keepdim=True))) / 30
ssim = 1 - piq.ssim(X, Y, data_range=1.0)
cosine_similarity = F.cosine_similarity(X.view(X.size(0), -1), Y.view(Y.size(0), -1), dim=1).mean()
cosine = 1 - (cosine_similarity + 1) / 2
harrpsi = 1 - piq.haarpsi(X, Y, data_range=1.0)

psnr_values = piq.psnr(X, Y, data_range=1.0)
ssim_values = piq.ssim(X, Y, data_range=1.0)
lpips_criterion = piq.LPIPS(reduction='none')
lpips_values = torch.mean(lpips_criterion(X, Y,))
train_log = f'npccl1: {"{:.4f}".format(npccl1.item())}, npccl2: {"{:.4f}".format(npccl2.item())}, cosine_similarity: {"{:.4f}".format(cosine_similarity.item())}, harrpsi: {"{:.4f}".format((1-harrpsi).item())}, psnr: {"{:.4f}".format(psnr_values.item())}, ssim: {"{:.4f}".format(ssim_values)}, lpips: {"{:.4f}".format(lpips_values.item())}'
logging.info(train_log)
print(train_log,'\n')



# Val
a00 = data[array_names[4]]
b00 = data[array_names[5]]
c00 = data[array_names[6]]

print('Val_Image_Shape:',a00.shape)
logging.info(f'Val_Image_Shape: ,{a00.shape}')
logging.info('-----------------------------Val-----------------------------')
print('-----------------------------Val-----------------------------')
# ori & tar
X = torch.tensor(a00)
Y = torch.tensor(b00)
X = (X - X.min()) / (X.max() - X.min())  # 【20, 1, 256, 256】
Y = (Y - Y.min()) / (Y.max() - Y.min())
npccl1 = torch.mean(torch.sqrt(torch.sum(torch.abs(X - Y), dim=(2, 3), keepdim=True))) / 50
npccl2 = torch.mean(torch.sqrt(torch.sum((X - Y) ** 2, dim=(2, 3), keepdim=True))) / 30
ssim = 1 - piq.ssim(X, Y, data_range=1.0)
cosine_similarity = F.cosine_similarity(X.view(X.size(0), -1), Y.view(Y.size(0), -1), dim=1).mean()
cosine = 1 - (cosine_similarity + 1) / 2
harrpsi = 1 - piq.haarpsi(X, Y, data_range=1.0)

psnr_values = piq.psnr(X, Y, data_range=1.0)
ssim_values = piq.ssim(X, Y, data_range=1.0)
lpips_criterion = piq.LPIPS(reduction='none')
lpips_values = torch.mean(lpips_criterion(X, Y,))
train_log = f'npccl1: {"{:.4f}".format(npccl1.item())}, npccl2: {"{:.4f}".format(npccl2.item())}, cosine_similarity: {"{:.4f}".format(cosine_similarity.item())}, harrpsi: {"{:.4f}".format((1-harrpsi).item())}, psnr: {"{:.4f}".format(psnr_values.item())}, ssim: {"{:.4f}".format(ssim_values)}, lpips: {"{:.4f}".format(lpips_values.item())}'
logging.info(train_log)
print(train_log,'\n')

# output & tar
X = torch.tensor(c00)
Y = torch.tensor(b00)
X = (X - X.min()) / (X.max() - X.min())  # 【20, 1, 256, 256】
Y = (Y - Y.min()) / (Y.max() - Y.min())
npccl1 = torch.mean(torch.sqrt(torch.sum(torch.abs(X - Y), dim=(2, 3), keepdim=True))) / 50
npccl2 = torch.mean(torch.sqrt(torch.sum((X - Y) ** 2, dim=(2, 3), keepdim=True))) / 30
ssim = 1 - piq.ssim(X, Y, data_range=1.0)
cosine_similarity = F.cosine_similarity(X.view(X.size(0), -1), Y.view(Y.size(0), -1), dim=1).mean()
cosine = 1 - (cosine_similarity + 1) / 2
harrpsi = 1 - piq.haarpsi(X, Y, data_range=1.0)

psnr_values = piq.psnr(X, Y, data_range=1.0)
ssim_values = piq.ssim(X, Y, data_range=1.0)
lpips_criterion = piq.LPIPS(reduction='none')
lpips_values = torch.mean(lpips_criterion(X, Y,))
train_log = f'npccl1: {"{:.4f}".format(npccl1.item())}, npccl2: {"{:.4f}".format(npccl2.item())}, cosine_similarity: {"{:.4f}".format(cosine_similarity.item())}, harrpsi: {"{:.4f}".format((1-harrpsi).item())}, psnr: {"{:.4f}".format(psnr_values.item())}, ssim: {"{:.4f}".format(ssim_values)}, lpips: {"{:.4f}".format(lpips_values.item())}'
logging.info(train_log)
print(train_log,'\n')

