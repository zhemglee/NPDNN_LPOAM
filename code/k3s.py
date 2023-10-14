# -*- coding: utf-8 -*-
# @Time : 2023/9/29 9:32
# @Author : zhemglee
# @FileName: plots.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

if not os.path.exists('./newpic/3s'):
    os.makedirs('./newpic/3s')


data = np.load('./output6_0.5/figure2_data.npz')
# data = np.load('./output_oam_0.5/figure2_data.npz')
array_names = data.files
print(array_names)

a = data[array_names[0]]
b = data[array_names[1]]
c = data[array_names[2]]
d = data[array_names[3]]


# d = np.concatenate([a[:15,:,:,:], b[:15,:,:,:], c[:15,:,:,:]], axis=0)
# print(d.shape)
# nrows = d.shape[0]//3
# ncols = 3
# fig, axes = plt.subplots(nrows,ncols)
# for ii in range(nrows):
#     for jj in range(ncols):
#         axes[ii,jj].imshow(d[jj * 6 + ii][0])  # cmap='gray'
#         axes[ii,jj].axis('off')
# column_titles = ['Original Image', 'Target Image', 'Output Image']
# for i in range(3):
#     axes[0,i].set_title(column_titles[i])
# plt.tight_layout()
# plt.show()


# i=1, col=8;      i=2, col=2;        i=3, col=1         # output6_0.5_0.5
# i=4, col=11;     i=5, col=7;        i=6, col=7          # output6_0.5_0.5
# i=7, col=1;      i=8, col=4;        i=9, col=10          # output_oam_0.5_0.5
i = 3
print(a.shape)
matrix1 = a[1][0]
matrix2 = b[1][0]
matrix3 = c[1][0]

plt.subplots(figsize=(3, 3))
plt.imshow(matrix1) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/3s/input_{}.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()

plt.subplots(figsize=(3, 3))
plt.imshow(matrix2) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/3s/target_{}.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()

plt.subplots(figsize=(3, 3))
plt.imshow(matrix3) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/3s/output_{}.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()




data.close()
