# -*- coding: utf-8 -*-
# @Time : 2023/9/29 9:32
# @Author : zhemglee
# @FileName: plots.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn

import numpy as np
import os
import matplotlib.pyplot as plt

if not os.path.exists('./newpic/2s'):
    os.makedirs('./newpic/2s')


data = np.load('./output6_epoch_3/figure2_data.npz')
data = np.load('./output6_epoch_20/figure2_data.npz')
# data = np.load('./output6_epoch_50/figure2_data.npz')
# data = np.load('./output6_epoch_1250/figure2_data.npz')
array_names = data.files
print(array_names)

a = data[array_names[0]]
b = data[array_names[1]]
c = data[array_names[2]]
d = data[array_names[3]]


d = np.concatenate([a[:15,:,:,:], b[:15,:,:,:], c[:15,:,:,:]], axis=0)
print(d.shape)
nrows = d.shape[0]//3
ncols = 3
fig, axes = plt.subplots(nrows,ncols)
for ii in range(nrows):
    for jj in range(ncols):
        axes[ii,jj].imshow(d[jj * 6 + ii][0])  # cmap='gray'
        axes[ii,jj].axis('off')
column_titles = ['Original Image', 'Target Image', 'Output Image']
for i in range(3):
    axes[0,i].set_title(column_titles[i])
plt.tight_layout()
plt.show()

#  epoch 3         epoch 20       epoch 50          epoch 1250
# i=1, col=14;     i=2, col=4;    i=3, col=2;       i=4, col=22;
i = 2
print(a.shape)
matrix1 = a[4][0]
matrix2 = b[4][0]
matrix3 = c[4][0]

plt.subplots(figsize=(3, 3))
plt.imshow(matrix1) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/2s/input_{}s.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()

plt.subplots(figsize=(3, 3))
plt.imshow(matrix2) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/2s/target_{}s.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()

plt.subplots(figsize=(3, 3))
plt.imshow(matrix3) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/2s/output_{}s.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()


data.close()
