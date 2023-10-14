# -*- coding: utf-8 -*-
# @Time : 2023/9/29 9:32
# @Author : zhemglee
# @FileName: plots.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn

import numpy as np
import os
import matplotlib.pyplot as plt

if not os.path.exists('./newpic/5s'):
    os.makedirs('./newpic/5s')


# data = np.load('./output6_2.0/figure2_data.npz')
# data = np.load('./output_oam_2.0/figure2_data.npz')
# data = np.load('./output6_0.0_0.3/figure2_data.npz')
data = np.load('./output_oam_0.0_0.3/figure2_data.npz')
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

#  0.0 0.0           0.0 0.5          0.5,0.0             0.5,0.5         0.0, 1.0        0.0, 2.0
# i=1, col=5;      i=2, col=7;       i=3, col=6;        i=4, col=13;     i=5, col=6;     i=6, col=1;       #LG
# is=1, col=5;     is=2, col=2;      is=3, col=0;       is=4, col=5;     is=5, col=1;    is=6, col=14;      #OAM

# 0.0 0.3
# i=7, col=8;       #LG
# is=7, col=14;      #OAM
i = 7
print(a.shape)
matrix1 = a[14][0]
matrix2 = b[14][0]
matrix3 = c[14][0]

plt.subplots(figsize=(3, 3))
plt.imshow(matrix1) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/5s/input_{}s.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()

plt.subplots(figsize=(3, 3))
plt.imshow(matrix2) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/5s/target_{}s.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()

plt.subplots(figsize=(3, 3))
plt.imshow(matrix3) # cmap='gray'
plt.axis('off')
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.tight_layout()
plt.savefig('./newpic/5s/output_{}s.png'.format(i), dpi=1200, bbox_inches="tight",pad_inches=0)
plt.show()


data.close()
