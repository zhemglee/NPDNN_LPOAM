# -*- coding: utf-8 -*-
# @Time : 2023/9/29 9:32
# @Author : zhemglee
# @FileName: plots.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if not os.path.exists('newpic'):
    os.makedirs('newpic')




x = np.linspace(-np.pi, np.pi, 100)

y0 = np.pi * np.tanh(x)
y1 = np.maximum(0, x)

fig, ax = plt.subplots()

plt.plot(x, y0, '-o', markersize=2.7,  linewidth=1)
plt.plot(x, y1, '-.v', markersize=2.7, linewidth=1)
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in')
font = FontProperties(family='Times New Roman', style='normal', weight='normal', size=12)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.xlabel('Input Phase', fontname='Times New Roman', fontsize=15)
plt.ylabel('Output Phase', fontname='Times New Roman', fontsize=15)
legend = plt.legend(['OTanh', 'ORelu'], fontsize=15, frameon=False)
for text in legend.get_texts():
    text.set_fontname('Times New Roman')
plt.savefig('./newpic/simgs2_1.png', dpi=1200)  # 指定分辨率为300 DPI
plt.show()
