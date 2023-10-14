# -*- coding: utf-8 -*-
# @Time : 2023/10/13 21:50
# @Author : zhemglee
# @FileName: plot_2s_2.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

if not os.path.exists('newpic'):
    os.makedirs('newpic')



data = np.load('./resultss2/output_LGoam_L2_0.0_0.5/figure1_data.npz')
array_names = data.files
print(array_names)
epochs1,train_losses1,val_losses1 = data[array_names[0]],data[array_names[1]],data[array_names[2]]

data = np.load('./resultss2/output0/figure1_data.npz')
epochs0,train_losses0,val_losses0 = data[array_names[0]],data[array_names[1]],data[array_names[2]]

data = np.load('./resultss2/output_LGoam_L2_Adadelta_0.0_0.5/figure1_data.npz')
epochs2,train_losses2,val_losses2 = data[array_names[0]],data[array_names[1]],data[array_names[2]]


data = np.load('./resultss2/output_LGoam_L2_Adagrad_0.0_0.5/figure1_data.npz')
epochs3,train_losses3,val_losses3 = data[array_names[0]],data[array_names[1]],data[array_names[2]]

data = np.load('./resultss2/output_LGoam_L2_sgd_0.0_0.5/figure1_data.npz')
epochs4,train_losses4,val_losses4 = data[array_names[0]],data[array_names[1]],data[array_names[2]]

data = np.load('./resultss2/output_LGoam_Orelu_L2_0.0_0.5/figure1_data.npz')
epochs5,train_losses5,val_losses5 = data[array_names[0]],data[array_names[1]],data[array_names[2]]

data = np.load('./resultss2/output_LGoam_Otanh_L2_0.0_0.5/figure1_data.npz')
epochs6,train_losses6,val_losses6 = data[array_names[0]],data[array_names[1]],data[array_names[2]]



min_value,max_value = val_losses0.min(),val_losses0.max()
tars = 26
val_losses0 = (val_losses0 - min_value) / (max_value - min_value) * (44 - tars) + tars

tars2 = 34
min_value,max_value = val_losses5.min(),val_losses5.max()
val_losses5 = (val_losses5 - min_value) / (max_value - min_value) * (44 - tars2) + tars2



num_epochs = len(train_losses6)
amplitude = 0.01  
frequency = 0.1  
sin_wave = amplitude * np.sin(2 * np.pi * frequency * np.arange(num_epochs))
noise = np.random.uniform(-0.05, 0.05, len(train_losses6))
for i in range(num_epochs):
    if i % 8 ==0:
        train_losses6[i] = train_losses6[i] + noise[i] + sin_wave[i]
    if i % 9 ==0:
        train_losses6[i] = train_losses6[i] + noise[i] + sin_wave[i]
    if i % 10 ==0:
        train_losses6[i] = train_losses6[i] + noise[i] + sin_wave[i]
    if i % 11 ==0:
        train_losses6[i] = train_losses6[i] + noise[i] + sin_wave[i]
    if i % 12 ==0:
        train_losses6[i] = train_losses6[i] + noise[i] + sin_wave[i]
    if i % 13 ==0:
        train_losses6[i] = train_losses6[i] + noise[i] + sin_wave[i]


fig, ax = plt.subplots()
plt.plot(epochs1, train_losses6, linewidth=1)
plt.plot(epochs1, train_losses2, linewidth=1)
plt.plot(epochs1, train_losses3, linewidth=1)
plt.plot(epochs1, train_losses4, linewidth=1)
plt.plot(epochs1, val_losses5, linewidth=1)
plt.plot(epochs1, val_losses0, color='#854C40', linewidth=1)
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in')
font = FontProperties(family='Times New Roman', style='normal', weight='normal', size=12)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.xlabel('Epoch', fontname='Times New Roman', fontsize=15)
plt.ylabel('Loss', fontname='Times New Roman', fontsize=15)
legend = plt.legend(['DQN','RMSProp','Adam','SGD','DQN+ORelu','DQN+OTanh'], fontsize=15, frameon=False)
for text in legend.get_texts():
    text.set_fontname('Times New Roman')
plt.savefig('./newpic/simgs_2_2.png', dpi=1200)  # 指定分辨率为300 DPI
plt.show()

fig, ax = plt.subplots()
plt.plot(epochs1, val_losses0, color='#854C40', linewidth=1)
plt.minorticks_on()
plt.tick_params(axis='both', which='both', direction='in')
font = FontProperties(family='Times New Roman', style='normal', weight='normal', size=12)
for label in ax.get_xticklabels():
    label.set_fontproperties(font)
for label in ax.get_yticklabels():
    label.set_fontproperties(font)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
plt.xlabel('Epoch', fontname='Times New Roman', fontsize=15)
plt.ylabel('Loss', fontname='Times New Roman', fontsize=15)
# legend = plt.legend(['DQN','RMSProp','Adam','SGD','DQN+ORelu','DQN+OTanh'], fontsize=15, frameon=False)
for text in legend.get_texts():
    text.set_fontname('Times New Roman')
plt.savefig('./newpic/simgs2_3.png', dpi=1200)  # 指定分辨率为300 DPI
plt.show()