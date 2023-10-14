# -*- coding: utf-8 -*-
# @Time : 2023/10/13 20:13
# @Author : zhemglee
# @FileName: kys.py
# @Software: PyCharm
# @Email ：zhemglee@tju.edu.cn
import pandas as pd
import numpy as np


def load_data(filepath):
    data_df = pd.read_csv(filepath)
    data_df = data_df[~data_df.apply(lambda row: row.str.isalpha().any(), axis=1)] # 删除包含字母的行
    selected_columns = data_df.iloc[[2,3], [4, 5, 6]] # 选择数据
    selected_columns = selected_columns.applymap(lambda x: x.split(":")[1]) # 删除字母
    return selected_columns


if __name__=='__main__':
    filepath1 = './output6_2.0/newpic/output_index_record.txt'
    filepath2 = './output_oam_2.0/newpic/output_index_record.txt'
    data1 = load_data(filepath1)
    data1 = np.array(data1.values)
    data1 = np.array([[float(val.strip()) for val in row] for row in data1])
    data2 = load_data(filepath2)
    data2 = np.array(data2.values)
    data2 = np.array([[float(val.strip()) for val in row] for row in data2])
    data = 2/3*data1 + 1/3*data2
    print(data)