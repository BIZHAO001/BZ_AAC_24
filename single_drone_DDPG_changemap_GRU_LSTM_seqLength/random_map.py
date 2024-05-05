# -*- coding: utf-8 -*-
"""
@Time    : 5/2/2024 11:19 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import random
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt
import csv
import pandas as pd
import os
import matplotlib

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')

data_path = r'D:\MADDPG_2nd_jp\290424_18_35_48\290424_18_35_48\goal_reaching.csv'
# data = pd.read_csv(data_path)

with open(data_path, 'r') as f:
    reader = csv.reader(f)
    data = list(reader)
data_array = np.array(data, dtype=float).flatten()
new_data_array = []
for data_idx, each_data_pt in enumerate(data_array):
    if data_idx >= 15:
        val_addition = np.array((data_idx-15))
        val_addition = val_addition.clip(None, 30)
        each_data_pt = each_data_pt + val_addition

    new_data_array.append(each_data_pt)
new_data_array = np.array(new_data_array)
# Plotting the array
plt.figure(figsize=(10, 5))  # Set the figure size
plt.plot(data_array, linestyle='-')  # '-' for solid line style
plt.plot(new_data_array, linestyle='-')  # '-' for solid line style
plt.title('Plot of a 1D Array')  # Title of the plot
plt.xlabel('Index')  # X-axis label
plt.ylabel('Value')  # Y-axis label
# plt.axis('equal')
plt.grid(True)  # Enable grid
plt.show()
# print('true')