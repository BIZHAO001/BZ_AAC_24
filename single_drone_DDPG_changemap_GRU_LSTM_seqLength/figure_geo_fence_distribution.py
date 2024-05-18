# -*- coding: utf-8 -*-
"""
@Time    : 5/13/2024 10:26 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
matplotlib.use('TkAgg')

with open(r'D:\MADDPG_2nd_jp\100524_14_45_23\toplot\geo_fence_number_perEps.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)


# Calculate the minimum and maximum
data_min = min(all_episode_situation)
data_max = max(all_episode_situation)

# Calculate the number of bins based on the range of data and the desired bin width
num_bins = 2

# Generate bins starting from the minimum value to cover the range
bins = np.linspace(data_min, data_max, num_bins+1)


plt.hist(all_episode_situation, bins=bins, edgecolor='black', align='left')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution of Data Points')
plt.show()
# print("done")
