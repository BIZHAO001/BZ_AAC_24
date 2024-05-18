# -*- coding: utf-8 -*-
"""
@Time    : 5/14/2024 11:25 AM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')

def simple_moving_average(data, window_size):
    """Calculate the simple moving average with edge handling."""
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(data, window, 'same')  # 'same' returns the convolution at each point of the original array

# Convert to pandas DataFrame

MLP_ar = np.loadtxt(r'D:\MADDPG_2nd_jp\100524_14_45_23\GFG.csv', delimiter=',')
# Adjust the window size to fit your data's characteristics
window_size = 100
MLP_ar_avg = simple_moving_average(MLP_ar, window_size)

GRU_ar = np.loadtxt(r'D:\MADDPG_2nd_jp\120524_10_35_30\GFG.csv', delimiter=',')
GRU_ar_avg = simple_moving_average(GRU_ar, window_size)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))
line1 = ax1.plot(MLP_ar, label='DDPG accumulated reward', color='lightblue', alpha=0.5)  # Raw data
line2 = ax1.plot(MLP_ar_avg, label='DDPG moving average', color='blue', linewidth=2)  # Moving average
line3 = ax1.plot(GRU_ar, label='GRU-DDPG accumulated reward', color='lightgreen', alpha=0.5)  # Raw data
line4 = ax1.plot(GRU_ar_avg, label='GRU-DDPG moving average', color='green', linewidth=2)  # Moving average

GRU_reaching_rate = np.loadtxt(r'D:\MADDPG_2nd_jp\120524_10_35_30\goal_reaching.csv', delimiter=',')
MLP_reaching_rate = np.loadtxt(r'D:\MADDPG_2nd_jp\100524_14_45_23\goal_reaching.csv', delimiter=',')


# Create a second y-axis for the secondary data
ax2 = ax1.twinx()
# ax2.plot(GRU_reaching_rate, label='GRU_reaching_rate', color='lightgreen', alpha=0.5)
line5 = ax2.plot(np.arange(0, 10100, 100), GRU_reaching_rate, label='GRU-DDPG_reaching_rate', marker='o', linestyle='-', color='red')
line6 = ax2.plot(np.arange(0, 10100, 100), MLP_reaching_rate, label='DDPG_reaching_rate', marker='^', linestyle='-', color='black')

lines = [line1, line2, line3, line4, line5, line6]
labels = [l[0].get_label() for l in lines]
plt.title('Accumulated episode Reward')
ax1.set_xlabel('Episodes')  # X-axis label for both datasets
ax1.set_ylabel('Accumulated Reward')
ax2.set_ylabel('Reaching Rate (%)')
# plt.legend()
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, loc='best')  # You can specify the location as needed
# Optionally turn off the original legend if it still shows up
ax2.legend().set_visible(False)
# plt.legend(handles=[lines, labels])
plt.show()