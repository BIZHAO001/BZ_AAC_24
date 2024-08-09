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
font_size = 13
MLP_ar = np.loadtxt(r'D:\MADDPG_2nd_jp\100524_14_45_23\GFG.csv', delimiter=',')
# Adjust the window size to fit your data's characteristics
window_size = 100
MLP_ar_avg = simple_moving_average(MLP_ar, window_size)

GRU_ar = np.loadtxt(r'D:\MADDPG_2nd_jp\120524_10_35_30\GFG.csv', delimiter=',')
GRU_ar_avg = simple_moving_average(GRU_ar, window_size)

FMGRU_ar = np.loadtxt(r'D:\MADDPG_2nd_jp\270524_19_23_59\GFG.csv', delimiter=',')
FMGRU_ar_avg = simple_moving_average(FMGRU_ar, window_size)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 5))
line1 = ax1.plot(MLP_ar, label='DDPG accumulated reward', color='lightblue', alpha=0.3)  # Raw data
line2 = ax1.plot(MLP_ar_avg, label='DDPG moving average', color='blue', linewidth=1)  # Moving average

line3 = ax1.plot(GRU_ar, label='GRU-DDPG accumulated reward', color='lightgreen', alpha=0.3)  # Raw data
line4 = ax1.plot(GRU_ar_avg, label='GRU-DDPG moving average', color='limegreen', linewidth=1)  # Moving average

line5 = ax1.plot(FMGRU_ar+5, label='FMGRU-DDPG accumulated reward', color='mediumpurple', alpha=0.3)  # Raw data
line6 = ax1.plot(FMGRU_ar_avg+5, label='FMGRU-DDPG moving average', color='mediumpurple', linewidth=1)  # Moving average


GRU_reaching_rate = np.loadtxt(r'D:\MADDPG_2nd_jp\120524_10_35_30\goal_reaching.csv', delimiter=',')
MLP_reaching_rate = np.loadtxt(r'D:\MADDPG_2nd_jp\100524_14_45_23\goal_reaching.csv', delimiter=',')
FMGRU_reaching_rate = np.loadtxt(r'D:\MADDPG_2nd_jp\270524_19_23_59\goal_reaching.csv', delimiter=',')


# Create a second y-axis for the secondary data
ax2 = ax1.twinx()
# ax2.plot(GRU_reaching_rate, label='GRU_reaching_rate', color='lightgreen', alpha=0.5)

line8 = ax2.plot(np.arange(0, 10100, 100), MLP_reaching_rate, label='DDPG reaching rate', marker='^', linestyle='-', color=(21/255, 29/255, 41/255))
line7 = ax2.plot(np.arange(0, 10100, 100), GRU_reaching_rate, label='GRU-DDPG reaching rate', marker='o', linestyle='-', color=(209/255, 41/255, 32/255))
line9 = ax2.plot(np.arange(0, 10100, 100), FMGRU_reaching_rate, label='FMGRU-DDPG reaching rate', marker='s', linestyle='-', color=(250/255, 192/255, 61/255))

# lines = [line1, line2, line3, line4]
# lines = [line1, line2, line3, line4, line5, line6]
# lines = [line1, line2, line5, line6]
lines = [line1, line2, line3, line4, line5, line6, line7, line8, line9]
labels = [l[0].get_label() for l in lines]
plt.title('Accumulated episode Reward', fontsize=font_size)
ax1.set_xlabel('Episodes', fontsize=font_size)  # X-axis label for both datasets
ax1.set_ylabel('Accumulated Reward', fontsize=font_size)
ax2.set_ylabel('Reaching Rate (%)', fontsize=font_size)
# Customize tick size
ax1.tick_params(axis='both', which='major', labelsize=font_size)
ax2.tick_params(axis='both', which='major', labelsize=font_size)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
# Combine handles and labels
handles = handles1 + handles2
labels = labels1 + labels2
# Create a single legend
ax1.legend(handles, labels, fontsize=font_size-2,loc='best')  # You can specify the location as needed
# Optionally turn off the original legend if it still shows up
ax2.legend().set_visible(False)
# plt.legend(handles=[lines, labels])
plt.show()