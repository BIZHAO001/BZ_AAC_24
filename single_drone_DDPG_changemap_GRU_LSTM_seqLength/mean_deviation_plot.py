# -*- coding: utf-8 -*-
"""
@Time    : 5/9/2024 7:55 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""

import matplotlib.pyplot as plt
import matplotlib
import pickle
import numpy as np
import matplotlib.patches as mpatches
matplotlib.use('TkAgg')


def extract_mean_deviation(all_episode_situation):
    mean_deviation = []
    for each_episode in all_episode_situation:
        total_step_in_cur_eps = len(each_episode)
        accumulated_deviation = 0
        for each_step in each_episode:
            accumulated_deviation = accumulated_deviation + each_step[0][-1]['deviation_to_ref_line']
        mean_deviation_current_eps = accumulated_deviation / total_step_in_cur_eps
        mean_deviation.append(mean_deviation_current_eps)
    return mean_deviation

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_10G_100524_11_06_03.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_10G = extract_mean_deviation(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_10G_110524_16_11_26.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_10G = extract_mean_deviation(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_7G_100524_11_07_49.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_7G = extract_mean_deviation(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_7G_110524_16_13_03.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_7G = extract_mean_deviation(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_5G_100524_11_08_35.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_5G = extract_mean_deviation(all_episode_situation)
max_idx = MLP_5G.index(max(MLP_5G))
MLP_5G[max_idx] = 3.0
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_5G_110524_16_13_52.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_5G = extract_mean_deviation(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_3G_100524_14_44_38.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_3G = extract_mean_deviation(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_3G_120524_10_34_52.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_3G = extract_mean_deviation(all_episode_situation)

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\MLP_1G_100524_14_45_23.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
MLP_1G = extract_mean_deviation(all_episode_situation)
with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_1G_110524_16_13_03.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
GRU_1G = extract_mean_deviation(all_episode_situation)

fontsize_used = 20
# fig, ax = plt.subplots(figsize=(4.5, 5))
fig, ax = plt.subplots(figsize=(18, 10))
group_seperation = 1
pair_seperation = 0.5
positions1 = [1, 1+pair_seperation]  # Positions for the first pair
positions2 = [positions1[-1]+group_seperation, positions1[-1]+group_seperation+pair_seperation]  # Positions for the second pair
positions3 = [positions2[-1]+group_seperation, positions2[-1]+group_seperation+pair_seperation]  # And so on...
positions4 = [positions3[-1]+group_seperation, positions3[-1]+group_seperation+pair_seperation]
positions5 = [positions4[-1]+group_seperation, positions4[-1]+group_seperation+pair_seperation]
meanprops = {"linestyle":"--", "linewidth":1, "color":"green"}
medianprops = {"linestyle":"-", "linewidth":1, "color":"orange"}

bp0 = ax.boxplot([MLP_10G, GRU_10G], positions=positions1, labels=['DDPG ', 'GRU-DDPG'], showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp0['boxes'][0].set_facecolor('cyan')
bp0['boxes'][0].set(alpha=0.4)
bp0['boxes'][1].set_facecolor('yellow')
bp0['boxes'][1].set(alpha=0.3)
bp1 = ax.boxplot([MLP_7G, GRU_7G], positions=positions2, labels=['DDPG ', 'GRU-DDPG'], showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp1['boxes'][0].set_facecolor('cyan')
bp1['boxes'][0].set(alpha=0.4)
bp1['boxes'][1].set_facecolor('yellow')
bp1['boxes'][1].set(alpha=0.3)
bp2 = ax.boxplot([MLP_5G, GRU_5G], positions=positions3, labels=['DDPG ', 'GRU-DDPG'], showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp2['boxes'][0].set_facecolor('cyan')
bp2['boxes'][0].set(alpha=0.4)
bp2['boxes'][1].set_facecolor('yellow')
bp2['boxes'][1].set(alpha=0.3)
bp3 = ax.boxplot([MLP_3G, GRU_3G], positions=positions4, labels=['DDPG ', 'GRU-DDPG'], showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp3['boxes'][0].set_facecolor('cyan')
bp3['boxes'][0].set(alpha=0.4)
bp3['boxes'][1].set_facecolor('yellow')
bp3['boxes'][1].set(alpha=0.3)
bp4 = ax.boxplot([MLP_1G, GRU_1G], positions=positions5, labels=['DDPG ', 'GRU-DDPG'], showmeans=True, meanline=True, meanprops=meanprops, medianprops=medianprops, patch_artist=True)
bp4['boxes'][0].set_facecolor('cyan')
bp4['boxes'][0].set(alpha=0.4)
bp4['boxes'][1].set_facecolor('yellow')
bp4['boxes'][1].set(alpha=0.3)

median_line = mpatches.Patch(color='orange', label='Median')
mean_line = mpatches.Patch(color='green', label='Mean')

# Adding text labels below each pair of box plots
labels = ['10G', '7G', '5G', '3G', '1G']
positions = [(positions1[0]+positions1[1])/2,
             (positions2[0]+positions2[1])/2,
             (positions3[0]+positions3[1])/2, (positions4[0]+positions4[1])/2, (positions5[0]+positions5[1])/2]  # x positions for the labels, centered between pairs
for pos, label in zip(positions, labels):
    ax.text(pos, -1.85, label, ha='center', va='top', fontsize=fontsize_used)  # Adjust these coordinates and properties as needed

# ax.legend(handles=[median_line, mean_line], loc='upper right')
ax.legend([bp0['medians'][0], bp0['means'][0]], ['median', 'mean'],loc='upper right',fontsize=fontsize_used+3)
# ax.legend(fontsize=fontsize_used, loc='upper left')
ax.set_xlabel('Number of geo-fences generated during evaluation', fontsize=fontsize_used+3)
ax.set_ylabel('Mean deviation across all evaluation episodes (m)', fontsize=fontsize_used+3)
ax.xaxis.set_label_coords(0.5, -0.12)

ax.tick_params(axis='y', labelsize=fontsize_used)
ax.tick_params(axis='x', labelsize=fontsize_used)
# ax.set_yticks(np.arange(0, 50, 10), fontsize=fontsize_used)
# ax.set_xticks(np.arange(0, 25, 5), fontsize=fontsize_used)
plt.tight_layout()
plt.show()
