# -*- coding: utf-8 -*-
"""
@Time    : 5/14/2024 5:07 PM
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

def extract_mean_and_std_of_mean_deviation(all_episode_situation):
    mean_deviation = []
    for each_episode in all_episode_situation:
        total_step_in_cur_eps = len(each_episode)
        accumulated_deviation = 0
        for each_step in each_episode:
            accumulated_deviation = accumulated_deviation + each_step[0][-1]['deviation_to_ref_line']
        mean_deviation_current_eps = accumulated_deviation / total_step_in_cur_eps
        mean_deviation.append(mean_deviation_current_eps)
    # Calculate the mean time
    mean_mean_deviation = np.mean(mean_deviation)
    # Calculate the standard deviation
    std_mean_deviation = np.std(mean_deviation)
    return mean_mean_deviation, std_mean_deviation

def obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation):
    reaching = [None] * len(all_episode_situation)
    time_to_reach_goal = []
    for each_episode_idx, each_episode in enumerate(all_episode_situation):
        total_step_in_cur_eps = len(each_episode)
        final_step = each_episode[-1]
        if final_step[0][2] == 15.0:
            reaching[each_episode_idx] = True
            time_to_reach_goal.append(len(each_episode))
        else:
            reaching[each_episode_idx] = False
    # Calculate mean success rate
    mean_success_rate = np.mean(reaching)
    # Calculate standard deviation for the proportion
    std_dev = np.sqrt(mean_success_rate * (1 - mean_success_rate) / len(reaching))
    # Expressing as percentage
    mean_reaching_rate_percent = mean_success_rate * 100
    std_reaching_rate_percent = std_dev * 100
    # Calculate the mean time
    mean_time_to_goal = np.mean(time_to_reach_goal)
    # Calculate the standard deviation
    std_time_to_goal = np.std(time_to_reach_goal)
    return mean_reaching_rate_percent, std_reaching_rate_percent, mean_time_to_goal, std_time_to_goal

with open(r'F:\githubClone\Multi_agent_AAC\single_drone_DDPG_changemap_GRU_LSTM_seqLength\GRU_7G_140524_14_49_09_1000eps_random1.pickle', 'rb') as handle:
    all_episode_situation = pickle.load(handle)
mean_mean_deviation, std_mean_deviation = extract_mean_and_std_of_mean_deviation(all_episode_situation)
mean_reaching_rate_percent, std_reaching_rate_percent, \
mean_time_to_goal, std_time_to_goal, = obtain_mean_std_reaching_rate_time_to_goal(all_episode_situation)

print("done")