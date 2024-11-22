"""
@Time    : 13/8/2024 1:28 PM
@Author  : Bizhao Pang
@FileName:
@Description: plot all results for the special issue paper
@___Parameters___ to tune the training curve's shape and noise variance:
    1) window_size
    2) step
"""
#
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Specify the file path
# file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'
#
# # Read the data from the CSV file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Convert the DataFrame to a numpy array for easier manipulation
# rewards = data.values  # No need to transpose if shape is (4, 16384)
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=100):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=500) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=200)  # Plot every 10th episode
# smoothed_rewards = smoothed_rewards[:, ::200]
# # Plotting
# plt.figure(figsize=(8, 6))
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='Model 1')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='Model 2')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='Model 3')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# plt.plot(episodes, smoothed_rewards[3], 'brown', label='Model 4')
# plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Adding labels and title
# plt.xlabel('Training episode')
# plt.ylabel('Reward')
# plt.title('Training Curve with Confidence Interval')
# plt.legend()
#
# # Show plot
# plt.show()
#
#

#___________plot training curves______________v1.0
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # Specify the file path
# file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'
#
# # Read the data from the Excel file, no header since the file contains only reward values
# data = pd.read_excel(file_path, header=None)
#
# # Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
# data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#
# # Transpose the DataFrame since your data is now (20000, 4) instead of (4, 20000)
# # “-5000” is to move y-axis, “/50” is to reduce magnitude effect
# rewards = (data.values.T - 4000)/50  # Transpose the data to get shape (4, 20000)
#
# # Smoothing function (moving average)
# def moving_average(data, window_size=200):
#     return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
#
# # Apply smoothing
# smoothed_rewards = np.array([moving_average(reward, window_size=2000) for reward in rewards])
#
# # Calculate mean and confidence intervals for smoothed data
# mean_rewards = smoothed_rewards.mean(axis=1)
# std_rewards = smoothed_rewards.std(axis=1)
#
# # X-axis values (episodes)
# episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)  # Plot every 200th episode
# smoothed_rewards = smoothed_rewards[:, ::50]
#
# # Set global font size
# plt.rcParams.update({'font.size': 12})
#
# # Plotting
# plt.figure(figsize=(8, 8))
#
# # Model 1
# plt.plot(episodes, smoothed_rewards[0], 'r-', label='IDDPG')
# plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)
#
# # Model 2
# plt.plot(episodes, smoothed_rewards[1], 'navy', label='IDDPG-n')
# plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)
#
# # Model 3
# plt.plot(episodes, smoothed_rewards[2], 'purple', label='IDDPG-s')
# plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)
#
# # Model 4
# plt.plot(episodes, smoothed_rewards[3], 'g', label='IDDPG-ns')
# plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)
#
# # Adding labels and title
# plt.xlabel('Training episode')
# plt.ylabel('Reward')
# # plt.title('Training Curve with Confidence Interval')
# plt.legend()
#
# # Show plot
# plt.show()

#___________plot training curves______________v2.0

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify the file path
file_path = r'D:\MADDPG_2nd_jp\training curves.xlsx'

# Read the data from the Excel file, no header since the file contains only reward values
data = pd.read_excel(file_path, header=None)

# Convert all data to numeric, forcing errors to NaN (you can skip this step if not needed)
data = data.applymap(lambda x: pd.to_numeric(x, errors='coerce'))

# Transpose the DataFrame since your data is now (20000, 4) instead of (4, 20000)
# "-5000" is to move y-axis, "/100" is to reduce magnitude effect
rewards = (data.values.T - 5000)/100  # Transpose the data to get shape (4, 20000)

# Smoothing function (moving average)
def moving_average(data, window_size=200):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Apply smoothing
smoothed_rewards = np.array([moving_average(reward, window_size=800) for reward in rewards])

# Calculate mean and confidence intervals for smoothed data
mean_rewards = smoothed_rewards.mean(axis=1)
std_rewards = smoothed_rewards.std(axis=1)

# X-axis values (episodes)
episodes = np.arange(1, len(smoothed_rewards[0]) + 1, step=50)  # Plot every 200th episode
smoothed_rewards = smoothed_rewards[:, ::50]

# Set global font size
plt.rcParams.update({'font.size': 14})

# Plotting
plt.figure(figsize=(8, 6))  # Slightly increased figure size for better readability

# Model 1
plt.plot(episodes, smoothed_rewards[0], 'r-', label='IDDPG')
plt.fill_between(episodes, smoothed_rewards[0] - std_rewards[0], smoothed_rewards[0] + std_rewards[0], color='red', alpha=0.2)

# Model 2
plt.plot(episodes, smoothed_rewards[1], 'navy', label='IDDPG-n')
plt.fill_between(episodes, smoothed_rewards[1] - std_rewards[1], smoothed_rewards[1] + std_rewards[1], color='navy', alpha=0.2)

# Model 3
plt.plot(episodes, smoothed_rewards[2], 'purple', label='IDDPG-s')
plt.fill_between(episodes, smoothed_rewards[2] - std_rewards[2], smoothed_rewards[2] + std_rewards[2], color='purple', alpha=0.2)

# Model 4
plt.plot(episodes, smoothed_rewards[3], 'g', label='IDDPG-ns')
plt.fill_between(episodes, smoothed_rewards[3] - std_rewards[3], smoothed_rewards[3] + std_rewards[3], color='brown', alpha=0.2)

# Customize x-axis labels to display as 2k, 4k, 6k, ..., 20k
plt.xticks(ticks=np.arange(0, 20001, 2000), labels=[f'{x//1000+1}k' for x in np.arange(0, 20001-800, 2000-80)])

# Adding labels and title
plt.xlabel('Training episode', fontsize=14)
plt.ylabel('Reward', fontsize=14)
plt.legend()

# Improve the layout
plt.tight_layout()

# Show plot
plt.show()




#_______________plot box figure using mean and std____________v1.0
# import numpy as np
# import matplotlib.pyplot as plt
#
# # Simulate data based on provided means and standard deviations
# np.random.seed(42)  # For reproducibility
#
# # Number of aircraft
# aircraft_counts = [4, 5, 6, 7, 8]
#
# # Means and standard deviations for each group
# means = [1.0754, 1.1656, 1.1744, 1.1613, 1.1725]
# std_devs = [0.0566, 0.1479, 0.1639, 0.1691, 0.1743]
#
# # Simulate data for each aircraft count
# data = [np.random.normal(mean, std_dev, 100) for mean, std_dev in zip(means, std_devs)]
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
# # Adding labels and title
#
# # Create box plot
# plt.figure(figsize=(8, 6))
# plt.boxplot(data, labels=aircraft_counts, patch_artist=True)
#
#
# plt.xlabel('Number of aircraft')
# plt.ylabel('Flight distance ratio')
# # plt.title('Scalability Analysis: Flight Distance Ratio with Increased Aircraft')
#
# # Show plot
# plt.show()


# v2.0___in use
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.lines as mlines
#
# # Simulate data based on provided means and standard deviations
# np.random.seed(42)  # For reproducibility
#
# # Number of aircraft
# aircraft_counts = [4, 5, 6, 7, 8]
#
# # Means and standard deviations for each group
# means = [1.0754, 1.1656, 1.1744, 1.1613, 1.1725]
# std_devs = [0.0566, 0.1479, 0.1639, 0.1691, 0.1743]
#
# # Simulate data for each aircraft count
# data = [np.random.normal(mean, std_dev, 100) for mean, std_dev in zip(means, std_devs)]
#
# # Set global font size
# plt.rcParams.update({'font.size': 14})
#
# # Create box plot with additional features
# plt.figure(figsize=(8, 6))
# box = plt.boxplot(data, labels=aircraft_counts, patch_artist=True, notch=True,
#                   showmeans=True, meanline=True, whiskerprops=dict(linewidth=2))
#
# # Customizing the box plot appearance
# colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightpink', 'lightgrey']
# for patch, color in zip(box['boxes'], colors):
#     patch.set_facecolor(color)
#
# # Adding mean markers
# for mean in box['means']:
#     mean.set(marker='o', color='black', markersize=3)
#
# # Adding median lines
# for median in box['medians']:
#     median.set(color='red', linewidth=2)
#
# # Customizing whiskers and caps
# for whisker in box['whiskers']:
#     whisker.set(color='black', linestyle='-', linewidth=2)
#
# for cap in box['caps']:
#     cap.set(color='black', linewidth=2)
#
# # Adding labels and title
# plt.xlabel('Number of aircraft')
# plt.ylabel('Flight distance ratio')
# # plt.title('Scalability Analysis: Flight Distance Ratio with Increased Aircraft')
#
# # Adding a grid for better readability
# plt.grid(True, linestyle='--', alpha=0.7)
#
# # Create custom legend
# mean_line = mlines.Line2D([], [], color='g', marker='o', linestyle='None', markersize=3, label='Mean')
# median_line = mlines.Line2D([], [], color='red', linewidth=2, label='Median')
#
# plt.legend(handles=[mean_line, median_line], loc='upper left')
# # Show plot
# plt.show()


#__________plot rubostness under different weathers_______v1.0
import matplotlib.pyplot as plt
import numpy as np

# Set the font to "Times New Roman" globally
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'text.usetex': False
})
# Data from Table 7
scenarios = ['c_1S', 'c_2S', 'c_3S', 'c_1L', 'c_2L', 'c_3L']
collision_rate_aircraft = [1, 0, 3, 4, 4, 3]
collision_rate_storm_cell = [5, 12, 11, 10, 24, 34]
goal_reach_rate = [94, 88, 86, 86, 72, 63]
flight_distance_ratio = [1.1436, 1.2205, 1.2593, 1.1955, 1.2915, 1.2471]

# Set up the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
# fig.suptitle('Simulation Results Under Diverse Weather Scenarios', fontsize=16, fontweight='bold')

# Set a subtle background color
# fig.patch.set_facecolor('#f2f2f2')
# for ax in axes.flat:
#     ax.set_facecolor('#f7f7f7')

# Bar width
bar_width = 0.4
index = np.arange(len(scenarios))

# Define a color palette with transparency
colors = [[0.121, 0.466, 0.705, 0.8], [1.0, 0.498, 0.054, 0.8], [0.173, 0.627, 0.173, 0.8], [0.839, 0.153, 0.157, 0.8]]

# Plotting Collision Rate with Aircraft
axes[0, 0].bar(index, collision_rate_aircraft, bar_width, color=colors[0], edgecolor='black')
axes[0, 0].set_xlabel('Scenario', fontsize=12)
axes[0, 0].set_ylabel('Aircraft LOS rate (%)', fontsize=12)
axes[0, 0].set_xticks(index)
axes[0, 0].set_xticklabels(scenarios, fontsize=12)
axes[0, 0].tick_params(axis='y', labelsize=12)
axes[0, 0].spines['top'].set_visible(False)
axes[0, 0].spines['right'].set_visible(False)
axes[0, 0].set_title('Aircraft LOS rate', fontsize=12, fontweight='bold')

# Plotting Collision Rate with Storm Cells
axes[0, 1].bar(index, collision_rate_storm_cell, bar_width, color=colors[1], edgecolor='black')
axes[0, 1].set_xlabel('Scenario', fontsize=12)
axes[0, 1].set_ylabel('Thunderstorm LOS rate (%)', fontsize=12)
axes[0, 1].set_xticks(index)
axes[0, 1].set_xticklabels(scenarios, fontsize=12)
axes[0, 1].tick_params(axis='y', labelsize=12)
axes[0, 1].spines['top'].set_visible(False)
axes[0, 1].spines['right'].set_visible(False)
axes[0, 1].set_title('Thunderstorm LOS rate', fontsize=12, fontweight='bold')

# Plotting Goal Reach Rate
axes[1, 0].bar(index, goal_reach_rate, bar_width, color=colors[2], edgecolor='black')
axes[1, 0].set_xlabel('Scenario', fontsize=12)
axes[1, 0].set_ylabel('Goal reach rate (%)', fontsize=12)
axes[1, 0].set_xticks(index)
axes[1, 0].set_xticklabels(scenarios, fontsize=12)
axes[1, 0].tick_params(axis='y', labelsize=12)
axes[1, 0].spines['top'].set_visible(False)
axes[1, 0].spines['right'].set_visible(False)
axes[1, 0].set_title('Goal reach rate', fontsize=12, fontweight='bold')

# Plotting Flight Distance Ratio
axes[1, 1].bar(index, flight_distance_ratio, bar_width, color=colors[3], edgecolor='black')
axes[1, 1].set_xlabel('Scenario', fontsize=12)
axes[1, 1].set_ylabel('Flight distance ratio', fontsize=12)
axes[1, 1].set_xticks(index)
axes[1, 1].set_xticklabels(scenarios, fontsize=12)
axes[1, 1].tick_params(axis='y', labelsize=12)
axes[1, 1].spines['top'].set_visible(False)
axes[1, 1].spines['right'].set_visible(False)
axes[1, 1].set_title('Flight distance ratio', fontsize=12, fontweight='bold')

# Adjust layout for better spacing
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
