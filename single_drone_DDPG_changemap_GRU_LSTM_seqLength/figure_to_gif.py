# -*- coding: utf-8 -*-
"""
@Time    : 5/31/2024 3:43 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pytz
import matplotlib.animation as animation
import pandas as pd
from datetime import datetime


class LineObject:
    def __init__(self, line_idx, x_start=0, y_start=0, x_end=0, y_end=0):
        self.line_idx = line_idx
        self.x_start = x_start
        self.y_start = y_start
        self.x_end = x_end
        self.y_end = y_end
        self.x_data = []
        self.y_data = []



def convert_to_sg_time(utc_timestamp):
    # Convert UTC timestamp to datetime
    utc_time = datetime.utcfromtimestamp(utc_timestamp).replace(tzinfo=pytz.utc)

    # Convert UTC time to Singapore time
    sg_time = utc_time.astimezone(singapore)

    # Format as MM_DD_HH_MM_SS
    formatted_sg_time = sg_time.strftime('%m_%d_%H_%M_%S')

    return formatted_sg_time


def convert_to_utc_timestamp(filename):
    # Parse the filename (without file extension)
    name, _ = os.path.splitext(filename)
    month, day, hour, minute, second = map(int, name.split('_'))

    # Create a datetime object (assuming the year is 2024)
    dt = datetime(2024, month, day, hour, minute, second)

    # Convert to UTC timestamp
    utc_timestamp = int(dt.timestamp())

    return utc_timestamp


def generate_random_lines(num_lines, x_min, x_max, y_min, y_max):
    lines = []
    for _ in range(num_lines):
        x_start = np.random.uniform(x_min, x_max)
        y_start = np.random.uniform(y_min, y_max)
        length = 0  # Start with length 0
        angle = np.random.uniform(0, 2 * np.pi)
        x_end = x_start + length * np.cos(angle)
        y_end = y_start + length * np.sin(angle)
        lines.append((x_start, y_start, x_end, y_end, length, angle))
    return lines


def add_axes_to_image(ax, img, x_min, x_max, y_min, y_max):
    # ax.imshow(np.array(img), extent=[x_min, x_max, y_min, y_max])
    image = ax.imshow(np.array(img), extent=[x_min, x_max, y_min, y_max])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    return image


def compile_gif_with_axes(image_folder, output_file, x_min, x_max, y_min, y_max, frame_duration, flight_number_dict):
    # Get list of image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files.sort()  # Ensure the images are in the correct order

    fig, ax = plt.subplots()
    # Initialize with the first image
    first_img_path = os.path.join(image_folder, image_files[0])
    first_img = Image.open(first_img_path)
    img_artist = add_axes_to_image(ax, first_img, x_min, x_max, y_min, y_max)

    num_lines = len(flight_number_dict)
    lines_to_draw = []
    line_segments = []
    for line_number in range(num_lines):
        # line_segment, = plt.plot([], [], color='blue', linewidth=1)
        line_segment, = plt.plot([], [], linewidth=0.5)
        line_segments.append(line_segment)
        lines_to_draw.append(LineObject(line_number))

    def animate(i):
        nonlocal img_artist
        img_path = os.path.join(image_folder, image_files[i])
        if i == len(image_files) - 1:
            limiting_frame = i
            # print("This is the last frame.")
        else:
            limiting_frame = i+1
        mm_dd_hh_mm_ss, file_extension = os.path.splitext(image_files[limiting_frame])
        utc_timestamp_limit = convert_to_utc_timestamp(mm_dd_hh_mm_ss)
        img = Image.open(img_path)
        img_artist.remove()  # Remove the previous image
        img_artist = add_axes_to_image(ax, img, x_min, x_max, y_min, y_max)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')

        plot_traj_count = 0
        # # plot trajectories
        for line_idx, flight_number in enumerate(flight_number_dict):
            flight_data = flight_number_dict[flight_number]
            flight_data = flight_data.drop_duplicates(subset=11, keep='first')
            # if flight_data[flight_data[11] == mm_dd_hh_mm_ss].empty:
            #     pass
            # else:

            # obtain data points that is less than utc_timestamp_limit
            # Filter the DataFrame
            data_to_plot_this_frame = flight_data[flight_data[10] < utc_timestamp_limit]

            # Clear previous data
            lines_to_draw[line_idx].x_data.clear()
            lines_to_draw[line_idx].y_data.clear()

            for row_index, row_data in data_to_plot_this_frame.iterrows():
                if len(line_segments[line_idx]._x) == 0: # meaning it is the first position point of the flight
                    lines_to_draw[line_idx].x_start = row_data[2]
                    lines_to_draw[line_idx].x_end = row_data[2] + 0.000001
                    lines_to_draw[line_idx].y_start = row_data[1]
                    lines_to_draw[line_idx].y_end = row_data[1] + 0.000001

                    # Append the starting point
                    lines_to_draw[line_idx].x_data.append(lines_to_draw[line_idx].x_start)
                    lines_to_draw[line_idx].y_data.append(lines_to_draw[line_idx].y_start)
                else:
                    lines_to_draw[line_idx].x_end = row_data[2]
                    lines_to_draw[line_idx].y_end = row_data[1]
                    # line_segments[0].set_data([lines_to_draw[0].x_start, lines_to_draw[0].x_end],
                    #                                        [lines_to_draw[0].y_start, lines_to_draw[0].y_end])

                # Append new points to the data lists
                lines_to_draw[line_idx].x_data.append(lines_to_draw[line_idx].x_end)
                lines_to_draw[line_idx].y_data.append(lines_to_draw[line_idx].y_end)

            # Update line segment with all accumulated points
            line_segments[line_idx].set_data(lines_to_draw[line_idx].x_data, lines_to_draw[line_idx].y_data)
            plot_traj_count = plot_traj_count + 1
            if plot_traj_count == 100:
                break
    ani = animation.FuncAnimation(fig, animate, frames=len(image_files), interval=frame_duration)

    ani.save(output_file, writer='pillow', fps=1000 // frame_duration)

    plt.close(fig)
    print(f'GIF saved as {output_file}')

matplotlib.use('TkAgg')
# Define parameters
image_folder = r'D:\4DT prediction_2024\weather image 4.1_1841 - 2138_36 images_3hours'
csv_file = r'D:\4DT prediction_2024\SIN_SPACELAUNCH_24-04-01.csv'
output_file = 'output_traj_100_alti_20000_LW_05.gif'
# Define the Singapore timezone
singapore = pytz.timezone('Asia/Singapore')
x_min, x_max = 101.15, 117.50
y_min, y_max = -1.85, 11.50
frame_duration = 500  # Duration for each frame in milliseconds
# Read CSV data
data = pd.read_csv(csv_file, header=None)
# Assign headers starting from 0
data.columns = range(data.shape[1])

# do a filter by trajectory position, long-x, lat-y
# [1] is y-axis, latitude
# [2] is x-axis, longitude
data = data[(data[1] >= y_min) & (data[1] <= y_max) &
                     (data[2] >= x_min) & (data[2] <= x_max)]
# filter by altitude
data = data[data[4] >= 20000]
filenames = os.listdir(image_folder)
# Convert all filenames to UTC timestamps
utc_timestamps = [convert_to_utc_timestamp(filename) for filename in filenames]
# Determine the minimum and maximum UTC timestamps
min_timestamp = min(utc_timestamps)
max_timestamp = max(utc_timestamps)

# Filter the DataFrame by the range of UTC timestamps
filtered_by_UTC_timestamps = data[(data[10] >= min_timestamp) & (data[10] <= max_timestamp)]

grouped_by_flight_number = filtered_by_UTC_timestamps.groupby(0)  # "0" stand for column 0.

flight_number_dict = {}

for key, FL_group in grouped_by_flight_number:
    FL_group_sorted_by_utc_timestamp = FL_group.sort_values(by=10)
    FL_group_sorted_by_utc_timestamp[11] = FL_group_sorted_by_utc_timestamp[10].apply(convert_to_sg_time)
    # drop unwanted data
    FL_group_sorted_by_utc_timestamp_clean = FL_group_sorted_by_utc_timestamp[[0, 1, 2, 3, 4, 5, 10, 11]]
    flight_number_dict[key] = FL_group_sorted_by_utc_timestamp_clean

# Compile GIF with axes
compile_gif_with_axes(image_folder, output_file, x_min, x_max, y_min, y_max, frame_duration, flight_number_dict)