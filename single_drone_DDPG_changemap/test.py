# -*- coding: utf-8 -*-
"""
@Time    : 1/15/2024 2:51 PM
@Author  : Thu Ra
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import heapq
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure

##############################################################################

# plot grid
# https://www.analytics-link.com/post/2018/09/14/applying-the-a-path-finding-algorithm-in-python-part-1-2d-square-grid
##############################################################################


# grid = np.array([
#
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0],
#
#     [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#
#     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#
# # start point and goal
#
# start = (0, 0)
#
# goal = (0, 19)
#
#
# ##############################################################################
#
# # heuristic function for path scoring
#
# ##############################################################################
#
#
# def heuristic(a, b):
#     return np.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)
#
#
# ##############################################################################
#
# # path finding function
#
# ##############################################################################
#
#
# def astar(array, start, goal):
#     neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
#
#     close_set = set()
#
#     came_from = {}
#
#     gscore = {start: 0}
#
#     fscore = {start: heuristic(start, goal)}
#
#     oheap = []
#
#     heapq.heappush(oheap, (fscore[start], start))
#
#     while oheap:
#
#         current = heapq.heappop(oheap)[1]
#
#         if current == goal:
#
#             data = []
#
#             while current in came_from:
#                 data.append(current)
#
#                 current = came_from[current]
#
#             return data
#
#         close_set.add(current)
#
#         for i, j in neighbors:
#
#             neighbor = current[0] + i, current[1] + j
#
#             tentative_g_score = gscore[current] + heuristic(current, neighbor)
#
#             if 0 <= neighbor[0] < array.shape[0]:
#
#                 if 0 <= neighbor[1] < array.shape[1]:
#
#                     if array[neighbor[0]][neighbor[1]] == 1:
#                         continue
#
#                 else:
#
#                     # array bound y walls
#
#                     continue
#
#             else:
#
#                 # array bound x walls
#
#                 continue
#
#             if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
#                 continue
#
#             if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
#                 came_from[neighbor] = current
#
#                 gscore[neighbor] = tentative_g_score
#
#                 fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
#
#                 heapq.heappush(oheap, (fscore[neighbor], neighbor))
#
#     return False
#
#
# route = astar(grid, start, goal)
#
# route = route + [start]
#
# route = route[::-1]
#
# print(route)
#
# ##############################################################################
#
# # plot the path
#
# ##############################################################################
#
#
# # extract x and y coordinates from route list
#
# x_coords = []
#
# y_coords = []
#
# for i in (range(0, len(route))):
#     x = route[i][0]
#
#     y = route[i][1]
#
#     x_coords.append(x)
#
#     y_coords.append(y)
#
# # plot map and path
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# matplotlib.use('TkAgg')
# fig, ax = plt.subplots(figsize=(20, 20))
#
# ax.imshow(grid, cmap=plt.cm.Dark2)
#
# ax.scatter(start[1], start[0], marker="*", color="yellow", s=200)
#
# ax.scatter(goal[1], goal[0], marker="*", color="red", s=200)
#
# ax.plot(y_coords, x_coords, color="black")
#
# plt.show()
import random
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
matplotlib.use('TkAgg')
# Generate the data again due to the previous issue
# Create a random line
line_start = (random.uniform(0, 10), random.uniform(0, 10))
line_end = (random.uniform(0, 10), random.uniform(0, 10))
line = LineString([line_start, line_end])

# Scatter 5 points around the line
scatter_points = []
buffer_distance = 1.0  # Define a buffer distance to spread points around the line
for _ in range(5):
    # Find a random point along the line
    random_point_on_line = line.interpolate(random.uniform(0, line.length), normalized=True)
    # Buffer the point to a random position within a certain distance
    angle = random.uniform(0, 2 * np.pi)
    offset = (buffer_distance * np.cos(angle), buffer_distance * np.sin(angle))
    scatter_point = Point(random_point_on_line.x + offset[0], random_point_on_line.y + offset[1])
    scatter_points.append(scatter_point)
# Find the nearest point on the line from these scatter points
# nearest_points = [line.interpolate(line.project(point)) for point in scatter_points]
# nearest_point = min(nearest_points, key=lambda point: line.distance(point))

nearest_points_list = []
for point in scatter_points:
    nearest_pt = nearest_points(point, line)[1]
    nearest_points_list.append(nearest_pt)

# Plotting the line and the points
x, y = line.xy
plt.plot(x, y, 'b', linewidth=3, label='Reference Line')

# Scatter points
scatter_x = [point.x for point in scatter_points]
scatter_y = [point.y for point in scatter_points]
plt.scatter(scatter_x, scatter_y, color='red', zorder=5, label='Scatter Points')

# Nearest points on the line
nearest_x = [point.x for point in nearest_points_list]
nearest_y = [point.y for point in nearest_points_list]
plt.scatter(nearest_x, nearest_y, color='green', zorder=5, label='Nearest Points on Line')

# # Highlight the single nearest point
# plt.scatter(nearest_point.x, nearest_point.y, color='yellow', edgecolor='black',
#             zorder=10, label='Single Nearest Point', s=100)

plt.xlabel('X-coordinate')
plt.ylabel('Y-coordinate')
plt.title('Scatter Points and Nearest Points on the Reference Line')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()