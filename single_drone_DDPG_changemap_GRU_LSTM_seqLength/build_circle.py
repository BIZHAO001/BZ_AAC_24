# -*- coding: utf-8 -*-
"""
@Time    : 5/30/2024 4:29 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from shapely.ops import unary_union
from shapely.geometry import MultiPoint
from shapely.geometry import Point, Polygon
import math
import matplotlib
matplotlib.use('TkAgg')


# Corrected function to create small circles such that exterior points are tangent points
def create_corrected_small_circles_outside(center, exterior_coords, small_circle_radius, big_circles):
    small_circles = []
    for x, y in exterior_coords:
        # Calculate the direction vector from the center of the big circle to the exterior point
        direction_vector = np.array([x - center[0], y - center[1]]) / np.linalg.norm([x - center[0], y - center[1]])

        # Translate the center of the small circle in the direction of the direction vector outside the big circle
        center_x, center_y = x + direction_vector[0] * small_circle_radius, y + direction_vector[
            1] * small_circle_radius
        small_circle = Point(center_x, center_y).buffer(small_circle_radius)

        # Check if the small circle overlaps with any of the big circles
        overlaps = any(small_circle.intersects(big_circle) for big_circle in big_circles if
                       big_circle != Point(center).buffer(big_circle_radius))
        if not overlaps:
            small_circles.append(small_circle)
    return small_circles


# Function to create multiple big circles and tangent small circles
def create_and_plot_multiple_circles(big_circle_centers, big_circle_radius, small_circle_radius):
    fig, ax = plt.subplots()

    big_circles = [Point(center).buffer(big_circle_radius) for center in big_circle_centers]

    for center, big_circle in zip(big_circle_centers, big_circles):
        # Extract exterior points of the big circle
        exterior_coords = np.array(big_circle.exterior.coords)

        # Create small circles with the corrected method for outside placement
        corrected_small_circles_outside = create_corrected_small_circles_outside(center, exterior_coords,
                                                                                 small_circle_radius, big_circle_radius,
                                                                                 big_circles)

        # Plot the big circle
        x_big, y_big = big_circle.exterior.xy
        ax.plot(x_big, y_big, color='blue')

        # Plot corrected small circles
        for circle in corrected_small_circles_outside:
            x, y = circle.exterior.xy
            ax.plot(x, y, color='red')

    ax.set_aspect('equal')
    plt.title('Multiple Big Circles with Corrected Tangent Small Circles Outside')
    plt.show()


# Define the centers for multiple big circles
big_circle_centers = [(0, 0), (15, 0), (0.9, 10.6), (15, 15), (-15, -15)]

# Define the big circle and the small circles' radii
big_circle_radius = 5
small_circle_radius = 2.5

# Call the function to create and plot multiple circles
create_and_plot_multiple_circles(big_circle_centers, big_circle_radius, small_circle_radius)


# # Corrected function to create small circles such that exterior points are tangent points
# def create_corrected_small_circles_outside(center, exterior_coords, small_circle_radius, big_circle_radius,
#                                            big_circles):
#     small_circles = []
#     for x, y in exterior_coords:
#         # Calculate the direction vector from the center of the big circle to the exterior point
#         direction_vector = np.array([x - center[0], y - center[1]]) / np.linalg.norm([x - center[0], y - center[1]])
#
#         # Translate the center of the small circle in the direction of the direction vector outside the big circle
#         center_x, center_y = x + direction_vector[0] * small_circle_radius, y + direction_vector[
#             1] * small_circle_radius
#         small_circle = Point(center_x, center_y).buffer(small_circle_radius)
#
#         # Check if the small circle overlaps with any of the big circles
#         overlaps = any(small_circle.intersects(big_circle) for big_circle in big_circles if
#                        big_circle != Point(center).buffer(big_circle_radius))
#         if not overlaps:
#             small_circles.append(small_circle)
#     return small_circles
#
#
# # Function to create multiple big circles and tangent small circles
# def create_and_plot_multiple_circles(big_circle_centers, big_circle_radius, small_circle_radius):
#     fig, ax = plt.subplots()
#     big_circles = [Point(center).buffer(big_circle_radius) for center in big_circle_centers]
#
#     for big_circle in big_circles:
#         # Extract exterior points of the big circle
#         exterior_coords = np.array(big_circle.exterior.coords)
#
#         # Create small circles with the corrected method for outside placement
#         corrected_small_circles_outside = create_corrected_small_circles_outside(big_circle.centroid.coords[0],
#                                                                                  exterior_coords, small_circle_radius,
#                                                                                  big_circle_radius, big_circles)
#
#         # Plot the big circle
#         x_big, y_big = big_circle.exterior.xy
#         ax.plot(x_big, y_big, color='blue')
#
#         # Plot corrected small circles
#         for circle in corrected_small_circles_outside:
#             x, y = circle.exterior.xy
#             ax.plot(x, y, color='red')
#
#     ax.set_aspect('equal')
#     plt.title('Multiple Big Circles with Corrected Tangent Small Circles Outside')
#     plt.show()
#
#
# # Define the centers for multiple big circles
# big_circle_centers = [(0, 0), (15, 0), (0, 15), (3.6, 2), (16.7, 20.8)]
#
# # Define the big circle and the small circles' radii
# big_circle_radius = 5
# small_circle_radius = 2.5
#
# # Call the function to create and plot multiple circles
# create_and_plot_multiple_circles(big_circle_centers, big_circle_radius, small_circle_radius)