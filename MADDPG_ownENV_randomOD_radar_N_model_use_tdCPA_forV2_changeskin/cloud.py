# -*- coding: utf-8 -*-
"""
@Time    : 6/27/2024 6:25 PM
@Author  : Mingcheng & Bizhao
@FileName:
@Description:
@Package dependency:
"""


class cloud_agent:
    def __init__(self, cloud_idx):
        self.agent_name = 'cloud_%s' % cloud_idx
        self.ini_pos = None
        self.pos = None
        self.pre_pos = None
        self.cloud_actual_cur_shape = None
        self.cloud_actual_previous_shape = None
        self.vel = 2  # m/s
        self.radius = 18  # nautical mile in radius
        self.goal = None
        self.reach_target = False
        self.spawn_cluster_pt_x_range = (-10, 10)
        self.spawn_cluster_pt_y_range = (-10, 10)
        self.cluster_pt_contour_x_range = (-15, 15)
        self.cluster_pt_contour_y_range = (-15, 15)
        # self.appoximate_circle_diameter = 4  # by approximation, the cloud's contours should not exceed this range
        self.trajectory = []
        self.cluster_centres = None
        self.x_fact = 3
        self.y_fact = 2



