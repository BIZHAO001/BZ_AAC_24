# -*- coding: utf-8 -*-
"""
@Time    : 3/2/2023 7:42 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import copy
import jps
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon
from scipy.spatial import KDTree
import random
import itertools
from copy import deepcopy
from agent_randomOD_tidy import Agent
import pandas as pd
import math
import numpy as np
import os
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from shapely.affinity import scale
import matplotlib.pyplot as plt
import matplotlib
import re
import time
from Utilities_own_randomOD_tidy import *
import torch as T
import torch
import torch.nn.functional as F
import torch.nn as nn


class env_simulator:
    def __init__(self, world_map, building_polygons, grid_length, bound, allGridPoly, agentConfig):  # allGridPoly[0][0] is all grid=1
        self.world_map_2D = world_map  # 2D binary matrix, in ndarray form.
        self.world_map_2D_jps = None
        self.centroid_to_position_empty = {}
        self.centroid_to_position_occupied = {}
        self.world_map_2D_polyList = allGridPoly  # [0][0] is all occupied polygon, [0][1] is all non-occupied polygon
        self.agentConfig = agentConfig
        self.gridlength = grid_length
        self.buildingPolygons = building_polygons  # contain all polygon in the world that has building
        self.world_STRtree = None  # contains all polygon in the environment
        self.allbuildingSTR = None  
        self.bound = bound
        self.global_time = 0.0  # in sec
        self.time_step = 0.5  # in second as well
        self.all_agents = None
        self.cur_allAgentCoor_KD = None
        self.OU_noise = None
        self.normalizer = None
        self.dummy_agent = None  # template for create a new agent
        self.max_agent_num = None

        self.spawn_area1 = []
        self.spawn_area1_polymat = []
        self.spawn_area2 = []
        self.spawn_area2_polymat = []
        self.spawn_area3 = []
        self.spawn_area3_polymat = []
        self.spawn_area4 = []
        self.spawn_area4_polymat = []
        self.spawn_pool = None
        self.target_area1 = []
        self.target_area1_polymat = None
        self.target_area2 = []
        self.target_area2_polymat = None
        self.target_area3 = []
        self.target_area3_polymat = None
        self.target_area4 = []
        self.target_area4_polymat = None
        self.target_pool = None

    def create_world(self, total_agentNum, n_actions, gamma, tau, target_update, largest_Nsigma, smallest_Nsigma, ini_Nsigma, max_xy, max_spd, acc_range):
        # config OU_noise
        self.OU_noise = OUNoise(n_actions, largest_Nsigma, smallest_Nsigma, ini_Nsigma)
        self.normalizer = NormalizeData([self.bound[0], self.bound[1]], [self.bound[2], self.bound[3]], max_spd, acc_range)
        self.all_agents = {}
        self.allbuildingSTR = STRtree(self.world_map_2D_polyList[0][0])
        worldGrid_polyCombine = []
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        self.world_STRtree = STRtree(worldGrid_polyCombine[0])
        for agent_i in range(total_agentNum):
            agent = Agent(n_actions, agent_i, gamma, tau, total_agentNum, max_spd)
            agent.target_update_step = target_update
            self.all_agents[agent_i] = agent
        self.dummy_agent = self.all_agents[0]

        # adjustment to world_map_2D
        # draw world_map_scatter
        scatterX = []
        scatterY = []
        centroid_pair_empty = []
        centroid_pair_occupied = []
        for poly in self.world_map_2D_polyList[0][1]:  # [0] is occupied, [1] is non occupied centroid
            scatterX.append(poly.centroid.x)
            scatterY.append(poly.centroid.y)
            centroid_pair_empty.append((poly.centroid.x, poly.centroid.y))
        for poly in self.world_map_2D_polyList[0][0]:  # [0] is occupied, [1] is non occupied centroid
            # scatterX.append(poly.centroid.x)
            # scatterY.append(poly.centroid.y)
            centroid_pair_occupied.append((poly.centroid.x, poly.centroid.y))
        start_x = int(min(scatterX))
        start_y = int(min(scatterY))
        end_x = int(max(scatterX))
        end_y = int(max(scatterY))
        world_2D = np.zeros((len(range(int(start_x), int(end_x+1), self.gridlength)), len(range(int(start_y), int(end_y+1), self.gridlength))))
        for j_idx, j_val in enumerate(range(start_y, end_y+1, self.gridlength)):
            for i_idx, i_val in enumerate(range(start_x, end_x+1, self.gridlength)):
                if (i_val, j_val) in centroid_pair_empty:
                    world_2D[i_idx][j_idx] = 0
                    self.centroid_to_position_empty[(i_val, j_val)] = [float(i_idx), float(j_idx)]
                elif (i_val, j_val) in centroid_pair_occupied:
                    world_2D[i_idx][j_idx] = 1
                    self.centroid_to_position_occupied[(i_val, j_val)] = [float(i_idx), float(j_idx)]
                else:
                    print("no corresponding coordinate found in side world 2D grid centroids, please debug!")
        self.world_map_2D = world_2D
        self.world_map_2D_jps = world_2D.astype(int).tolist()

        # segment them using two lines
        self.spawn_pool = [self.spawn_area1, self.spawn_area2, self.spawn_area3, self.spawn_area4]
        self.target_pool = [self.target_area1, self.target_area2, self.target_area3, self.target_area4]
        # target_pool_idx = [i for i in range(len(target_pool))]
        # get centroid of all square polygon
        non_occupied_polygon = self.world_map_2D_polyList[0][1]
        x_segment = (self.bound[1] - self.bound[0]) / 2 + self.bound[0]
        y_segment = (self.bound[3] - self.bound[2]) / 2 + self.bound[2]
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        for poly in non_occupied_polygon:
            centre_coord = (poly.centroid.x, poly.centroid.y)
            if poly.intersects(x_left_bound):
                self.spawn_area1.append(poly)
                # left line
                self.spawn_area1_polymat.append(shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='y'))
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_bottom_bound):
                # bottom line
                self.spawn_area2.append(poly)
                self.spawn_area2_polymat.append(
                    shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='m'))
                # ax.add_patch(poly_mat)
            elif poly.intersects(x_right_bound):
                # right line
                self.spawn_area3.append(poly)
                self.spawn_area3_polymat.append(
                    shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='b'))
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_top_bound):
                # top line
                self.spawn_area4.append(poly)
                self.spawn_area4_polymat.append(
                    shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='g'))
                # ax.add_patch(poly_mat)

            if centre_coord[0] < x_segment and centre_coord[1] < y_segment:
                self.target_area1.append(centre_coord)
                # bottom left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='y', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] < y_segment:
                self.target_area2.append(centre_coord)
                # bottom right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='m', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] > y_segment:
                self.target_area3.append(centre_coord)
                # top right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='b', markersize=2)
            else:
                self.target_area4.append(centre_coord)
                # top left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='g', markersize=2)

    def reset_world(self, total_agentNum, show):  # set initialize position and observation for all agents
        self.global_time = 0.0
        self.time_step = 0.5
        # reset OU_noise as well
        self.OU_noise.reset()
        # reset the drone index to 0,1,2, ensure all index reset at starting of a new episode
        keys = list(self.all_agents.keys())
        for new_Idx, current_Idx in zip(range(total_agentNum), keys):
            self.all_agents[new_Idx] = self.all_agents.pop(current_Idx)
            self.all_agents[new_Idx].agent_name = 'agent_%s' % new_Idx
            self.all_agents[new_Idx].pre_surroundingNeighbor = {}  # at start of each episode ensure all surrounding/pre-surrounding neighbours are cleared.
            self.all_agents[new_Idx].surroundingNeighbor = {}

        # start_time = time.time()
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value

        # with open('all_agents.pickle', 'rb') as handle:
        #     b = pickle.load(handle)
        # print('time used is {}'.format(time.time()-start_time))

        # start_time = time.time()
        start_pos_memory = []

        # any_collision = 0
        # loop_count = 0
        # while not any_collision:
        #     start_pos_memory = []  # make it here just to test any initial collision condition
        for agentIdx in self.all_agents.keys():

            # ---------------- using random initialized agent position for traffic flow ---------
            random_start_index = random.randint(0, len(self.target_pool) - 1)
            numbers_left = list(range(0, random_start_index)) + list(range(random_start_index + 1, len(self.target_pool)))
            random_target_index = random.choice(numbers_left)
            random_start_pos = random.choice(self.target_pool[random_start_index])
            if len(start_pos_memory) > 0:
                while len(start_pos_memory) < len(self.all_agents):  # make sure the starting drone generated do not collide with any existing drone
                    # Generate a new point
                    random_start_index = random.randint(0, len(self.target_pool) - 1)
                    numbers_left = list(range(0, random_start_index)) + list(
                        range(random_start_index + 1, len(self.target_pool)))
                    random_target_index = random.choice(numbers_left)
                    random_start_pos = random.choice(self.target_pool[random_start_index])
                    # Check that the distance to all existing points is more than 5
                    if all(np.linalg.norm(np.array(random_start_pos)-point) > self.all_agents[agentIdx].protectiveBound*2 for point in start_pos_memory):
                        break

            random_end_pos = random.choice(self.target_pool[random_target_index])
            dist_between_se = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            # while dist_between_se >= 100:  # the distance between start & end point is more than a threshold, we reset SE pairs.
            #     random_end_pos = random.choice(self.target_pool[random_target_index])
            #     dist_between_se = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))
            host_current_circle = Point(np.array(random_start_pos)[0], np.array(random_start_pos)[1]).buffer(self.all_agents[agentIdx].protectiveBound)

            possiblePoly = self.allbuildingSTR.query(host_current_circle)
            for element in possiblePoly:
                if self.allbuildingSTR.geometries.take(element).intersection(host_current_circle):
                    any_collision = 1
                    print("Initial start point {} collision with buildings".format(np.array(random_start_pos)))
                    break

            self.all_agents[agentIdx].pos = np.array(random_start_pos)
            self.all_agents[agentIdx].ini_pos = np.array(random_start_pos)
            start_pos_memory.append(np.array(random_start_pos))
            self.all_agents[agentIdx].removed_goal = None

            # large_start = [random_start_pos[0] / self.gridlength, random_start_pos[1] / self.gridlength]
            # large_end = [random_end_pos[0] / self.gridlength, random_end_pos[1] / self.gridlength]
            # small_area_map_start = [large_start[0] - math.ceil(self.bound[0] / self.gridlength),
            #                         large_start[1] - math.ceil(self.bound[2] / self.gridlength)]
            # small_area_map_end = [large_end[0] - math.ceil(self.bound[0] / self.gridlength),
            #                       large_end[1] - math.ceil(self.bound[2] / self.gridlength)]

            small_area_map_s = self.centroid_to_position_empty[random_start_pos]
            small_area_map_e = self.centroid_to_position_empty[random_end_pos]

            width = self.world_map_2D.shape[0]
            height = self.world_map_2D.shape[1]

            jps_map = self.world_map_2D_jps
            outPath = jps.find_path(small_area_map_s, small_area_map_e, width, height, jps_map)[0]
            # outPath = jps.find_path(small_area_map_start, small_area_map_end, 22, 13, self.world_map_2D)[0]

            refinedPath = []
            curHeading = math.atan2((outPath[1][1] - outPath[0][1]),
                                    (outPath[1][0] - outPath[0][0]))
            refinedPath.append(outPath[0])
            for id_ in range(2, len(outPath)):
                nextHeading = math.atan2((outPath[id_][1] - outPath[id_ - 1][1]),
                                         (outPath[id_][0] - outPath[id_ - 1][0]))
                if curHeading != nextHeading:  # add the "id_-1" th element
                    refinedPath.append(outPath[id_ - 1])
                    curHeading = nextHeading  # update the current heading
            refinedPath.append(outPath[-1])

            # load the to goal, but remove the 1st point, which is the initial position
            self.all_agents[agentIdx].goal = [[(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength, (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength]for points in refinedPath if not np.array_equal(np.array([(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength, (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength]), self.all_agents[agentIdx].ini_pos)]  # if not np.array_equal(np.array(points), self.all_agents[agentIdx].ini_pos)

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])

            # random_spd = random.randint(1, self.all_agents[agentIdx].maxSpeed)  # initial speed is randomly picked from 1 to max speed
            random_spd = random.randint(1, 3)  # initial speed is randomly picked from 1 to max speed
            self.all_agents[agentIdx].vel = np.array([random_spd*math.cos(self.all_agents[agentIdx].heading),
                                             random_spd*math.sin(self.all_agents[agentIdx].heading)])

            # NOTE: UAV's max speed don't change with time, so when we find it normalized bound, we use max speed
            # the below is the maximum normalized velocity range for map range -1 to 1, and maxSPD = 15m/s
            norm_vel_x_range = [
                -self.normalizer.scale_vel([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[0],
                self.normalizer.scale_vel([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[0]]
            norm_vel_y_range = [
                -self.normalizer.scale_vel([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[1],
                self.normalizer.scale_vel([self.all_agents[agentIdx].maxSpeed, self.all_agents[agentIdx].maxSpeed])[1]]

            # ----------------end of initialize normalized velocity, but based on normalized map. map pos_x & pos_y are normalized to [-1, 1]---------------

            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])

            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
            # # ----------------------- end of random initialized ------------------------------

            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            agentsCoor_list.append(self.all_agents[agentIdx].pos)
        
        #     loop_count = loop_count + 1
        #     if loop_count % 500 == 0:
        #         print("set {} generated".format(loop_count))
        # print("The {}th set starting point generated, one initial collision case happened".format(loop_count))

        # overall_state, norm_overall_state = self.cur_state_norm_state_fully_observable(agentRefer_dict)
        # print('time used is {}'.format(time.time() - start_time))
        overall_state, norm_overall_state = self.cur_state_norm_state_v3(agentRefer_dict)

        if show:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 1)
            for agentIdx, agent in self.all_agents.items():
                plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                                                                        transform=Affine2D().rotate_deg(
                                                                            math.degrees(agent.heading))), color='y')
                plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                # plot self_circle of the drone
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                ax.add_patch(detec_circle_mat)

                ini = agent.pos
                for wp in agent.goal:
                    plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                    ini = wp

            # draw occupied_poly
            for one_poly in self.world_map_2D_polyList[0][0]:
                one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                ax.add_patch(one_poly_mat)
            # draw non-occupied_poly
            for zero_poly in self.world_map_2D_polyList[0][1]:
                zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                # ax.add_patch(zero_poly_mat)

            # show building obstacles
            for poly in self.buildingPolygons:
                matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                ax.add_patch(matp_poly)

            # for ele in self.spawn_area1_polymat:
            #     ax.add_patch(ele)
            # for ele2 in self.spawn_area2_polymat:
            #     ax.add_patch(ele2)
            # for ele3 in self.spawn_area3_polymat:
            #     ax.add_patch(ele3)
            # for ele4 in self.spawn_area4_polymat:
            #     ax.add_patch(ele4)

            # plt.axvline(x=self.bound[0], c="green")
            # plt.axvline(x=self.bound[1], c="green")
            # plt.axhline(y=self.bound[2], c="green")
            # plt.axhline(y=self.bound[3], c="green")

            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.axis('equal')
            plt.show()

        return overall_state, norm_overall_state

    def reset_world_fixedOD(self, show):
        self.global_time = 0.0
        self.time_step = 0.5
        # reset OU_noise as well
        self.OU_noise.reset()

        #  custom agent position
        # x-bound: [0, 1800), y-bound: [0, 1300)
        # read the Excel file into a pandas dataframe
        df = pd.read_excel(self.agentConfig)
        # convert the dataframe to a NumPy array
        custom_agent_data = np.array(df)
        # custom_agent_data = custom_agent_data.astype(float)
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        for agentIdx in self.all_agents.keys():
            self.all_agents[agentIdx].pos = custom_agent_data[agentIdx][0:2]
            self.all_agents[agentIdx].ini_pos = custom_agent_data[agentIdx][0:2]
            self.all_agents[agentIdx].removed_goal = None

            if not isinstance(custom_agent_data[agentIdx][2:4][0], str):
                self.all_agents[agentIdx].goal = [custom_agent_data[agentIdx][2:4]]
            else:
                x_coords = np.array([int(coord.split('; ')[0]) for coord in custom_agent_data[agentIdx][2:4]])
                y_coords = np.array([int(coord.split('; ')[1]) for coord in custom_agent_data[agentIdx][2:4]])
                self.all_agents[agentIdx].goal = [x_coords, y_coords]

            # self.all_agents[agentIdx].vel = custom_agent_data[agentIdx][4:6]

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])

            random_spd = random.randint(0, self.all_agents[agentIdx].maxSpeed)  # initial speed is randomly picked from 0 to max speed
            self.all_agents[agentIdx].vel = np.array([random_spd*math.cos(self.all_agents[agentIdx].heading),
                                             random_spd*math.sin(self.all_agents[agentIdx].heading)])

            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])
            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            # agentSTR_list.append(cur_circle)
            agentsCoor_list.append(self.all_agents[agentIdx].pos)

        self.cur_allAgentCoor_KD = KDTree(agentsCoor_list)
        overall_state, norm_overall_state = self.cur_state_norm_state_v3(agentRefer_dict)

        if show:
            os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            matplotlib.use('TkAgg')
            fig, ax = plt.subplots(1, 1)
            for agentIdx, agent in self.all_agents.items():
                plt.plot(agent.pos[0], agent.pos[1], marker=MarkerStyle(">", fillstyle="right",
                                                                        transform=Affine2D().rotate_deg(
                                                                            math.degrees(agent.heading))), color='y')
                plt.text(agent.pos[0], agent.pos[1], agent.agent_name)
                # plot self_circle of the drone
                self_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.protectiveBound, cap_style='round')
                grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')
                ax.add_patch(grid_mat_Scir)

                # plot drone's detection range
                detec_circle = Point(agent.pos[0], agent.pos[1]).buffer(agent.detectionRange / 2, cap_style='round')
                detec_circle_mat = shapelypoly_to_matpoly(detec_circle, False, 'r')
                ax.add_patch(detec_circle_mat)

                ini = agent.pos
                for wp in agent.goal:
                    plt.plot(wp[0], wp[1], marker='*', color='y', markersize=10)
                    plt.plot([wp[0], ini[0]], [wp[1], ini[1]], '--', color='c')
                    ini = wp

            # draw occupied_poly
            for one_poly in self.world_map_2D_polyList[0][0]:
                one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
                ax.add_patch(one_poly_mat)
            # draw non-occupied_poly
            for zero_poly in self.world_map_2D_polyList[0][1]:
                zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
                # ax.add_patch(zero_poly_mat)

            # show building obstacles
            for poly in self.buildingPolygons:
                matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
                ax.add_patch(matp_poly)

            # plt.axvline(x=self.bound[0], c="green")
            # plt.axvline(x=self.bound[1], c="green")
            # plt.axhline(y=self.bound[2], c="green")
            # plt.axhline(y=self.bound[3], c="green")

            plt.xlabel("X axis")
            plt.ylabel("Y axis")
            plt.axis('equal')
            plt.show()

        return overall_state, norm_overall_state

    def fill_agent_reset(self, cur_state, norm_cur_state, added_agent_keys):
        if len(added_agent_keys) == 0:
            return cur_state, norm_cur_state
        # this function is serving as a small reset function, hence when filling "agentsCoor_list" & "agentRefer_dict" we need to include all existing agents
        agentsCoor_list = []  # for store all agents as circle polygon
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        # segment them using two lines
        spawn_area1 = []  # (yellow, bottom left)
        spawn_area2 = []  # (green, top left)
        spawn_area3 = []  # (megent, bottom right)
        spawn_area4 = []  # (black, middle right)
        spawn_pool = [spawn_area1, spawn_area2, spawn_area3, spawn_area4]
        target_area1 = []
        target_area2 = []
        target_area3 = []
        target_area4 = []
        target_pool = [target_area1, target_area2, target_area3, target_area4]
        # target_pool_idx = [i for i in range(len(target_pool))]
        # get centroid of all square polygon
        non_occupied_polygon = self.world_map_2D_polyList[0][1]
        x_segment = (self.bound[1] - self.bound[0]) / 2 + self.bound[0]
        y_segment = (self.bound[3] - self.bound[2]) / 2 + self.bound[2]
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])
        for poly in non_occupied_polygon:
            centre_coord = (poly.centroid.x, poly.centroid.y)
            if poly.intersects(x_left_bound):
                spawn_area1.append(poly)
                # left line
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='y')
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_bottom_bound):
                # bottom line
                spawn_area2.append(poly)
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='m')
                # ax.add_patch(poly_mat)
            elif poly.intersects(x_right_bound):
                # right line
                spawn_area3.append(poly)
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='b')
                # ax.add_patch(poly_mat)
            elif poly.intersects(y_top_bound):
                # top line
                spawn_area4.append(poly)
                poly_mat = shapelypoly_to_matpoly(poly, inFill=True, Edgecolor='black', FcColor='g')
                # ax.add_patch(poly_mat)

            if centre_coord[0] < x_segment and centre_coord[1] < y_segment:
                target_area1.append(centre_coord)
                # bottom left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='y', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] < y_segment:
                target_area2.append(centre_coord)
                # bottom right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='m', markersize=2)
            elif centre_coord[0] > x_segment and centre_coord[1] > y_segment:
                target_area3.append(centre_coord)
                # top right
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='b', markersize=2)
            else:
                target_area4.append(centre_coord)
                # top left
                # plt.plot(centre_coord[0], centre_coord[1], marker='.', color='g', markersize=2)

        for agentIdx in added_agent_keys:  # initialize for all newly added agents
            # ---------------- using random initialized agent position for traffic flow ---------
            random_start_index = random.randint(0, len(target_pool) - 1)
            numbers_left = list(range(0, random_start_index)) + list(range(random_start_index + 1, len(target_pool)))
            random_target_index = random.choice(numbers_left)
            random_start_pos = random.choice(target_pool[random_start_index])
            random_end_pos = random.choice(target_pool[random_target_index])
            dist_between = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))
            while dist_between <= 30:  # the distance between start & end point must be more than 30 meters
                random_end_pos = random.choice(target_pool[random_target_index])
                dist_between = np.linalg.norm(np.array(random_end_pos) - np.array(random_start_pos))

            self.all_agents[agentIdx].pre_surroundingNeighbor = {}
            self.all_agents[agentIdx].surroundingNeighbor = {}
            self.all_agents[agentIdx].observableSpace = []
            self.all_agents[agentIdx].reach_target = False
            self.all_agents[agentIdx].pre_vel = None
            self.all_agents[agentIdx].pre_pos = None
            self.all_agents[agentIdx].pos = np.array(random_start_pos)
            self.all_agents[agentIdx].ini_pos = np.array(random_start_pos)

            large_start = [random_start_pos[0] / self.gridlength, random_start_pos[1] / self.gridlength]
            large_end = [random_end_pos[0] / self.gridlength, random_end_pos[1] / self.gridlength]
            small_area_map_start = [large_start[0] - math.ceil(self.bound[0] / self.gridlength),
                                    large_start[1] - math.ceil(self.bound[2] / self.gridlength)]
            small_area_map_end = [large_end[0] - math.ceil(self.bound[0] / self.gridlength),
                                  large_end[1] - math.ceil(self.bound[2] / self.gridlength)]
            width = self.world_map_2D.shape[0]
            height = self.world_map_2D.shape[1]
            outPath = jps.find_path(small_area_map_start, small_area_map_end, width, height)[0]

            refinedPath = []
            curHeading = math.atan2((outPath[1][1] - outPath[0][1]),
                                    (outPath[1][0] - outPath[0][0]))
            refinedPath.append(outPath[0])
            for id_ in range(2, len(outPath)):
                nextHeading = math.atan2((outPath[id_][1] - outPath[id_ - 1][1]),
                                         (outPath[id_][0] - outPath[id_ - 1][0]))
                if curHeading != nextHeading:  # add the "id_-1" th element
                    refinedPath.append(outPath[id_ - 1])
                    curHeading = nextHeading  # update the current heading
            refinedPath.append(outPath[-1])

            self.all_agents[agentIdx].goal = [
                [(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                 (points[1] + math.ceil(self.bound[2] / self.gridlength)) * self.gridlength] for points in refinedPath if
                not np.array_equal(np.array([(points[0] + math.ceil(self.bound[0] / self.gridlength)) * self.gridlength,
                                             (points[1] + math.ceil(
                                                 self.bound[2] / self.gridlength)) * self.gridlength]), self.all_agents[
                                       agentIdx].ini_pos)]  # if not np.array_equal(np.array(points), self.all_agents[agentIdx].ini_pos)

            # heading in rad, must be goal_pos-intruder_pos, and y2-y1, x2-x1
            self.all_agents[agentIdx].heading = math.atan2(self.all_agents[agentIdx].goal[0][1] -
                                                           self.all_agents[agentIdx].pos[1],
                                                           self.all_agents[agentIdx].goal[0][0] -
                                                           self.all_agents[agentIdx].pos[0])

            random_spd = random.randint(0, self.all_agents[agentIdx].maxSpeed)
            self.all_agents[agentIdx].vel = np.array([random_spd * math.cos(self.all_agents[agentIdx].heading),
                                                      random_spd * math.sin(self.all_agents[agentIdx].heading)])

            self.all_agents[agentIdx].observableSpace = self.current_observable_space(self.all_agents[agentIdx])
            cur_circle = Point(self.all_agents[agentIdx].pos[0],
                               self.all_agents[agentIdx].pos[1]).buffer(self.all_agents[agentIdx].protectiveBound,
                                                                        cap_style='round')
        # after add an agent, all agent's neighbour should be updated
        for agentIdx in self.all_agents.keys():
            agentRefer_dict[(self.all_agents[agentIdx].pos[0],
                             self.all_agents[agentIdx].pos[1])] = self.all_agents[agentIdx].agent_name

            # agentSTR_list.append(cur_circle)
            agentsCoor_list.append(self.all_agents[agentIdx].pos)
        # self.cur_allAgentCoor_KD = KDTree(agentsCoor_list)
        overall_state, norm_overall_state = self.cur_state_norm_state_fully_observable(agentRefer_dict)  # update agent's surrounding is inside here
        return overall_state, norm_overall_state

    def get_current_agent_nei(self, cur_agent, agentRefer_dict):
        # identify neighbors (use distance)
        point_to_search = cur_agent.pos
        # subtract a small value to exclude point at exactly "search_distance"
        search_distance = (cur_agent.detectionRange / 2) + cur_agent.protectiveBound - 1e-6
        # indices_from_KDtree = self.cur_allAgentCoor_KD.query_ball_point(point_to_search, search_distance)
        for agent_idx, agent in self.all_agents.items():  # loop through all agent to confirm its neighbour
            if agent.agent_name == cur_agent.agent_name:  # skip the current querying agent
                continue
            cur_ts_dist = np.linalg.norm(agent.pos-cur_agent.pos)
            if cur_ts_dist <= search_distance:
                cur_agent.surroundingNeighbor[agent_idx] = np.array([agent.pos[0], agent.pos[1],
                                                                     agent.vel[0], agent.vel[1],
                                                                     agent.protectiveBound])


        return cur_agent.surroundingNeighbor

    def cur_state_norm_state_v2(self, agentRefer_dict):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        overall_state_p2 = []
        # prepare normalized output states
        norm_overall_state_p1 = []
        norm_overall_state_p2 = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')
            # get agent's observable space around it
            self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength(self.all_agents[agentIdx])
            # identify neighbors (use distance)
            # update the "surroundingNeighbor" attribute
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
            # reset_world function we initialized both "surroundingNeighbor" and "pre_surroundingNeighbor" identically
            agent.pre_surroundingNeighbor = agent.surroundingNeighbor

            # populate the output stateV2 overall_state is a list of length equals to total number of agents
            agent_own = np.array(
                [agent.pos[0], agent.pos[1], agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1],
                 agent.vel[0], agent.vel[1]])
            # populate normalized agent_own
            # norm_agent_own = []
            norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])

            norm_G_diff = self.normalizer.nmlz_pos_diff(
                [agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1]])

            norm_vel = self.normalizer.nmlz_vel([agent.vel[0], agent.vel[1]])
            norm_agent_own = list(norm_pos + norm_G_diff + norm_vel)

            other_pos = []
            norm_other_pos = []
            for other_agentIdx, other_agent in self.all_agents.items():
                if other_agentIdx != agent_idx:
                    other_pos.append(other_agent.pos - agent.pos)
                    norm_pos = self.normalizer.nmlz_pos(other_agent.pos - agent.pos)
                    norm_other_pos.append(np.array(norm_pos))

            # overall_state.append(np.array([agent_own, agent.observableSpace, agent.surroundingNeighbor], dtype=object))
            overall_state_p1.append(np.concatenate((agent_own, np.array(other_pos).flatten())))
            overall_state_p2.append(agent.observableSpace)
            norm_overall_state_p1.append(np.concatenate((norm_agent_own, np.array(norm_other_pos).flatten())))
            norm_overall_state_p2.append(agent.observableSpace)
        overall.append(overall_state_p1)
        overall.append(overall_state_p2)
        norm_overall.append(norm_overall_state_p1)
        norm_overall.append(norm_overall_state_p2)
        return overall, norm_overall

    def cur_state_norm_state_v3(self, agentRefer_dict):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        overall_state_p2 = []
        overall_state_p3 = []
        # prepare normalized output states
        norm_overall_state_p1 = []
        norm_overall_state_p2 = []
        norm_overall_state_p3 = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')

            # get agent's observable space around it
            # obs_grid_time = time.time()
            # self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength_fromv2_flow(self.all_agents[agentIdx])
            self.all_agents[agentIdx].observableSpace = np.zeros((9))
            # print("generate grid time is {} milliseconds".format((time.time()-obs_grid_time)*1000))
            #
            # identify neighbors (use distance)
            # obs_nei_time = time.time()
            agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
            # print("generate nei time is {} milliseconds".format((time.time() - obs_nei_time) * 1000))

            rest_compu_time = time.time()

            norm_pos = self.normalizer.scale_pos([agent.pos[0], agent.pos[1]])

            norm_vel = self.normalizer.scale_vel([agent.vel[0], agent.vel[1]])

            norm_G = self.normalizer.nmlz_pos([agent.goal[-1][0], agent.goal[-1][1]])

            norm_deltaG = norm_G - norm_pos

            agent_own = np.array([agent.pos[0], agent.pos[1], agent.vel[0], agent.vel[1],
                                  agent.goal[-1][0]-agent.pos[0], agent.goal[-1][1]-agent.pos[1]])

            norm_agent_own = np.concatenate([norm_pos, norm_vel, norm_deltaG], axis=0)

            other_agents = []
            norm_other_agents = []
            if len(agent.surroundingNeighbor) > 0:  # meaning there is surrounding neighbors around the current agent
                for other_agentIdx, other_agent in agent.surroundingNeighbor.items():
                    if other_agentIdx != agent_idx:
                        surround_agent = np.array([[other_agent[0] - agent.pos[0],
                                                   other_agent[1] - agent.pos[1],
                                                   other_agent[-2] - other_agent[0],
                                                   other_agent[-1] - other_agent[1],
                                                   other_agent[2], other_agent[3]]])

                        norm_pos_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1]])

                        norm_G_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[-2] - other_agent[0], other_agent[-1] - other_agent[1]])

                        norm_vel = self.normalizer.nmlz_vel([other_agent[2], other_agent[3]])
                        norm_surround_agent = np.array([list(norm_pos_diff + norm_G_diff + norm_vel)])

                        other_agents.append(surround_agent)
                        norm_other_agents.append(norm_surround_agent)
                overall_state_p3.append(other_agents)
                norm_overall_state_p3.append(norm_other_agents)
            else:
                overall_state_p3.append([np.zeros((1, 6))])
                norm_overall_state_p3.append([np.zeros((1, 6))])

            overall_state_p1.append(agent_own)
            overall_state_p2.append(agent.observableSpace)

            norm_overall_state_p1.append(norm_agent_own)
            norm_overall_state_p2.append(agent.observableSpace)

        overall.append(overall_state_p1)
        overall.append(overall_state_p2)
        overall.append(overall_state_p3)
        for list_ in overall_state_p3:
            if len(list_) == 0:
                print("check")
        norm_overall.append(norm_overall_state_p1)
        norm_overall.append(norm_overall_state_p2)
        norm_overall.append(norm_overall_state_p3)
        # print("rest compute time is {} milliseconds".format((time.time() - rest_compu_time) * 1000))
        return overall, norm_overall

    def cur_state_norm_state_fully_observable(self, agentRefer_dict):
        overall = []
        norm_overall = []
        # prepare for output states
        overall_state_p1 = []
        fullyObs_p1 = []
        overall_state_p2 = []
        overall_state_p3 = []
        # prepare normalized output states
        norm_overall_state_p1 = []
        norm_fullyObs_p1 = []
        norm_overall_state_p2 = []
        norm_overall_state_p3 = []
        # loop over all agent again to obtain each agent's detectable neighbor
        # second loop is required, because 1st loop is used to create the STR-tree of all agents
        # circle centre at their position
        for agentIdx, agent in self.all_agents.items():

            # get current agent's name in term of integer
            match = re.search(r'\d+(\.\d+)?', agent.agent_name)
            if match:
                agent_idx = int(match.group())
            else:
                agent_idx = None
                raise ValueError('No number found in string')
            # get agent's observable space around it
            self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength_fromv2_flow(self.all_agents[agentIdx])
            # self.all_agents[agentIdx].observableSpace = self.current_observable_space_fixedLength(self.all_agents[agentIdx])
            # identify neighbors (use distance)
            # update the "surroundingNeighbor" attribute
            agent.surroundingNeighbor = self.get_current_agent_nei_V2(agent, agentRefer_dict)

            # all_other_posdiff = []
            # pair_posdiff = []
            # for other_agentIdx, other_agent in self.all_agents.items():
            #     if other_agentIdx == agentIdx:
            #         continue
            #     all_other_posdiff.append(other_agent.pos[0]-agent.pos[0])
            #     all_other_posdiff.append(other_agent.pos[1]-agent.pos[1])
            #     pair_posdiff.append(other_agent.pos-agent.pos)
            # all_other_posdiff = np.array(all_other_posdiff)


            agent_own = np.array(
                [agent.pos[0], agent.pos[1], agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1],
                 agent.vel[0], agent.vel[1], agent.acc[0], agent.acc[1]])

            # agent_own = np.concatenate((agent_own, all_other_posdiff))
            # populate normalized agent_own
            # norm_agent_own = []
            norm_pos = self.normalizer.nmlz_pos([agent.pos[0], agent.pos[1]])

            norm_G_diff = self.normalizer.nmlz_pos_diff(
                [agent.goal[0][0] - agent.pos[0], agent.goal[0][1] - agent.pos[1]])

            # norm_other_diff = tuple(self.normalizer.nmlz_pos_diff(pos_diff_pair) for pos_diff_pair in pair_posdiff)

            norm_vel = self.normalizer.nmlz_vel([agent.vel[0], agent.vel[1]])
            norm_acc = self.normalizer.nmlz_acc([agent.acc[0], agent.acc[1]])
            norm_agent_own = np.array(list(norm_pos + norm_G_diff + norm_vel+norm_acc))

            other_agents = []
            combined_other_agent = []
            norm_combined_other_agent = []
            norm_other_agents = []
            if len(agent.surroundingNeighbor) > 0:  # meaning there is surrounding neighbors around the current agent
                for other_agentIdx, other_agent in agent.surroundingNeighbor.items():
                    if other_agentIdx != agent_idx:
                        surround_agent = np.array([[other_agent[0] - agent.pos[0],
                                                   other_agent[1] - agent.pos[1],
                                                   other_agent[-2] - other_agent[0],
                                                   other_agent[-1] - other_agent[1],
                                                   other_agent[2], other_agent[3]]])

                        norm_pos_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1]])

                        norm_G_diff = self.normalizer.nmlz_pos_diff(
                            [other_agent[-2] - other_agent[0], other_agent[-1] - other_agent[1]])

                        norm_vel = self.normalizer.nmlz_vel([other_agent[2], other_agent[3]])
                        norm_surround_agent = np.array([list(norm_pos_diff + norm_G_diff + norm_vel)])

                        combined_other_agent.append([other_agent[0] - agent.pos[0], other_agent[1] - agent.pos[1], other_agent[2], other_agent[3]])  # position difference with host drone + own velocity, so is nx4
                        norm_combined_other_agent.append(list(norm_pos_diff+norm_vel))

                        other_agents.append(surround_agent)
                        norm_other_agents.append(norm_surround_agent)
                combined_other_agent = list(itertools.chain.from_iterable(combined_other_agent))
                norm_combined_other_agent = list(itertools.chain.from_iterable(norm_combined_other_agent))
                overall_state_p3.append(other_agents)
                norm_overall_state_p3.append(norm_other_agents)
            else:
                overall_state_p3.append([np.zeros((1, 6))])
                norm_overall_state_p3.append([np.zeros((1, 6))])

            # overall_state_p1.append(agent_own)
            fullyObs_p1.append(np.concatenate((agent_own, np.array(combined_other_agent))))
            overall_state_p2.append(agent.observableSpace)

            # norm_overall_state_p1.append(norm_agent_own)
            norm_fullyObs_p1.append(np.concatenate((norm_agent_own, np.array(norm_combined_other_agent))))
            norm_overall_state_p2.append(agent.observableSpace)

        overall.append(fullyObs_p1)
        overall.append(overall_state_p2)
        overall.append(overall_state_p3)
        norm_overall.append(norm_fullyObs_p1)
        norm_overall.append(norm_overall_state_p2)
        norm_overall.append(norm_overall_state_p3)
        return overall, norm_overall

    def current_observable_space(self, cur_agent):
        occupied_building_val = 10
        occupied_drone_val = 50
        non_occupied_val = 1
        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        polygonSet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        self_circle_inter = []
        worldGrid_polyCombine = []
        # self.world_map_2D_polyList[0][0] is all grid=1, or list of occupied grids
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        world_STRtree = STRtree(worldGrid_polyCombine[0])
        detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
        self_circle = Point(t_x, t_y).buffer(cur_agent.protectiveBound, cap_style='round')
        possible_matches = world_STRtree.query(detection_circle)

        for poly in world_STRtree.geometries.take(possible_matches):
            if detection_circle.intersects(poly):
                polygonSet.append(poly)
            if self_circle.intersects(poly):
                self_circle_inter.append(poly)

        # all detectable grids (not arranged)
        no_sorted_polySet = polygonSet

        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(polygonSet)
        for poly in sorted_polySet:
            if self_circle.intersects(poly):
                currentObservableState.append(occupied_drone_val)
                continue
            if poly in self.world_map_2D_polyList[0][0]:
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we are using arranged polygonSet and 1D array
        return currentObservableState

    def get_current_agent_nei_V2(self, cur_agent, agentRefer_dict):
        # identify neighbors (use distance)
        point_to_search = cur_agent.pos
        # subtract a small value to exclude point at exactly "search_distance"
        # search_distance = (cur_agent.detectionRange / 2) + cur_agent.protectiveBound
        search_distance = (300000 / 2) + cur_agent.protectiveBound

        for agent_pos, agent_idx_string in agentRefer_dict.items():
            if cur_agent.agent_name == agent_idx_string:
                continue
            if np.linalg.norm(np.array(point_to_search) - np.array(agent_pos))<=search_distance:
                other_agent_idx = int(re.search(r'\d+(\.\d+)?', agent_idx_string).group())
                cur_agent.surroundingNeighbor[other_agent_idx] = np.array([self.all_agents[other_agent_idx].pos[0],
                                                                           self.all_agents[other_agent_idx].pos[1],
                                                                           self.all_agents[other_agent_idx].vel[0],
                                                                           self.all_agents[other_agent_idx].vel[1],
                                                                           self.all_agents[other_agent_idx].goal[0][0],
                                                                           self.all_agents[other_agent_idx].goal[0][1]])

        return cur_agent.surroundingNeighbor

    def current_observable_space_fixedLength(self, cur_agent):
        occupied_building_val = -10
        occupied_drone_val = 50
        non_occupied_val = 1
        host_polygon = None
        worldGrid_polyCombine = []
        # self.world_map_2D_polyList[0][0] is all grid=1, or list of occupied grids
        worldGrid_polyCombine.append(self.world_map_2D_polyList[0][0] + self.world_map_2D_polyList[0][1])
        world_STRtree = STRtree(worldGrid_polyCombine[0])
        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        curPt = Point(t_x, t_y)
        no_sorted_polySet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        possible_poly_idx = world_STRtree.query(curPt)
        containList = []
        for poly in world_STRtree.geometries.take(possible_poly_idx).tolist():
            if poly.contains(curPt):
                containList.append(poly)
        if len(containList)!=1:
            containList = [None]  # we clear the list.
            detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
            possible_matches = world_STRtree.query(detection_circle)
            highest_overlap_area = 0
            for poly in world_STRtree.geometries.take(possible_matches):
                intersect = poly.intersection(detection_circle)
                intersection_area = intersect.area
                if intersection_area > highest_overlap_area:
                    if intersection_area == highest_overlap_area:
                        print("There are two surrounding polygon has the same overlap area on the host drone's protective area please debug!")
                    highest_overlap_area = intersection_area
                    containList[0] = poly
            if len(containList) == 0:
                print("none intersection detected please debug!")

        larger_polygon = scale(containList[0], xfact=3, yfact=3,
                               origin='center')  # when xfact and yfact both equals to 3, my bigger polygon is of square shape, edge length is 3 times compared to the input square polygon

        poly_idx = world_STRtree.query(larger_polygon)
        for poly in world_STRtree.geometries.take(poly_idx).tolist():
            if larger_polygon.contains(poly):
                no_sorted_polySet.append(poly)


        # for poly in world_STRtree.geometries.take(possible_matches):
        #     if detection_circle.intersects(poly):
        #         polygonSet.append(poly)
        #     if self_circle.intersects(poly):
        #         self_circle_inter.append(poly)


        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(no_sorted_polySet)
        for poly in sorted_polySet:
            if poly.equals(containList[0]):
                currentObservableState.append(occupied_drone_val)
            if poly in self.world_map_2D_polyList[0][0]:  # if polygon is an element of occupied polygon
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we are using arranged polygonSet and 1D array
        return np.array(currentObservableState)

    def current_observable_space_fixedLength_fromv2_flow(self, cur_agent):
        # This function should output an array of length 9. In case that, when agent is at edge of the 2D map, we just patch with 0.
        occupied_building_val = -10
        occupied_drone_val = 50
        non_occupied_val = 1
        max_out_length = 9
        host_polygon = None

        currentObservableState = []
        cur_hostPos_from_input = np.array([cur_agent.pos[0], cur_agent.pos[1]])
        t_x = cur_hostPos_from_input[0]
        t_y = cur_hostPos_from_input[1]
        curPt = Point(t_x, t_y)
        no_sorted_polySet = []  # this polygonSet including the polygon that intersect with the "self_circle"
        possible_poly_idx = self.world_STRtree.query(curPt)
        containList = []
        for poly in self.world_STRtree.geometries.take(possible_poly_idx).tolist():
            if poly.contains(curPt):
                containList.append(poly)
        if len(containList)!=1:
            containList = [None]  # we clear the list.
            detection_circle = Point(t_x, t_y).buffer(cur_agent.detectionRange / 2, cap_style='round')
            possible_matches = self.world_STRtree.query(detection_circle)
            highest_overlap_area = 0
            for poly in self.world_STRtree.geometries.take(possible_matches):
                intersect = poly.intersection(detection_circle)
                intersection_area = intersect.area
                if intersection_area > highest_overlap_area:
                    if intersection_area == highest_overlap_area:
                        print("There are two surrounding polygon has the same overlap area on the host drone's protective area please debug!")
                    highest_overlap_area = intersection_area
                    containList[0] = poly
            if len(containList) == 0:
                print("none intersection detected please debug!")
        if containList[0] == None:
            print("debug")
        larger_polygon = scale(containList[0], xfact=3, yfact=3,
                               origin='center')  # when xfact and yfact both equals to 3, my bigger polygon is of square shape, edge length is 3 times compared to the input square polygon

        poly_idx = self.world_STRtree.query(larger_polygon)
        for poly in self.world_STRtree.geometries.take(poly_idx).tolist():
            if larger_polygon.contains(poly):
                no_sorted_polySet.append(poly)


        # for poly in world_STRtree.geometries.take(possible_matches):
        #     if detection_circle.intersects(poly):
        #         polygonSet.append(poly)
        #     if self_circle.intersects(poly):
        #         self_circle_inter.append(poly)


        # all detectable grids (arranged)
        sorted_polySet = sort_polygons(no_sorted_polySet)
        for poly in sorted_polySet:
            if poly.equals(containList[0]):
                currentObservableState.append(occupied_drone_val)
            elif poly in self.world_map_2D_polyList[0][0]:  # check if polygon is an element of occupied polygon
                currentObservableState.append(occupied_building_val)
            else:
                currentObservableState.append(non_occupied_val)
        # currently we are using arranged polygonSet and 1D array
        if len(currentObservableState) < max_out_length:
            currentObservableState.extend([0] * (max_out_length - len(currentObservableState)))
        if not currentObservableState:
            print("check")
        return np.array(currentObservableState)

    def get_actions_noCR(self):
        outActions = {}
        noCR = 1
        vel = [None] * 2
        for agent_idx, agent in self.all_agents.items():
            # heading in rad must be goal_pos-intruder_pos, and y2-y1, x2-x1
            agent.heading = math.atan2(agent.goal[0][1] - agent.pos[1],
                                       agent.goal[0][0] - agent.pos[0])
            vel[0] = (agent.maxSpeed/2) * math.cos(agent.heading)
            vel[1] = (agent.maxSpeed/2) * math.sin(agent.heading)
            outActions[agent_idx] = np.array([vel[0], vel[1]])
        return outActions

    def get_actions_NN(self, combine_state, eps):  # decentralized execution, only actor net is used here
        outActions = {}
        for agent_idx, agent in self.all_agents.items():
            # obtain the observation for each individual actor
            individual_obs = extract_individual_obs(combine_state, agent_idx)
            # chosen_action = agent.choose_actions(individual_obs)  # when deals with three different section of inputs
            input_tensor = T.tensor(individual_obs.reshape(1, -1), dtype=T.float).to(agent.actorNet.device)
            input_tensor_d = input_tensor.detach()
            chosen_action = agent.actorNet(input_tensor_d)
            # squeeze(0) is to remove the batch information
            # then convert to numpy
            # chosen_action = chosen_action.squeeze(0).detach().numpy()
            chosen_action = chosen_action + T.tensor(self.OU_noise.noise()) # add noise for exploration first, before clamp
            # clip the action
            chosen_action = T.clamp(chosen_action, -1, 1)
            # for ea_idx, ea_a in enumerate(chosen_action):
            #     if ea_a < -1:
            #         chosen_action[ea_idx] = -1
            #     if ea_a > 1:
            #         chosen_action[ea_idx] = 1

            # update current sigma used for the exploration
            self.OU_noise.sigma = self.OU_noise.largest_sigma * eps + (1 - eps) * self.OU_noise.smallest_sigma
            outActions[agent_idx] = np.squeeze(chosen_action.data.cpu().numpy())  # load to output dict

        return outActions

    def get_step_reward_5_v3(self, current_ts, step_reward_record):  # this is for individual drones, current_ts = current time step
        reward, done = [], []
        agent_to_remove = []
        one_step_reward = []
        check_goal = [False] * len(self.all_agents)
        reward_record_idx = 0  # this is used as a list index, increase with for loop. No need go with agent index, this index is also shared by done checking
        # crash_penalty = -200
        crash_penalty = -300
        reach_target = 300
        potential_conflict_count = 0
        final_goal_toadd = 0
        fixed_domino_reward = 1
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])

        for drone_idx, drone_obj in self.all_agents.items():
            one_agent_reward_record = []
            # re-initialize these two list for individual agents at each time step,this is to ensure collision
            # condition is reset for each agent at every time step
            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            dist_toHost = []
            # we assume the maximum potential conflict the current drone could have at each time step is equals
            # to the total number of its neighbour at each time step
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(self.all_agents[drone_idx].pos)
            if isinstance(self.all_agents[drone_idx].removed_goal, np.ndarray):
                host_refline = LineString([self.all_agents[drone_idx].removed_goal, self.all_agents[drone_idx].goal[0]])
            else:
                host_refline = LineString([self.all_agents[drone_idx].ini_pos, self.all_agents[drone_idx].goal[0]])
            cross_track_deviation = curPoint.distance(host_refline)  # deviation from the reference line, cross track error

            host_pass_line = LineString([self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos])
            host_passed_volume = host_pass_line.buffer(self.all_agents[drone_idx].protectiveBound, cap_style='round')

            # neigh_keys is the drone_idx for current neighbors
            # loop through neighbors from previous time step
            for neigh_keys in self.all_agents[drone_idx].pre_surroundingNeighbor:
                # compute potential conflicts before and after the action for the current drone with its neighbours
                try:
                    one_pc_before = compute_potential_conflict(drone_obj.pre_pos, drone_obj.pre_vel,
                                                           drone_obj.protectiveBound, self.all_agents[neigh_keys].pre_pos,
                                                           self.all_agents[neigh_keys].pre_vel,
                                                           self.all_agents[neigh_keys].protectiveBound, neigh_keys,
                                                           current_ts)
                except:
                    print("pause")
                if len(one_pc_before) > 0:
                    pc_before.append(one_pc_before)
            # loop through neighbors from current time step
            for neigh_keys in self.all_agents[drone_idx].surroundingNeighbor:
                # compute potential conflicts before and after the action for the current drone with its neighbours
                one_pc_after = compute_potential_conflict(drone_obj.pos, drone_obj.vel,
                                                      drone_obj.protectiveBound, self.all_agents[neigh_keys].pos,
                                                      self.all_agents[neigh_keys].vel,
                                                      self.all_agents[neigh_keys].protectiveBound, neigh_keys,
                                                      current_ts)
                if len(one_pc_after) > 0:
                    pc_after.append(one_pc_after)

                # get distance from host to all the surrounding vehicles
                diff_dist_vec = drone_obj.pos - self.all_agents[neigh_keys].pos  # host pos vector - intruder pos vector
                dist_toHost.append(np.linalg.norm(diff_dist_vec))
                # check whether the current drone has collides with any surrounding neighbors due to current action
                neigh_pass_line = LineString([self.all_agents[neigh_keys].pre_pos, self.all_agents[neigh_keys].pos])
                neigh_passed_volume = neigh_pass_line.buffer(self.all_agents[neigh_keys].protectiveBound,
                                                             cap_style='round')
                if host_passed_volume.intersects(neigh_passed_volume):
                    print("drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys, current_ts))
                    collision_drones.append(neigh_keys)



            # if pc_max_after == 0:  # upper bound in terms of value case for dominoTerm
            #     dominoTerm = fixed_domino_reward
            # elif pc_max_before == 0:  # lower bound in terms of value case for dominoTerm
            #     dominoTerm = -1
            # elif (len(pc_after)/pc_max_after) == 0:  # check if denominator of the dominoTerm equals to 0
            #     # if denominator equals to 0, meaning initial velocity is 0,
            #     # so is like initial condition, then we can just assign this dominoTerm as 0.
            #     dominoTerm = 0
            # else:
            #     dominoTerm = ((len(pc_before)/pc_max_before) -
            #                   (len(pc_after)/pc_max_after)) / (len(pc_after)/pc_max_after)

            # if len(pc_after) == 0:
            #     dominoTerm = fixed_domino_reward
            # else:
            #     dominoTerm = (len(pc_before) - len(pc_after)) / len(pc_after)
            dominoTerm = []  # reset for every new decision making drone
            for neigh_keys, t_cpa, d_cpa in pc_after:
                # td_cpa[1] is d_cpa, td_cpa[0] is t_cpa
                # dominoValue = ((2*drone_obj.protectiveBound)/math.exp(td_cpa[1])) * (math.exp(-td_cpa[0]))
                dominoValue = ((5/math.exp((d_cpa-5)/3))+1) * (1-(1/(5**(3-t_cpa))))
                dominoTerm.append(dominoValue)
            dominoTerm_sum = -sum(dominoTerm)  # use -ve to indicate a penalty

            # check whether current actions leads to a collision with any buildings in the airspace
            allBuildingSTR = STRtree(self.world_map_2D_polyList[0][0])
            possiblePoly = allBuildingSTR.query(host_passed_volume)
            for element in possiblePoly:
                if allBuildingSTR.geometries.take(element).intersection(host_passed_volume):
                    collide_building = 1
                    print("drone_{} crash into building when moving from {} to {} at time step {}".format(drone_idx, self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos, current_ts))
                    break

            tar_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            # when there is no intersection between two geometries, "RuntimeWarning" will appear
            # RuntimeWarning is, "invalid value encountered in intersection"
            goal_cur_intru_intersect = host_passed_volume.intersection(tar_circle)

            # if len(pc_after) > 0:
            #     print("debug")
            #     matplotlib.use('TkAgg')
            #     plt.ion()
            #     # Create figure and axis objects
            #     fig, ax = plt.subplots()
            #     ax.set_xlim(self.bound[0], self.bound[1])  # Set x-axis limits from 0 to 5
            #     ax.set_ylim(self.bound[2], self.bound[3])  # Set y-axis limits from 0 to 5
            #     for one_poly in self.world_map_2D_polyList[0][0]:
            #         one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
            #         ax.add_patch(one_poly_mat)
            #     # plot current host drone
            #     ax.text(drone_obj.pos[0], drone_obj.pos[1], drone_obj.agent_name)
            #     self_circle = Point(drone_obj.pos[0],
            #                         drone_obj.pos[1]).buffer(drone_obj.protectiveBound, cap_style='round')
            #     ax.arrow(drone_obj.pre_pos[0], drone_obj.pre_pos[1], drone_obj.pos[0]-drone_obj.pre_pos[0], drone_obj.pos[1]-drone_obj.pre_pos[1], head_width=0.05, head_length=0.1, fc='r', ec='k')
            #     grid_mat_Scir = shapelypoly_to_matpoly(self_circle, inFill=False, Edgecolor='r')
            #
            #     ax.add_patch(grid_mat_Scir)
            #     # for drones that are possible having conflict
            #     for neigh_key in pc_after.keys():
            #         ax.text(self.all_agents[neigh_key].pos[0], self.all_agents[neigh_key].pos[1],
            #                 self.all_agents[neigh_key].agent_name)
            #         possible_conflict_circle = Point(self.all_agents[neigh_key].pos[0],
            #                                          self.all_agents[neigh_key].pos[1]).buffer(
            #             self.all_agents[neigh_key].protectiveBound, cap_style='round')
            #         ax.arrow(self.all_agents[neigh_key].pre_pos[0], self.all_agents[neigh_key].pre_pos[1], self.all_agents[neigh_key].pos[0]-self.all_agents[neigh_key].pre_pos[0], self.all_agents[neigh_key].pos[1]-self.all_agents[neigh_key].pre_pos[1],
            #                  head_width=0.05, head_length=0.1, fc='k', ec='k')
            #         grid_mat_Scir = shapelypoly_to_matpoly(possible_conflict_circle, inFill=False, Edgecolor='k')
            #         ax.add_patch(grid_mat_Scir)
            #     plt.show()
            #     print("done show")

            # ------------- pre-processed condition for a normal step -----------------
            # crossCoefficient = 0.1
            crossCoefficient = 1
            # goalCoefficient = 6
            goalCoefficient = 8
            dominoCoefficient = 1
            # cross track error term
            # cross_track_error = (20 / ((cross_track_deviation * cross_track_deviation) / 200 + 1)) - 3.5  # original
            cross_track_error = (math.e ** (5 - cross_track_deviation / 7) / 5) - 0.5  # original on 24_07
            # cross_track_error = (5 * math.e ** ((5 - cross_track_deviation) / 7)) - 1  # cross_track_deviation>16.266 -> <0, cross_track_deviation=0 -> 9.2
            # Distance between drone and its goal for two consecutive time step
            before_dist_hg = np.linalg.norm(drone_obj.pre_pos - drone_obj.goal[0])
            after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[0])  # distance to goal after action
            delta_hg = goalCoefficient * (before_dist_hg - after_dist_hg)
            if len(dist_toHost) == 0:  # meaning there is no neighbouring drone goes into host drone's detection range
                slowChanging_dist_penalty_others = 0
            else:
                # sort dist_toHost()
                # nearest
                dist_to_host_minimum = sorted(dist_toHost)[0]
                # slowChanging_dist_penalty_others = 1 * (-10 * math.exp((5 - dist_to_host_minimum) / 2))
                slowChanging_dist_penalty_others = -(math.e ** (5 - dist_to_host_minimum / 7) / 5)  # from -14.5 to 0

            # a small penalty for discourage the agent to stay in one single spot
            if (before_dist_hg - after_dist_hg) <= 2:
                small_step_penalty = 50
            else:
                small_step_penalty = 0
            alive_penalty = -60
            # -------------end of pre-processed condition for a normal step -----------------

            # exceed bound or crash into buildings or crash with other neighbors
            if collide_building == 1:
                reward.append(np.array(crash_penalty))
                done.append(True)
                # done.append(False)
            elif len(collision_drones) > 0:
                reward.append(np.array(crash_penalty))
                done.append(True)

            # exceed bound condition, don't use current point, use current circle or else will have condition that
            elif x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(host_passed_volume):
                print("drone_{} has crash into boundary at time step {}".format(drone_idx, current_ts))
                reward.append(np.array(crash_penalty))
                done.append(True)
                # done.append(False)

            elif not goal_cur_intru_intersect.is_empty:  # reached goal?

                if len(drone_obj.goal) > 1:  # meaning the current agent has more than one target/goal
                    print("drone_{} has reached its way point at time step {}".format(drone_idx, current_ts))
                    drone_obj.reach_target = False  # reset this flag
                    drone_obj.removed_goal = drone_obj.goal.pop(0)
                    # normal_step_rw = crossCoefficient*cross_track_error + slowChanging_dist_penalty_others + alive_penalty
                    # normal_step_rw = crossCoefficient * cross_track_error + delta_hg + alive_penalty
                    # normal_step_rw = crossCoefficient*cross_track_error + delta_hg + alive_penalty + slowChanging_dist_penalty_others
                    normal_step_rw = crossCoefficient*cross_track_error + delta_hg + alive_penalty + dominoCoefficient*dominoTerm_sum
                    reward.append(np.array(normal_step_rw))
                    # below two lines are used to debug
                    # reward.append(np.array(reach_target))
                    # agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                else:
                    check_goal[reward_record_idx] = True  # drone_obj.reach_target = True, and "check_goal" list also don't track agent's index
                    # normal_step_rw = crossCoefficient * cross_track_error + delta_hg
                    reward.append(np.array(reach_target))
                    # reward.append(np.array(normal_step_rw))
                    print("drone_{} has reached its final goal at time step {}".format(drone_idx, current_ts))
                    agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                if all(check_goal):
                    done.append(True)
                else:
                    done.append(False)
                # done.append(False)
            else:  # a normal step taken
                step_reward =crossCoefficient*cross_track_error + delta_hg + alive_penalty + dominoCoefficient*dominoTerm_sum

                # we remove the above termination condition
                done.append(False)

                step_reward = np.array(step_reward)
                reward.append(step_reward)
                # for debug, record the reward
                one_step_reward = [crossCoefficient*cross_track_error, delta_hg, alive_penalty, dominoCoefficient*dominoTerm_sum]
                step_reward_record[reward_record_idx] = one_step_reward
            reward_record_idx = reward_record_idx + 1

        # # we remove the reached agent here
        # for remove_idx in agent_to_remove:  # removed_idx is the key for the removed agent
        #     removed_value = self.all_agents.pop(remove_idx)
        #     # check all other agent's current surrounding neighbours, remove the current agent from their surroundingNeighbors
        #     for drone_idx, drone_obj in self.all_agents.items():  # this for loop already exclude the removed agent
        #         if remove_idx in drone_obj.surroundingNeighbor:
        #             del drone_obj.surroundingNeighbor[remove_idx]
        #         if remove_idx in drone_obj.pre_surroundingNeighbor:  # remember to remove the previous surrounding neighbours as well.
        #             del drone_obj.pre_surroundingNeighbor[remove_idx]
        #
        # # immediately add a dummy agent to recover the removed agent, we need to prevent the case where all agents reaches their destination, then the environmnet become empty
        # num_lack = len(agent_to_remove)
        # agent_filled = []  # contains the key of the newly added agents
        # for i in range(num_lack):  # the num_lack equals the number of times this for loop will execute
        #     # when both drones / all drones reaches the goal at same ts. Will have error. self.all_agents.keys() will become zeros. Debug this problem.
        #     agent = deepcopy(self.dummy_agent)
        #     if len(self.all_agents) == 0:
        #         current_max = max(agent_to_remove)  # all agents have reaches their destination at same time
        #     else:
        #         try:
        #             current_max = max(max(list(self.all_agents.keys())), max(agent_to_remove))  # new agent_idx to be added can the maximum of the remove agent or the maximum of the remining agent, we take whichever is higher.
        #         except:
        #             print("pause")
        #     agent.agent_name = 'agent_%s' % str(current_max + 1)
        #     self.all_agents[current_max + 1] = agent
        #     agent_filled.append(current_max + 1)
        # return reward, done, check_goal, step_reward_record, agent_filled
        return reward, done, check_goal, step_reward_record

    def ss_reward(self, current_ts, step_reward_record, eps_status_holder):
        reward, done = [], []
        agent_to_remove = []
        one_step_reward = []
        check_goal = [False] * len(self.all_agents)
        reward_record_idx = 0  # this is used as a list index, increase with for loop. No need go with agent index, this index is also shared by done checking
        # crash_penalty = -200
        crash_penalty_wall = 5
        crash_penalty_drone = 1
        reach_target = 1
        potential_conflict_count = 0
        final_goal_toadd = 0
        fixed_domino_reward = 1
        x_left_bound = LineString([(self.bound[0], -9999), (self.bound[0], 9999)])
        x_right_bound = LineString([(self.bound[1], -9999), (self.bound[1], 9999)])
        y_bottom_bound = LineString([(-9999, self.bound[2]), (9999, self.bound[2])])
        y_top_bound = LineString([(-9999, self.bound[3]), (9999, self.bound[3])])

        for drone_idx, drone_obj in self.all_agents.items():
            drone_status_record = []
            one_agent_reward_record = []
            # re-initialize these two list for individual agents at each time step,this is to ensure collision
            # condition is reset for each agent at every time step
            collision_drones = []
            collide_building = 0
            pc_before, pc_after = [], []
            dist_toHost = []
            # we assume the maximum potential conflict the current drone could have at each time step is equals
            # to the total number of its neighbour at each time step
            pc_max_before = len(drone_obj.pre_surroundingNeighbor)
            pc_max_after = len(drone_obj.surroundingNeighbor)

            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(self.all_agents[drone_idx].pos)
            if isinstance(self.all_agents[drone_idx].removed_goal, np.ndarray):
                host_refline = LineString([self.all_agents[drone_idx].removed_goal, self.all_agents[drone_idx].goal[0]])
            else:
                host_refline = LineString([self.all_agents[drone_idx].ini_pos, self.all_agents[drone_idx].goal[0]])
            cross_track_deviation = curPoint.distance(host_refline)  # deviation from the reference line, cross track error
            host_pass_line = LineString([self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos])
            host_passed_volume = host_pass_line.buffer(self.all_agents[drone_idx].protectiveBound, cap_style='round')
            host_current_circle = Point(self.all_agents[drone_idx].pos[0], self.all_agents[drone_idx].pos[1]).buffer(self.all_agents[drone_idx].protectiveBound)

            # loop through neighbors from current time step
            for neigh_keys in self.all_agents[drone_idx].surroundingNeighbor:
                # get distance from host to all the surrounding vehicles
                diff_dist_vec = drone_obj.pos - self.all_agents[neigh_keys].pos  # host pos vector - intruder pos vector
                if np.linalg.norm(diff_dist_vec) <= drone_obj.protectiveBound * 2:
                    # print("drone_{} collide with drone_{} at time step {}".format(drone_idx, neigh_keys, current_ts))
                    collision_drones.append(neigh_keys)

            # check whether current actions leads to a collision with any buildings in the airspace
            possiblePoly = self.allbuildingSTR.query(host_current_circle)
            for element in possiblePoly:
                if self.allbuildingSTR.geometries.take(element).intersection(host_current_circle):
                    collide_building = 1
                    print("drone_{} crash into building when moving from {} to {} at time step {}".format(drone_idx, self.all_agents[drone_idx].pre_pos, self.all_agents[drone_idx].pos, current_ts))
                    break

            # tar_circle = Point(self.all_agents[drone_idx].goal[0]).buffer(1, cap_style='round')
            tar_circle = Point(self.all_agents[drone_idx].goal[-1]).buffer(1, cap_style='round')  # set to [-1] so there are no more reference path
            # when there is no intersection between two geometries, "RuntimeWarning" will appear
            # RuntimeWarning is, "invalid value encountered in intersection"
            goal_cur_intru_intersect = host_current_circle.intersection(tar_circle)

            # ------------- pre-processed condition for a normal step -----------------
            rew = 3
            # rew = 0
            # after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[-1])  # distance to goal after action
            x_norm, y_norm = self.normalizer.nmlz_pos(drone_obj.pos)
            tx_norm, ty_norm = self.normalizer.nmlz_pos(drone_obj.goal[-1])
            after_dist_hg = math.sqrt(((x_norm-tx_norm)**2 + (y_norm-ty_norm)**2))
            alive_penalty = -60
            # -------------end of pre-processed condition for a normal step -----------------

            # Always check the boundary as the 1st condition, or else will encounter error where the agent crash into wall but also exceed the bound, but crash into wall did not stop the episode. So, we must put the check boundary condition 1st, so that episode can terminate in time and does not leads to exceed boundary with error in no polygon found.
            # exceed bound condition, don't use current point, use current circle or else will have condition that
            # must use "host_passed_volume", or else, we unable to confirm whether the host's circle is at left or right of the boundary lines
            if x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(host_passed_volume):
                print("drone_{} has crash into boundary at time step {}".format(drone_idx, current_ts))
                rew = rew - crash_penalty_wall
                done.append(True)
                # done.append(False)
                reward.append(np.array(rew))
            # crash into buildings or crash with other neighbors
            elif collide_building == 1:
                # done.append(True)
                done.append(True)
                rew = rew - crash_penalty_wall
                reward.append(np.array(rew))
            # elif len(collision_drones) > 0:
            #     # done.append(True)
            #     done.append(False)
            #     rew = rew - crash_penalty_drone
            #     reward.append(np.array(rew))
            elif not goal_cur_intru_intersect.is_empty:  # reached goal?
                # --------------- with way point -----------------------
                # if len(drone_obj.goal) > 1:  # meaning the current agent has more than one target/goal
                #     print("drone_{} has reached its way point at time step {}".format(drone_idx, current_ts))
                #     drone_obj.reach_target = False  # reset this flag
                #     drone_obj.removed_goal = drone_obj.goal.pop(0)
                #     after_dist_hg = np.linalg.norm(drone_obj.pos - drone_obj.goal[0])  # distance to next goal
                #     rew = rew - after_dist_hg
                #     reward.append(np.array(rew))
                #     # below two lines are used to debug
                #     # reward.append(np.array(reach_target))
                #     # agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                # else:
                #     check_goal[reward_record_idx] = True  # drone_obj.reach_target = True, and "check_goal" list also don't track agent's index
                #     # normal_step_rw = crossCoefficient * cross_track_error + delta_hg
                #     # reward.append(np.array(normal_step_rw))
                #     print("drone_{} has reached its final goal at time step {}".format(drone_idx, current_ts))
                #     agent_to_remove.append(drone_idx)  # NOTE: drone_idx is the key value.
                #     rew = rew + reach_target
                #     reward.append(np.array(rew))
                # --------------- end of with way point -----------------------
                # without wap point
                rew = rew + reach_target
                reward.append(np.array(rew))
                done.append(False)
            else:  # a normal step taken
                rew = rew - after_dist_hg
                # we remove the above termination condition
                done.append(False)
                step_reward = np.array(rew)
                reward.append(step_reward)
                # for debug, record the reward
                # one_step_reward = [crossCoefficient*cross_track_error, delta_hg, alive_penalty, dominoCoefficient*dominoTerm_sum]
                # step_reward_record[reward_record_idx] = one_step_reward
                step_reward_record[reward_record_idx] = [step_reward]
            reward_record_idx = reward_record_idx + 1
            actual_after_dist_hg = math.sqrt(((drone_obj.pos[0] - drone_obj.goal[-1][0]) ** 2 + (drone_obj.pos[1] - drone_obj.goal[-1][1]) ** 2))
            # print("current drone {} distance to goal is {}, current reward is {}".format(drone_idx, actual_after_dist_hg, reward[-1]))

            # record status of each step.
            eps_status_holder = self.display_one_eps_status(eps_status_holder, drone_idx, actual_after_dist_hg, reward[-1])
            # overall_status_record[2].append()  # 3rd is accumulated reward till that step for each agent

        # check if length of reward does not equals to 3
        if len(reward) != 3:
            print("check")

        return reward, done, check_goal, step_reward_record, eps_status_holder

    def display_one_eps_status(self, status_holder, drone_idx, cur_step_dist, cur_step_reward):
        if status_holder[drone_idx] == None:
            status_holder[drone_idx] = [[cur_step_dist, cur_step_reward]]
        else:
            status_holder[drone_idx].append([cur_step_dist, cur_step_reward])
        return status_holder


    def step(self, actions, current_ts):
        next_combine_state = []
        agentCoorKD_list_update = []
        agentRefer_dict = {}  # A dictionary to use agent's current pos as key, their agent name (idx) as value
        # we use 4 here, because the time-step for the simulation is 0.5 sec.
        # hence, 4 here is equivalent to the acceleration of 2m/s^2
        coe_a = 4  # coe_a is the coefficient of action is 4 because our time step is 0.5 sec
        # based on the input stack of actions we propagate all agents forward
        # for drone_idx, drone_act in actions.items():  # this is for evaluation with default action
        count = 1
        for drone_idx_obj, drone_act in zip(self.all_agents.items(), actions):

            drone_idx = drone_idx_obj[0]
            drone_obj = drone_idx_obj[1]
            # let current neighbor become neighbor recorded before action
            start_deepcopy_time = time.time()
            self.all_agents[drone_idx].pre_surroundingNeighbor = deepcopy(self.all_agents[drone_idx].surroundingNeighbor)
            # let current position become position is the previous state, so that new position can be updated
            self.all_agents[drone_idx].pre_pos = deepcopy(self.all_agents[drone_idx].pos)
            # fill previous velocities
            self.all_agents[drone_idx].pre_vel = deepcopy(self.all_agents[drone_idx].vel)
                        # fill previous acceleration
            self.all_agents[drone_idx].pre_acc = deepcopy(self.all_agents[drone_idx].acc)
            # print("deepcopy done, time used {} nanosecond".format((time.time()-start_deepcopy_time)*1000000000))
            print("deepcopy done, time used {} milliseconds".format((time.time()-start_deepcopy_time)*1000))

            # --------------- speed & heading angle control for training -------------------- #
            # raw_speed, raw_heading_angle = drone_act[0], drone_act[1]
            # speed = ((raw_speed + 1) / 2) * self.all_agents[drone_idx].maxSpeed  # map from -1 to 1 to 0 to maxSpd of the agent
            # heading_angle = raw_heading_angle * math.pi  # ensure the heading angle is between -pi to pi.
            # delta_x = speed * math.cos(heading_angle) * self.time_step
            # delta_y = speed * math.sin(heading_angle) * self.time_step
            # -------------- end of speed & heading angle control ---------------------#


            # ----------------- acceleration in x and acceleration in y state transition control for training-------------------- #
            ax, ay = drone_act[0], drone_act[1]
            # map output action from NN to actual range
            # here is the action scaling part
            # ax = map_range(ax, coe_a)
            # ay = map_range(ay, coe_a)

            ax = ax * coe_a
            ay = ay * coe_a

            # check velocity limit
            curVelx = self.all_agents[drone_idx].vel[0] + ax * self.time_step
            curVely = self.all_agents[drone_idx].vel[1] + ay * self.time_step
            next_heading = math.atan2(curVely, curVelx)
            if np.linalg.norm([curVelx, curVely]) >= self.all_agents[drone_idx].maxSpeed:

                # update host velocity when chosen speed has exceeded the max speed
                hvx = self.all_agents[drone_idx].maxSpeed * math.cos(next_heading)
                hvy = self.all_agents[drone_idx].maxSpeed * math.sin(next_heading)
                self.all_agents[drone_idx].vel = np.array([hvx, hvy])
            else:
                # update host velocity when max speed is not exceeded
                self.all_agents[drone_idx].vel = np.array([curVelx, curVely])

            #print("At time step {} the drone_{}'s output speed is {}".format(current_ts, drone_idx, np.linalg.norm(self.all_agents[drone_idx].vel)))

            # update the drone's position based on the update velocities
            delta_x = self.all_agents[drone_idx].vel[0] * self.time_step
            delta_y = self.all_agents[drone_idx].vel[1] * self.time_step
            
            # update current acceleration of the agent after an action
            self.all_agents[drone_idx].acc = np.array([ax, ay])


            counterCheck_heading = math.atan2(delta_y, delta_x)
            if abs(next_heading - counterCheck_heading) > 1e-3 :
                print("debug, heading different")
            # ------------- end of acceleration in x and acceleration in y state transition control ---------------#

            self.all_agents[drone_idx].pos = np.array([self.all_agents[drone_idx].pos[0] + delta_x,
                                                       self.all_agents[drone_idx].pos[1] + delta_y])

            # cur_circle = Point(self.all_agents[drone_idx].pos[0],
            #                    self.all_agents[drone_idx].pos[1]).buffer(self.all_agents[drone_idx].protectiveBound,
            #                                                             cap_style='round')

            agentCoorKD_list_update.append(self.all_agents[drone_idx].pos)
            agentRefer_dict[(self.all_agents[drone_idx].pos[0],
                             self.all_agents[drone_idx].pos[1])] = self.all_agents[drone_idx].agent_name
            count = count + 1
        # self.cur_allAgentCoor_KD = KDTree(agentCoorKD_list_update)  # update all agent coordinate KDtree

        # print("for loop run {} times".format(count))

        # next_state, next_state_norm = self.cur_state_norm_state_fully_observable(agentRefer_dict)
        # start_acceleration_time = time.time()
        next_state, next_state_norm = self.cur_state_norm_state_v3(agentRefer_dict)
        # print("obtain_current_state, time used {} milliseconds".format(
        #     (time.time() - start_acceleration_time) * 1000))

        # # update current agent's observable space state
        # agent.observableSpace = self.current_observable_space(agent)
        # #cur_ObsGrids.append(agent.observableSpace)
        #
        # # update the "surroundingNeighbor" attribute
        # agent.surroundingNeighbor = self.get_current_agent_nei(agent, agentRefer_dict)
        # #actor_obs.append(agent.surroundingNeighbor)
        #
        # # populate overall state
        # # next_combine_state.append(np.array([agent_own, agent.observableSpace, agent.surroundingNeighbor], dtype=object))
        # next_combine_state.append(np.concatenate((agent_own, np.array(other_pos).flatten())))



        # matplotlib.use('TkAgg')
        # plt.ion()
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_aspect('equal')
        #
        # for t in range(max_t):
        #     ax.set_xlim([self.bound[0], self.bound[1]])
        #     ax.set_ylim([self.bound[2], self.bound[3]])
        #     previous_time = deepcopy(self.global_time)
        #     cur_time = previous_time + self.time_step
        #     step_volume_collection = []
        #     agent_moving_vol = {}
        #     for agent_idx, agent in self.all_agents.items():
        #         previous_pos = deepcopy(agent.pos)
        #         dx = actions[agent_idx][0] * self.time_step
        #         dy = actions[agent_idx][1] * self.time_step
        #         agent.pos[0] = agent.pos[0] + dx
        #         agent.pos[1] = agent.pos[1] + dy
        #         cur_agent_passLine = LineString([(previous_pos[0], previous_pos[1]),
        #                                          (agent.pos[0], agent.pos[1])])
        #         cur_agent_passed_volume = cur_agent_passLine.buffer(agent.protectiveBound, cap_style='round')
        #         agent_moving_vol[agent_idx] = cur_agent_passed_volume
        #         step_volume_collection.append(cur_agent_passed_volume)
        #
        #         #plt.text(previous_pos[0], previous_pos[1], "{}, t={}".format(agent.agent_name, previous_time))
        #         matp_cur_volume = shapelypoly_to_matpoly(cur_agent_passed_volume, True, 'red', 'b')
        #         ax.add_patch(matp_cur_volume)
        #         plt.text(agent.pos[0], agent.pos[1], "{}".format(agent.agent_name))
        #
        #     step_volume_STR = STRtree(step_volume_collection)
        #
        #     # checking reach goal before the check collision. So that at the time step, when an agent reaches goal and
        #     # collide with other agent at the same time, it is consider as reaching destination instead of collision
        #
        #     collided_drone = []
        #     reached_drone = []
        #     for agentIdx_key, agent_passed_volume in agent_moving_vol.items():
        #         # check goal
        #         cur_drone_tar = Point(self.all_agents[agentIdx_key].goal[0][0],
        #                               self.all_agents[agentIdx_key].goal[0][1]).buffer(1, cap_style='round')
        #
        #         mat_cur_tar = shapelypoly_to_matpoly(cur_drone_tar, True, 'c', 'r')
        #         ax.add_patch(mat_cur_tar)
        #         plt.text(self.all_agents[agentIdx_key].goal[0][0],
        #                  self.all_agents[agentIdx_key].goal[0][1],
        #                  "{} goal".format(self.all_agents[agentIdx_key].agent_name))
        #
        #         if cur_drone_tar.intersects(agent_passed_volume):
        #             reached_drone.append(agentIdx_key)
        #             continue  # one drone reached its target no need to check any possible collision for this drone
        #
        #         # check collision
        #         possible_idx = step_volume_STR.query(agent_passed_volume)
        #         for other_agent_cir in step_volume_STR.geometries.take(possible_idx):
        #             if not other_agent_cir.equals(agent_passed_volume):
        #                 # record this volume only when not equals to itself.
        #                 collided_drone.append(agentIdx_key)
        #
        #     # if reached goal, remove the agent from the environment
        #     for i in reached_drone:
        #         del self.all_agents[i]
        #         print("agent_{} reached, it is removed from the environment".format(i))
        #     # Remove element in "collided_drone", such that these elements also present in "reached_drone"
        #     collided_drone = [x for x in collided_drone if x not in reached_drone]
        #     # remove possible duplicates in "collided_drone"
        #     collided_drone = list(set(collided_drone))
        #
        #     # if collide, remove any agents involved in the collision
        #     for i in collided_drone:
        #         del self.all_agents[i]
        #         print("removed agent_ {}, left {} agents".format(i, len(self.all_agents)))
        #     fig.canvas.draw()
        #     plt.show()
        #     time.sleep(2)
        #     fig.canvas.flush_events()
        #     ax.cla()
        
        return next_state, next_state_norm

    def fill_agents(self, max_agent_train, cur_state, norm_cur_state, remove_agent_keys):
        num_lack = int(max_agent_train-len(self.all_agents))
        # int_removed_name = [int(re.search(r'\d+(\.\d+)?', name).group()) for name in remove_agent_named]
        agent_filled = []
        if num_lack > 0:
            for i in range(num_lack):
                # when both drones / all drones reaches the goal at same ts. Will have error. self.all_agents.keys() will become zeros. Debug this problem.
                # ----------------original ------------------
                # selected_key = random.choice(list(self.all_agents.keys()))
                # selected_agent = self.all_agents[selected_key]
                # agent = deepcopy(selected_agent)
                # current_max = max(list(self.all_agents.keys()))
                # ----------------- end of original -----------
                agent = deepcopy(self.dummy_agent)
                if len(self.all_agents) == 0:
                    current_max = max(remove_agent_keys)  # all agents have reaches their destination at same time
                else:
                    try:
                        current_max = max(max(list(self.all_agents.keys())), max(remove_agent_keys))
                    except:
                        print("pause")

                agent.agent_name = 'agent_%s' % str(current_max+1)
                self.all_agents[current_max+1] = agent
                agent_filled.append(current_max+1)
            # overall_state, norm_overall_state = self.fill_agent_reset(agent_filled)
            updated_cur_state, updated_norm_cur_state = self.fill_agent_reset(agent_filled)
            return updated_cur_state, updated_norm_cur_state
        else:
            return cur_state, norm_cur_state

    def central_learning(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature, UPDATE_EVERY):
        with torch.autograd.set_detect_anomaly(True):
            critic_losses, actor_losses = [], []

            cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature, self.all_agents[0].max_grid_obs_dim)

            device = self.all_agents[0].actorNet.device

            # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
            cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
            next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

            # load action, reward, done to tensor
            action = T.tensor(np.array(action), dtype=T.float).to(device)
            reward = T.tensor(np.array(reward), dtype=T.float).to(device)
            done = T.tensor(np.array(done)).to(device)

            # all these three different actions are needed to calculate the loss function
            all_agents_new_actions = []  # actions according to the target network for the new state
            all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
            old_agents_actions = []  # actions the agent actually took

            for agent_idx, agent in self.all_agents.items():
                # actions according to the target network for the new state
                # next_own = T.tensor(next_state[agent_idx][0], dtype=T.float).to(device)
                # next_grid = T.tensor(next_state[agent_idx][1], dtype=T.float).to(device)
                # next_nei = T.tensor(next_state[agent_idx][2], dtype=T.float).to(device)
                # agent_new_states = [next_own, next_grid, next_nei]

                next_own = T.tensor(next_state[agent_idx], dtype=T.float).to(device)
                new_pi = agent.target_actorNet.forward(next_own)  # individual agent's target network

                all_agents_new_actions.append(new_pi)  # record actions generated by each agent's target network
                # actions according to the regular actor network for the current state
                # cur_own = T.tensor(cur_state[agent_idx][0], dtype=T.float).to(device)
                # cur_grid = T.tensor(cur_state[agent_idx][1], dtype=T.float).to(device)
                # cur_nei = T.tensor(cur_state[agent_idx][2], dtype=T.float).to(device)

                # mu_states = [cur_own, cur_grid, cur_nei]
                cur_own = T.tensor(cur_state[agent_idx], dtype=T.float).to(device)

                # using agent's predict network to generate action based off current states
                pi = agent.actorNet.forward(cur_own)

                all_agents_new_mu_actions.append(pi)
                # record actions the agent actually took
                old_agents_actions.append(action[agent_idx])

            new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)  # batch_size X (agent_num X action dim)
            mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
            old_actions = T.cat([acts for acts in old_agents_actions], dim=1)

            # These two losses are used to record overall losses for entire system at each learning step
            critic_losses = []
            actor_losses = []
            # in_order_count = 0
            # handle cost function
            for agent_idx, agent in self.all_agents.items():
                # squeeze() will remove all dimensions with size 1
                # without squeeze() is 10x1x1, so is batch_number x 1 by 1 array.
                # using individual agent's critic prediction network
                # current Q estimate, shape is batch_size X 1
                critic_value = agent.criticNet.forward(cur_state_pre_processed, old_actions)
                # get target Q value for each agent
                with T.no_grad():
                    # next_state_pre_processed is arranged in a way that, always from 1st agent to last agent and ...
                    # agent's idx is ignored, that's why they were stored in a list.
                    # hence, "for agent_idx, agent in self.all_agents.items():" is just to loop through all agents in order
                    # DO NOT use "agent_idx" as index, or else it will produce error, when add/remove agents are added.
                    # First line for critic_value_prime did not involve any individual agent's attributes
                    # because we are using centralized critic, shape is batch_size X 1
                    critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, new_actions)
                    critic_value_prime[done[agent_idx]] = 0.0
                    target = reward[agent_idx] + agent.gamma * critic_value_prime


                # calculate critic loss for each agent
                critic_loss = F.mse_loss(critic_value, target)
                # optimization
                agent.criticNet.optimizer.zero_grad()
                critic_loss.backward()
                agent.criticNet.optimizer.step()

                # actor loss
                actor_loss = agent.criticNet.forward(cur_state_pre_processed, mu).squeeze()
                actor_loss = -T.mean(actor_loss)
                # actor optimization
                agent.actorNet.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                # actor_loss.backward()
                agent.actorNet.optimizer.step()

                agent.update_count = agent.update_count + 1
                if agent.update_count == agent.target_update_step:
                    agent.update_network_parameters()  # soft-update is used here
                    print("{} network updated".format(agent.agent_name))
                    agent.update_count = 0  # reset update count

                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)

            return critic_losses, actor_losses

    def central_learning_v2(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature):
        critic_losses, actor_losses = [], []

        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature,
                                                                          self.all_agents[0].max_grid_obs_dim)

        device = self.all_agents[0].actorNet.device

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

        # load action, reward, done to tensor
        action = T.tensor(np.array(action), dtype=T.float).to(device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(device)
        done = T.tensor(np.array(done)).to(device)

        # all these three different actions are needed to calculate the loss function
        all_agents_new_actions = []  # actions according to the target network for the new state
        all_agents_new_mu_actions = []  # actions according to the regular actor network for the current state
        old_agents_actions = []  # actions the agent actually took

        # this for loop to sort the action, reward, done, as they are in the form (agentNo X batch size X feature)
        for agent_idx, agent in self.all_agents.items():
            # actions according to the target network for the new state
            next_own = T.tensor(next_state[agent_idx][0], dtype=T.float).to(device)
            next_grid = T.tensor(next_state[agent_idx][1], dtype=T.float).to(device)
            next_nei = T.tensor(next_state[agent_idx][2], dtype=T.float).to(device)
            agent_new_states = [next_own, next_grid, next_nei]
            new_pi = agent.target_actorNet.forward(agent_new_states)  # individual agent's target network
            all_agents_new_actions.append(new_pi)  # record actions generated by each agent's target network

            # actions according to the regular actor network for the current state
            cur_own = T.tensor(cur_state[agent_idx][0], dtype=T.float).to(device)
            cur_grid = T.tensor(cur_state[agent_idx][1], dtype=T.float).to(device)
            cur_nei = T.tensor(cur_state[agent_idx][2], dtype=T.float).to(device)
            cur_states = [cur_own, cur_grid, cur_nei]

            # using agent's predict network to generate action based off current states
            pi = agent.actorNet.forward(cur_states)
            all_agents_new_mu_actions.append(pi)  # store the actions generated by actor's predict net based off cur_state

            # record actions the agent actually took
            old_agents_actions.append(action[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)  # batch_size X agent_num X action dim
        old_actions = T.cat([acts for acts in old_agents_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)  # actions generate based off current agent's actor predict net

        for agent_idx, agent in self.all_agents.items():
            # zero grad()
            agent.criticNet.optimizer.zero_grad()
            agent.actorNet.optimizer.zero_grad()
            agent.criticNet.zero_grad()
            agent.actorNet.zero_grad()

            current_Q = agent.criticNet.forward(cur_state_pre_processed, old_actions)
            critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, new_actions)
            critic_value_prime[done[agent_idx]] = 0.0
            target_Q = reward[agent_idx] + agent.gamma * critic_value_prime

            loss_Q = nn.MSELoss()(current_Q, target_Q)
            loss_Q.backward()
            # torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)  # critic network grad clipping
            agent.criticNet.optimizer.step()

            # zero grad()
            agent.criticNet.optimizer.zero_grad()
            agent.actorNet.optimizer.zero_grad()
            agent.criticNet.zero_grad()
            agent.actorNet.zero_grad()

            # evaluate actor loss using current state and actor's critic
            actor_loss = -agent.criticNet.forward(cur_state_pre_processed, mu).squeeze().mean()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            agent.actorNet.optimizer.step()

            critic_losses.append(loss_Q)
            actor_losses.append(actor_loss)

            agent.update_count = agent.update_count + 1
            if agent.update_count == agent.target_update_step:
                agent.update_network_parameters()  # soft-update is used here
                print("{} network updated".format(agent.agent_name))
                agent.update_count = 0  # reset update count

        return critic_losses, actor_losses

    def central_update(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature, ts):
        critic_losses, actor_losses = [], []
        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature,
                                                                          self.all_agents[0].max_grid_obs_dim)
        FloatTensor = torch.FloatTensor
        device = self.all_agents[0].actorNet.device
        # load action, reward, done to tensor
        actionQ = T.tensor(np.array(action).transpose(1, 0, 2), dtype=T.float).contiguous().view(batch_size, -1).to(device)
        #reward = T.tensor(np.array(reward).transpose(1, 0, 2), dtype=T.float).contiguous().view(batch_size, -1).to(device)
        #done = T.tensor(np.array(done).transpose(1, 0, 2)).contiguous().view(batch_size, -1).to(device)
        next_ = T.tensor(np.array(next_state), dtype=T.float).to(device)

        # cur_ = T.tensor(np.array(cur_state), dtype=T.float).to(device)

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

        all_agents_next_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []  # actions the agent actually took

        # for agent_idx, agent in self.all_agents.items():  # for generate next actions
        #     # for next action, from next state go into actor's target net
        #     next_own = T.tensor(next_state[agent_idx], dtype=T.float).to(device)
        #     new_pi = agent.target_actorNet.forward(next_own)  # individual agent's target network
        #     all_agents_next_actions.append(new_pi)
        #     # for current action, from current state go into actor's prediction net
        #     cur_own = T.tensor(cur_state[agent_idx], dtype=T.float).to(device)
        #     pi = agent.actorNet(cur_own)
        #
        #     # pi = pi.clone()
        #
        #     all_agents_new_mu_actions.append(pi)
        #
        #     # record actions the agent actually took
        #     old_agents_actions.append(action[agent_idx])
        #
        # next_actions = T.cat([acts for acts in all_agents_next_actions], dim=1)
        # # next_actions = next_actions.clone()
        # mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        # # might be necessary
        # # mu_one = mu.clone()
        # cur_action = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in self.all_agents.items():

            # current Q estimate, shape is batch_size X 1
            critic_value = agent.criticNet(cur_state_pre_processed, actionQ)

            critic_value_prime = torch.zeros(batch_size, 1).type(FloatTensor)
            next_actions = [self.all_agents[i].target_actorNet(next_[i, :, :]) for i in range(len(self.all_agents))]
            # critic_value_prime = agent.target_criticNet(next_state_pre_processed, next_actions).detach()
            next_action_stack = torch.stack(next_actions).permute(1, 0, 2).contiguous().view(batch_size, -1)

            mask = T.tensor(done[agent_idx]).int()
            flipped_mask = 1 - mask
            critic_value_prime = flipped_mask*agent.target_criticNet(next_state_pre_processed, next_action_stack)

            target = T.tensor(reward[agent_idx]) + agent.gamma * critic_value_prime

            critic_loss = F.mse_loss(critic_value, target.detach())
            agent.criticNet.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.criticNet.optimizer.step()

            action_i = agent.actorNet(T.tensor(cur_state[agent_idx]))
            pi = T.tensor(action).clone()
            pi[agent_idx] = action_i  # only change the action batch with one update action
            mu = pi.view(batch_size, -1)

            # next_actions_cur = [self.all_agents[i].target_actorNet(cur_[i, :, :]) for i in range(len(self.all_agents))]
            # mu = torch.stack(next_actions_cur).permute(1, 0, 2).contiguous().view(batch_size, -1)

            actor_loss = -agent.criticNet(cur_state_pre_processed, mu).mean()
            agent.actorNet.optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.actorNet.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.actorNet.optimizer.step()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            agent.update_count = agent.update_count + 1
            if agent.update_count == agent.target_update_step:
                agent.update_network_parameters()  # soft-update is used here
                print("{} network updated at episode equals to {}".format(agent.agent_name, ts))
                agent.update_count = 0  # reset update count after one update

        return critic_losses, actor_losses

    def central_update_v2(self, ReplayBuffer, batch_size, maxIntruNum, intruFeature):
        critic_losses, actor_losses = [], []
        cur_state, action, reward, next_state, done = ReplayBuffer.sample(batch_size, maxIntruNum, intruFeature,
                                                                          self.all_agents[0].max_grid_obs_dim)

        device = self.all_agents[0].actorNet.device
        # load action, reward, done to tensor
        action = T.tensor(np.array(action), dtype=T.float).to(device)
        reward = T.tensor(np.array(reward), dtype=T.float).to(device)
        done = T.tensor(np.array(done)).to(device)

        # pre-process cur_state and next_state so that they can be used as input for every agent's critic network
        cur_state_pre_processed = preprocess_batch_for_critic_net_v2(cur_state, batch_size)
        next_state_pre_processed = preprocess_batch_for_critic_net_v2(next_state, batch_size)

        all_agents_next_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []  # actions the agent actually took

        for agent_idx, agent in self.all_agents.items():  # for generate next actions
            # for next action, from next state go into actor's target net
            next_own = T.tensor(next_state[agent_idx], dtype=T.float).to(device)
            new_pi = agent.target_actorNet.forward(next_own)  # individual agent's target network
            all_agents_next_actions.append(new_pi)
            # for current action, from current state go into actor's prediction net
            cur_own = T.tensor(cur_state[agent_idx], dtype=T.float).to(device)
            pi = agent.actorNet.forward(cur_own)
            all_agents_new_mu_actions.append(pi)

            # record actions the agent actually took
            old_agents_actions.append(action[agent_idx])

        next_actions = T.cat([acts for acts in all_agents_next_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        cur_action = T.cat([acts for acts in old_agents_actions], dim=1)

        for agent_idx, agent in self.all_agents.items():

            # current Q estimate, shape is batch_size X 1
            critic_value = agent.criticNet.forward(cur_state_pre_processed, cur_action)
            critic_value_prime = agent.target_criticNet.forward(next_state_pre_processed, next_actions)
            critic_value_prime[done[agent_idx]] = 0.0
            target = reward[agent_idx] + agent.gamma * critic_value_prime

            critic_loss = F.mse_loss(critic_value, target.detach())

            agent.criticNet.optimizer.zero_grad()
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.criticNet.optimizer.step()

            actor_loss = -agent.criticNet(cur_state_pre_processed, mu).mean()
            agent.actorNet.optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(agent.actorNet.parameters(), 1)
            # torch.nn.utils.clip_grad_norm_(agent.criticNet.parameters(), 1)
            agent.actorNet.optimizer.step()

            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

            agent.update_count = agent.update_count + 1
            if agent.update_count == agent.target_update_step:
                agent.update_network_parameters()  # soft-update is used here
                print("{} network updated".format(agent.agent_name))
                agent.update_count = 0  # reset update count

        return critic_losses, actor_losses

    def save_model_actor_net(self, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # only actor net is required to be saved, because when evaluation, only actor network is required
        for agent_idx, agent_obj in self.all_agents.items():
            torch.save(agent_obj.actorNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'actor_net')
            # torch.save(agent_obj.target_actorNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'target_actor_net')
            # torch.save(agent_obj.criticNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'critic_net')
            # torch.save(agent_obj.target_criticNet.state_dict(), file_path + '/' +agent_obj.agent_name + 'target_critic_net')






































