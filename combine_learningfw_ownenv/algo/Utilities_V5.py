# -*- coding: utf-8 -*-
"""
@Time    : 3/13/2023 1:28 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from matplotlib.patches import Polygon as matPolygon
import torch as T
import numpy as np
import torch
import os
import matplotlib
import matplotlib.pyplot as plt
from shapely.strtree import STRtree
from shapely.geometry import LineString, Point, Polygon


def sort_polygons(polygons):  # this sorting is left to right, but bottom to top. so, 0th is below 2nd. [[2,3],
    # [0,1]]
    boxes = [polygon.bounds for polygon in polygons]
    sorted_boxes = sorted(boxes, key=lambda box: (box[1], box[0]))
    sorted_polygons = [polygons[boxes.index(box)] for box in sorted_boxes]
    return sorted_polygons


# this is to remove the need for the package descrete
def shapelypoly_to_matpoly(ShapelyPolgon, inFill=False, Edgecolor='black', FcColor='blue'):
    xcoo, ycoo = ShapelyPolgon.exterior.coords.xy
    matPolyConverted = matPolygon(xy=list(zip(xcoo, ycoo)), fill=inFill, edgecolor=Edgecolor, facecolor=FcColor)
    return matPolyConverted


def extract_individual_obs(combine_state, agent_idx):
    individual_obs = []
    self_obs = combine_state[agent_idx][0]
    self_obs_grid = combine_state[agent_idx][1]
    self_surround = combine_state[agent_idx][2]
    individual_obs = [self_obs, self_obs_grid, self_surround]
    # return individual_obs  # for old V5
    return combine_state[agent_idx]


def map_range(value, coe_a):
    # Calculate the normalized value
    out_min = -coe_a
    out_max = coe_a
    # tanh() has range of -1 to 1
    in_min = -1
    in_max = 1
    normalized = (value - in_min) / (in_max - in_min)
    # Map the normalized value to the output range
    mapped = out_min + (normalized * (out_max - out_min))
    # Return the mapped value
    return mapped


def compute_potential_conflict(pc_list, cur_drone_pos, cur_drone_vel, cur_drone_protRad, cur_neigh_pos, cur_neigh_vel,
                               cur_neigh_protRad, cur_neigh_idx, current_ts):
    minus_rel_dist_before = -1 * (cur_drone_pos - cur_neigh_pos)  # always current drone - neighbours
    rel_vel_before = (cur_drone_vel - cur_neigh_vel)
    rel_vel_SQnorm_before = np.square(np.linalg.norm(rel_vel_before))
    if (current_ts == 0) & (rel_vel_SQnorm_before == 0):
        # this if-else if to remove the runtimeWarning due to getting a value of nan
        # because we initialized the velocity for each drone as 0 at start of each episode.
        # Therefore, will have runtime warning
        pass
    else:

        t_cpa_before = np.dot(minus_rel_dist_before, rel_vel_before) / rel_vel_SQnorm_before
        d_cpa_before = np.linalg.norm(((cur_drone_pos - cur_neigh_pos) + (rel_vel_before * t_cpa_before)))
        if (t_cpa_before >= 1) and (d_cpa_before < (cur_drone_protRad + cur_neigh_protRad)):
            pc_list.append(cur_neigh_idx)
    return pc_list


def padding_list(max_grid_obs_dim, input_list):
    # padding actions
    if isinstance(input_list, np.float64):
        print("check")
    tobePad_gridObs = list(np.zeros(max_grid_obs_dim - len(input_list), dtype=int))
    padded_gridObs = input_list + tobePad_gridObs
    return padded_gridObs


def preprocess_batch_for_critic_net(input_state, batch_size):
    critic_own_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents
    critic_grid_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents
    critic_neigh_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents
    for batch_idx in range(batch_size):
        critic_own_cur_state = []
        critic_own_grid_state = []
        critic_own_neigh_state = []
        for agent_cur in input_state:
            critic_own_cur_state.append(agent_cur[0][batch_idx, :])
            critic_own_grid_state.append(agent_cur[1][batch_idx, :])
            # for neigh, first: max_nei_num X single_nei_features flatten to 1D array
            flat_nei = agent_cur[2][batch_idx, :].flatten()  # default is flatten in a row.
            critic_own_neigh_state.append(flat_nei)
        critic_own_batched_cur_state.append(np.array(critic_own_cur_state).reshape((1, -1)))
        critic_grid_batched_cur_state.append(np.array(critic_own_grid_state).reshape((1, -1)))
        critic_neigh_batched_cur_state.append(np.array(critic_own_neigh_state).reshape((1, -1)))

    cur_state_pre_processed = [T.tensor(np.array(critic_own_batched_cur_state)),
                               T.tensor(np.array(critic_grid_batched_cur_state)),
                               T.tensor(np.array(critic_neigh_batched_cur_state))]
    return cur_state_pre_processed


def preprocess_batch_for_critic_net_v2(input_state, batch_size):
    critic_own_batched_cur_state = []  # batch_size X one_agent_feature * max_num_agents

    for batch_idx in range(batch_size):
        critic_own_cur_state = []

        for agent_cur in input_state:
            critic_own_cur_state.append(agent_cur[batch_idx, :])




        critic_own_batched_cur_state.append(np.array(critic_own_cur_state).reshape((1, -1)))


    cur_state_pre_processed = T.tensor(np.array(critic_own_batched_cur_state))  # batch X (1 x no_agent x feature size)

    return cur_state_pre_processed


class OUNoise:

    def __init__(self, action_dimension, largest_Nsigma, smallest_Nsigma, ini_sigma, mu=0, theta=0.15):  # sigma is the initial magnitude of the OU_noise
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = ini_sigma
        self.largest_sigma = largest_Nsigma
        self.smallest_sigma = smallest_Nsigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


def BetaNoise(action, noise_scale):
    action = action.detach().numpy()  # since the input is a tensor we must convert it to numpy before operations
    sign = np.sign(action)  # tracking the sign so we can flip the samples later
    action = abs(action)  # we only use right tail of beta
    alpha = 1 / noise_scale  # this determines the how contentrated the beta dsn is
    value = 0.5 + action / 2  # converting from action space of -1 to 1 to beta space of 0 to 1
    beta = alpha * (1 - value) / value  # calculating beta
    beta = beta + 1.0 * (
                (alpha - beta) / alpha)  # adding a little bit to beta prevents actions getting stuck at -1 or 1
    sample = np.random.beta(alpha, beta)  # sampling from the beta distribution
    sample = sign * sample + (1 - sign) / 2  # flipping sample if sign is <0 since we only use right tail of beta dsn

    action_output = 2 * sample - 1  # converting back to action space -1 to 1
    return torch.tensor(action_output)  # converting back to tensor


def GaussNoise(action, noise_scale):
    n = np.random.normal(0, 1, len(action))  # create some standard normal noise
    return torch.clamp(action + torch.tensor(noise_scale * n).float(), -1, 1)  # add the noise to the actions


def WeightedNoise(action, noise_scale, action_type):
    """
    Returns the epsilon scaled noise distribution for adding to Actor
    calculated action policy.
    """
    if action_type == 'continuous':
        target = np.random.uniform(-1, 1, 2)  # the action space is -1 to 1
    elif action_type == 'discrete':
        target = np.random.uniform(0, 1, 4)  # action space is discrete
        target = target / sum(target)
    action = noise_scale * target + (
                1 - noise_scale) * action.detach().numpy()  # take a weighted average with noise_scale as the noise weight
    return torch.tensor(action).float()

def display_trajectory(cur_env, combined_trajectory):
    episode_to_show = 4999
    episode_steps = combined_trajectory[episode_to_show]
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(1, 1)
    # # draw link towards destination for all drones, destination for each drone didn't change
    for agentIdx in cur_env.all_agents:
        plt.plot([cur_env.all_agents[agentIdx].ini_pos[0], cur_env.all_agents[agentIdx].goal[0][0]],
                 [cur_env.all_agents[agentIdx].ini_pos[1], cur_env.all_agents[agentIdx].goal[0][1]], '--', color='c')
    reward, done = [], []
    crash_penalty = -100
    reach_target = 1000
    potential_conflict_count = 0
    fixed_domino_reward = 1
    x_left_bound = LineString([(cur_env.bound[0], -9999), (cur_env.bound[0], 9999)])
    x_right_bound = LineString([(cur_env.bound[1], -9999), (cur_env.bound[1], 9999)])
    y_bottom_bound = LineString([(-9999, cur_env.bound[2]), (9999, cur_env.bound[2])])
    y_top_bound = LineString([(-9999, cur_env.bound[3]), (9999, cur_env.bound[3])])
    for step_idx, agents_traj in enumerate(episode_steps):
        step_R = []
        step_D = []
        for ea_idx, each_agent in enumerate(agents_traj):
            # propagate environment
            cur_env.all_agents[ea_idx].pre_pos = cur_env.all_agents[ea_idx].pos
            cur_env.all_agents[ea_idx].pos = np.array([each_agent[0], each_agent[1]])
            # check reward for each step
            # calculate the deviation from the reference path after an action has been taken
            curPoint = Point(cur_env.all_agents[ea_idx].pos)
            host_refline = LineString([cur_env.all_agents[ea_idx].ini_pos, cur_env.all_agents[ea_idx].goal[0]])
            cross_track_deviation = curPoint.distance(host_refline)  # deviation from the reference line, cross track error

            host_pass_line = LineString([cur_env.all_agents[ea_idx].pre_pos, cur_env.all_agents[ea_idx].pos])
            host_passed_volume = host_pass_line.buffer(cur_env.all_agents[ea_idx].protectiveBound, cap_style='round')
            tar_circle = Point(cur_env.all_agents[ea_idx].goal[0]).buffer(1, cap_style='round')
            goal_cur_intru_intersect = host_passed_volume.intersection(tar_circle)
            if not goal_cur_intru_intersect.is_empty:  # reached goal?
                print("drone_{} has reached its goal at time step {}".format(ea_idx, step_idx))
                step_D.append(True)
                step_R.append(np.array(reach_target))
            # exceed bound condition, don't use current point, use current circle or else will have condition that
            elif x_left_bound.intersects(host_passed_volume) or x_right_bound.intersects(host_passed_volume) or y_bottom_bound.intersects(host_passed_volume) or y_top_bound.intersects(host_passed_volume):
                print("drone_{} has crash into boundary at time step {}".format(ea_idx, step_idx))
                step_R.append(np.array(crash_penalty))
                step_D.append(True)
            # exceed bound or crash into buildings or crash with other neighbors
            # elif (self.all_agents[drone_idx].pos[0] <= self.bound[0] or self.all_agents[drone_idx].pos[0] >= self.bound[1] or
            #       self.all_agents[drone_idx].pos[1] <= self.bound[2] or self.all_agents[drone_idx].pos[1] >= self.bound[3] or
            #       collide_building == 1 or len(collision_drones) > 0):
            #     reward.append(np.array(crash_penalty))
            #     done.append(True)
            #     if (collide_building == 0) or len(collision_drones) == 0:
            #         print("drone_{} has crash into boundary at time step {}". format(drone_idx, current_ts))
            else:  # a normal step taken
                step_D.append(False)
                crossCoefficient = 1
                goalCoefficient = 6
                # cross track error term
                cross_track_error = (20 / ((cross_track_deviation * cross_track_deviation) / 200 + 1)) - 3.5
                # Distance between drone and its goal for two consecutive time step
                before_dist_hg = np.linalg.norm(cur_env.all_agents[ea_idx].pre_pos - cur_env.all_agents[ea_idx].goal[0])
                after_dist_hg = np.linalg.norm(cur_env.all_agents[ea_idx].pos - cur_env.all_agents[ea_idx].goal[0])  # distance to goal after action
                delta_hg = goalCoefficient * (before_dist_hg - after_dist_hg)
                # a small penalty for discourage the agent to stay in one single spot
                if (before_dist_hg - after_dist_hg) <= 2:
                    small_step_penalty = 50
                else:
                    small_step_penalty = 0
                # Domino term also use as an indicator for agent to avoid other drones. so no need to specifically
                # add a term to avoid surrounding drones
                # step_reward = crossCoefficient*cross_track_error + delta_hg + dominoTerm - small_step_penalty
                step_reward = delta_hg
                # convert to arr
                step_R.append(np.array(step_reward))
                plt.text(each_agent[0] + 5, each_agent[1], str(np.array(round(step_reward,1))))

            # plot self_circle of the drone
            self_circle = Point(each_agent[0], each_agent[1]).buffer(2.5, cap_style='round')
            grid_mat_Scir = shapelypoly_to_matpoly(self_circle, False, 'k')

            # label drone time step for the position
            plt.text(each_agent[0], each_agent[1], str(ea_idx))
            #plt.text(each_agent[0]+5, each_agent[1], str(step_idx))

            ax.add_patch(grid_mat_Scir)
        reward.append(step_R)
        done.append(step_D)
    # draw occupied_poly
    for one_poly in cur_env.world_map_2D_polyList[0][0]:
        one_poly_mat = shapelypoly_to_matpoly(one_poly, True, 'y', 'b')
        ax.add_patch(one_poly_mat)
    # draw non-occupied_poly
    for zero_poly in cur_env.world_map_2D_polyList[0][1]:
        zero_poly_mat = shapelypoly_to_matpoly(zero_poly, False, 'y')
        # ax.add_patch(zero_poly_mat)

    # show building obstacles
    for poly in cur_env.buildingPolygons:
        matp_poly = shapelypoly_to_matpoly(poly, False, 'red')  # the 3rd parameter is the edge color
        ax.add_patch(matp_poly)

    plt.axis('equal')
    plt.xlim(cur_env.bound[0], cur_env.bound[1])
    plt.ylim(cur_env.bound[2], cur_env.bound[3])
    plt.axvline(x=cur_env.bound[0], c="green")
    plt.axvline(x=cur_env.bound[1], c="green")
    plt.axhline(y=cur_env.bound[2], c="green")
    plt.axhline(y=cur_env.bound[3], c="green")
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.show()
    return reward

def action_selection_statistics(action_selection_collection):
    all_action_collection_x = []
    all_action_collection_y = []
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    matplotlib.use('TkAgg')
    fig, ax = plt.subplots(1, 1)
    for each_eps_collection in action_selection_collection:
        for each_step in each_eps_collection:
            for agent_idx, agent_val in each_step.items():
                all_action_collection_x.append(agent_val[0][0])
                all_action_collection_y.append(agent_val[0][1])

    # Set the number of bins for x and y
    num_bins = 20
    # Create the 2D histogram
    plt.hist2d(all_action_collection_x, all_action_collection_y, bins=num_bins)

    # Set the x-axis and y-axis labels
    plt.xlabel('X')
    plt.ylabel('Y')

    # Set the title of the histogram
    plt.title('2D Histogram of X and Y')

    # Add a colorbar
    plt.colorbar()

    # Show the histogram
    plt.show()





