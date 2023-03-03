# -*- coding: utf-8 -*-
"""
@Time    : 3/1/2023 7:58 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
import numpy as np
from grid_env_generation_newframe_V1 import env_generation
from env_simulator_V1 import env_simulator


def initialize_parameters():
    n_episodes = 500000
    max_t = 45
    eps_start = 1.0
    eps_end = 0.05  # The minimum number of epsilon
    #eps_decay = 0.99992
    eps_period = 250000  # The number of steps needed for the epsilon to drop until the minimum number of the "eps_end"
    eps = eps_start  # initialize epsilon
    agent_grid_obs = np.zeros((7, 7))
    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size, changed to 16 in V7
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters, 0.001, so 99.9% of the weights in the target network is
    learning_rate = 1e-5  # learning rate, previous = 0.0005 or 5e-4, now changed to 0.002
    UPDATE_EVERY = 50  # how often to update the network
    # generate static env from shape file
    # shapePath = 'D:\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    shapePath = 'F:\githubClone\deep_Q_learning\DQN_new_framework\lakesideMap\lakeSide.shp'
    staticEnv = env_generation(shapePath)  # it is a tuple of 4 element, 1st is the 2D binary array of the filled map, 2nd is the list of all buildings expressed as polygons, 3rd is the gird length, last is list of square grids in the map that has overlapped with the building polygons.
    seed = 3407  # this seed is only used for torch manuel.seed
    # set boundary
    xlow = 455
    xhigh = 680
    ylow = 255
    yhigh = 385
    bound = [xlow, xhigh, ylow, yhigh]
    total_agentNum = 5
    env = env_simulator(staticEnv[0], staticEnv[1], staticEnv[2], bound, staticEnv[3])
    return n_episodes, max_t, eps_start, eps_end, eps_period, eps, env, total_agentNum, agent_grid_obs, BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, learning_rate, UPDATE_EVERY

# if __name__ == '__main__':
#     initialize_parameters()