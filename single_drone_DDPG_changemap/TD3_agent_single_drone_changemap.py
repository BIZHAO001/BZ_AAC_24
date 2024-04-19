# -*- coding: utf-8 -*-
"""
@Time    : 3/4/2024 2:04 PM
@Author  : Mingcheng
@FileName: 
@Description: 
@Package dependency:
"""
from Nnetworks_randomOD_radar_single_drone_DDPG_changemap import critic_single_TwoPortion_TD3, critic_single_obs_wGRU_TwoPortion_TD3, GRUCELL_actor_TwoPortion, ActorNetwork, Stocha_actor, GRU_actor, GRUCELL_actor, CriticNetwork_woGru, CriticNetwork_wGru, critic_single_obs_wGRU, ActorNetwork_TwoPortion, critic_single_TwoPortion, ActorNetwork_OnePortion, critic_single_OnePortion
import torch
import copy
from copy import deepcopy
from torch.optim import Adam
from memory_randomOD_radar_single_drone_DDPG_changemap import ReplayMemory, Experience
# from random_process_MADDPGv3_randomOD import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import time
import numpy as np
import torch as T
from utils_randomOD_radar_single_drone_DDPG_changemap import device
import csv


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class TD3(object):
    def __init__(
            self,
            actor_dim,
            critic_dim,
            dim_act,
            actor_hidden_state_size,
            gru_history_length,
            n_agents,
            args,
            cr_lr,
            ac_lr,
            gamma,
            tau,
            full_observable_critic_flag,
            use_GRU_flag,
            policy_noise=0.2,
            noise_clip=0.5,
            policy_freq=2,
    ):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        if use_GRU_flag:
            self.actors = [GRUCELL_actor_TwoPortion(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
            self.critics = [critic_single_obs_wGRU_TwoPortion_TD3(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
        else:
            self.actors = [ActorNetwork_TwoPortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
            self.critics = [
                critic_single_TwoPortion_TD3(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for
                _ in range(n_agents)]

        self.actors_target = copy.deepcopy(self.actors)
        self.actors_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors]

        self.critics_target = copy.deepcopy(self.critics)
        self.critics_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics]

        self.total_steps = 0
        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        self.var = [1.0 for i in range(n_agents)]
        self.GAMMA = gamma
        self.tau = tau

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

    def choose_action(self, state, cur_total_step, cur_episode, step, total_training_steps, noise_start_level, actor_hiddens, use_LSTM_flag, noisy=True, use_GRU_flag=False):
        # ------------- MADDPG_test_181123_10_10_54 version noise -------------------
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        noise_value = np.zeros(2)

        # if len(gru_history) < self.args.gru_history_length:
        #     # Append zero arrays to fill the gru_history
        #     for _ in range(self.args.gru_history_length - len(gru_history)):
        #         zero_array = np.zeros((self.n_agents, self.n_actor_dim[0]))
        #         gru_history.append(zero_array)
        # gru_history_input = np.array(gru_history)
        # gru_history_input = np.expand_dims(gru_history_input, axis=0)

        actions = torch.zeros(self.n_agents, self.n_actions)
        if use_GRU_flag:
            act_hn = torch.zeros(self.n_agents, self.actors[0].rnn_hidden_dim)
        elif use_LSTM_flag:
            act_hn = torch.zeros(self.n_agents, self.actors[0].rnn_hidden_dim)
        else:
            act_hn = torch.zeros(self.n_agents, self.n_actions)
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # this for loop used to decrease noise level for all agents before taking any action
        # gru_history_input = torch.FloatTensor(gru_history_input).to(device)  # batch x seq_length x no_agent x feature_length
        gru_history_input = torch.FloatTensor(actor_hiddens).unsqueeze(0).to(device)  # batch x no_agent x feature_length
        for i in range(self.n_agents):
            self.var[i] = self.get_custom_linear_scaling_factor(cur_episode, total_training_steps, noise_start_level)  # self.var[i] will decrease as the episode increase
            # self.var[i] = self.linear_decay(episode, eps_end, noise_start_level)  # self.var[i] will decrease as the episode increase

        for i in range(self.n_agents):
            # sb = obs[i].detach()
            # sb_grid = obs_grid[i].detach()
            # sb_surAgent = all_obs_surAgent[i].detach()
            sb = obs[i]
            sb_grid = obs_grid[i]
            # sb_surAgent = all_obs_surAgent[i]
            # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            # act = self.actors[i]([sb.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
            # act, hn = self.actors[i](sb.unsqueeze(0), gru_history_input[:,:,i,:])
            # act, hn = self.actors[i](sb.unsqueeze(0), gru_history_input[:, i, :])
            self.actors[i].eval()
            if use_GRU_flag:
                act, hn = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)], gru_history_input[:, i, :])
            else:
                act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)])
            if noisy:
                noise_value = np.random.randn(2) * self.var[i]
                act += torch.from_numpy(noise_value).type(FloatTensor)
                # print("Episode {}, agent {}, noise level is {}".format(episode, i, self.var[i]))
                act = torch.clamp(act, -1.0, 1.0)  # when using stochastic policy, we are not require to clamp again.

            actions[i, :] = act
            if use_GRU_flag:
                act_hn[i, :] = hn
            else:
                act_hn[i, :] = torch.zeros(1, self.n_actions)
        self.actors[i].train()
        self.total_steps += 1
        # ------------- end of MADDPG_test_181123_10_10_54 version noise -------------------

        # obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        # obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        # noise_value = np.zeros(2)
        # all_obs_surAgent = []
        # for each_agent_sur in state[2]:
        #     try:
        #         each_obs_surAgent = np.squeeze(np.array(each_agent_sur), axis=1)
        #         all_obs_surAgent.append(torch.from_numpy(each_obs_surAgent).float().to(device))
        #     except:
        #         print("pause and check")
        #
        # # obs_surAgent = torch.from_numpy(np.stack(state[2])).float().to(device)
        #
        # actions = torch.zeros(self.n_agents, self.n_actions)
        # FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        # # this for loop used to decrease noise level for all agents before taking any action
        # # for i in range(self.n_agents):
        # #     self.var[i] = self.get_custom_linear_scaling_factor(episode, eps_end, noise_start_level)  # self.var[i] will decrease as the episode increase
        #
        # for i in range(self.n_agents):
        #     # sb = obs[i].detach()
        #     # sb_grid = obs_grid[i].detach()
        #     # sb_surAgent = all_obs_surAgent[i].detach()
        #     sb = obs[i]
        #     sb_grid = obs_grid[i]
        #     sb_surAgent = all_obs_surAgent[i]
        #     # act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
        #     # act = self.actors[i]([sb.unsqueeze(0), sb_surAgent.unsqueeze(0)]).squeeze()
        #     act = self.actors[i]([sb.unsqueeze(0)]).squeeze()
        #     if noisy:
        #         noise_value = np.random.randn(2) * self.var[i]
        #         act += torch.from_numpy(noise_value).type(FloatTensor)
        #         # print("Episode {}, agent {}, noise level is {}".format(episode, i, self.var[i]))
        #         act = torch.clamp(act, -1.0, 1.0)  # when using stochastic policy, we are not require to clamp again.
        #
        #     actions[i, :] = act
        #
        # for i in range(self.n_agents):
        #     if self.var[i] > 0.05:  # noise decrease at every step instead of every episode.
        #         self.var[i] = self.var[i] * 0.999998
        # self.steps_done += 1

        return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        # return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        # return actions.data.cpu().numpy(), noise_value

    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, action, wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):
        if (len(self.memory) <= self.batch_size):
            return None, None, single_eps_critic_cal_record
        c_loss = []
        a_loss = []
        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        # Sample replay buffer
        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        if use_GRU_flag:
            agents_next_hidden_state = torch.stack(batch.next_hidden).type(FloatTensor)
            agents_cur_hidden_state = torch.stack(batch.cur_hidden).type(FloatTensor)
        # stack tensors only once
        stacked_elem_0 = torch.stack([elem[0] for elem in batch.states]).to(device)
        stacked_elem_1 = torch.stack([elem[1] for elem in batch.states]).to(device)
        if full_observable_critic_flag == True:
            stacked_elem_0_combine = stacked_elem_0.view(self.batch_size, -1)  # own_state only
            stacked_elem_1_combine = stacked_elem_1.view(self.batch_size, -1)  # own_state only

        # for next state
        next_stacked_elem_0 = torch.stack([elem[0] for elem in batch.next_states]).to(device)
        next_stacked_elem_1 = torch.stack([elem[1] for elem in batch.next_states]).to(device)
        if full_observable_critic_flag == True:
            next_stacked_elem_0_combine = next_stacked_elem_0.view(self.batch_size, -1)
            next_stacked_elem_1_combine = next_stacked_elem_1.view(self.batch_size, -1)

        # for done
        dones_stacked = torch.stack([three_agent_dones for three_agent_dones in batch.dones]).to(device)

        for agent in range(self.n_agents):
            with torch.no_grad():
                # Select action according to policy and add clipped noise
                noise = (torch.randn_like(torch.from_numpy(action).to(device)) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
                # non_final_next_states_actorin = [next_stacked_elem_0]  # 2 portion available
                non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1]  # 2 portion available

                if use_GRU_flag:
                    non_final_next_actions = [(self.actors_target[i](
                        [non_final_next_states_actorin[0][:, i, :], non_final_next_states_actorin[1][:, i, :]],
                        agents_next_hidden_state[:, i, :])[0] + noise).clamp(-1,1) for i in range(self.n_agents)]
                else:
                    non_final_next_actions = [(self.actors_target[i](
                        [non_final_next_states_actorin[0][:, i, :], non_final_next_states_actorin[1][:, i, :]])+noise).clamp(-1,1) for i in
                                              range(self.n_agents)]

                if use_GRU_flag:
                    next_target_Q1, next_target_h_Q1, next_target_Q2, next_target_h_Q2 = \
                    self.critics_target[agent]([next_stacked_elem_0[:, agent, :], next_stacked_elem_1[:, agent, :]],
                                               non_final_next_actions[agent], agents_next_hidden_state[:, agent, :])
                else:
                    next_target_Q1, next_target_Q2 = self.critics_target[agent]([next_stacked_elem_0[:,agent,:], next_stacked_elem_1[:,agent,:]], non_final_next_actions[agent])

                target_Q = torch.min(next_target_Q1, next_target_Q2).squeeze(1)
                target_Q = (reward_batch[:, agent]) + (self.GAMMA * target_Q * (1-dones_stacked[:, agent]))
                target_Q = target_Q.unsqueeze(1)

            # Get current Q estimates
            if use_GRU_flag:
                current_Q1, current_h_Q1, current_Q2, current_h_Q2, = self.critics[agent](
                    [stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                    action_batch[:, agent, :], agents_cur_hidden_state[:, agent, :])
            else:
                current_Q1, current_Q2 = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                action_batch[:, agent, :])

            # Compute critic loss
            c_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)

            # Optimize the critic
            self.critics_optimizer[agent].zero_grad()
            c_loss.backward()
            self.critics_optimizer[agent].step()

        # Delayed policy updates
        if self.total_steps % self.policy_freq == 0:

            # Compute actor losse
            if use_GRU_flag:
                a_loss = -self.critics[agent].q1([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                    action_batch[:, agent, :], agents_cur_hidden_state[:, agent, :])[0].mean()
            else:
                a_loss = -self.critics[agent].q1([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],action_batch[:, agent, :]).mean()

            # Optimize the actor
            self.actors_optimizer[agent].zero_grad()
            a_loss.backward()
            self.actors_optimizer[agent].step()

            # Update the frozen target models
            for param, target_param in zip(self.critics[agent].parameters(), self.critics_target[agent].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actors[agent].parameters(), self.actors_target[agent].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        if isinstance(c_loss, list) or isinstance(a_loss, list):
            return None, None, single_eps_critic_cal_record
        return c_loss, a_loss, single_eps_critic_cal_record

    def save_model(self, episode, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), file_path + '/' +'episode_'+str(episode)+'_'+'agent_'+ str(i) + 'actor_net.pth')

    def load_model(self, filePath):
        if self.args.model_episode:
            path_flag = True
            if path_flag:
                print("load model!")
                for path_idx, path in enumerate(filePath):
                    self.actors[path_idx].load_state_dict(torch.load(path))

    def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0.05):
        # Calculate the slope of the linear decrease only up to eps_end
        if episode <= eps_end:
            slope = (end_scale - start_scale) / (eps_end - 1)
            current_scale = start_scale + slope * (episode - 1)
        else:
            current_scale = end_scale
        return current_scale