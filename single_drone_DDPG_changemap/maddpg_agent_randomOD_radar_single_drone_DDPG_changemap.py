# from Nnetworks_MADDPGv3 import CriticNetwork_0724, ActorNetwork
from Nnetworks_randomOD_radar_single_drone_DDPG_changemap import *
import torch
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


class MADDPG:
    def __init__(self, actor_dim, critic_dim, dim_act, actor_hidden_state_size, gru_history_length, n_agents, args, cr_lr, ac_lr, gamma, tau, full_observable_critic_flag, use_GRU_flag, use_attention_flag, attention_only, use_LSTM_flag):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []

        # original
        # self.actors = [Actor(dim_obs, dim_act) for _ in range(n_agents)]
        # self.critics = [Critic(n_agents, dim_obs, dim_act) for _ in range(n_agents)]

        # self.actors = [Stocha_actor(actor_dim, dim_act) for _ in range(n_agents)]  # use stochastic policy
        if use_GRU_flag:
            if use_attention_flag:
                # self.actors = [GRUCELL_actor_TwoPortion_wATT(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
                self.actors = [GRUCELL_actor_TwoPortion_wATT_v2(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
                self.critics = [critic_single_obs_wGRU_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
                # self.critics = [critic_single_obs_wGRU_TwoPortion_att(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
            else:
                self.actors = [GRU_batch_actor_TwoPortion(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
                # self.critics = [critic_single_obs_wGRU_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
                self.critics = [critic_single_obs_GRU_batch_twoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]

        elif use_LSTM_flag:
            self.actors = [LSTM_batch_actor_TwoPortion(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
            self.critics = [critic_single_obs_LSTM_batch_twoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                                    actor_hidden_state_size) for _ in range(n_agents)]

        elif attention_only:
            self.actors = [actor_TwoPortion_wATT(actor_dim, dim_act) for _ in
                           range(n_agents)]  # use deterministic policy
            self.critics = [critic_single_obs_TwoPortion_wATT(critic_dim, n_agents, dim_act) for _ in range(n_agents)]

        else:
            self.actors = [ActorNetwork_TwoPortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
            self.critics = [
                critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for
                _ in range(n_agents)]
        # self.actors = [ActorNetwork_OnePortion(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
        # self.actors = [GRUCELL_actor(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic with GRU module policy
        # self.critics = [CriticNetwork_0724(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        # self.critics = [CriticNetwork(critic_dim, n_agents, dim_act) for _ in range(n_agents)]
        # self.critics = [CriticNetwork_wGru(critic_dim, n_agents, dim_act, gru_history_length) for _ in range(n_agents)]
        # if full_observable_critic_flag:
        #     self.critics = [critic_combine_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
        #                                               actor_hidden_state_size) for _ in range(n_agents)]
        # else:
        #     self.critics = [critic_single_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
        # self.critics = [critic_single_OnePortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]

        self.n_agents = n_agents
        self.n_actor_dim = actor_dim
        self.n_critic_dim = critic_dim
        self.n_actions = dim_act

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.memory = ReplayMemory(args.memory_length)
        self.batch_size = args.batch_size
        self.use_cuda = torch.cuda.is_available()
        # self.episodes_before_train = args.episode_before_train

        # self.GAMMA = 0.95  # original
        # self.tau = 0.01  # original

        self.GAMMA = gamma
        self.tau = tau

        self.var = [1.0 for i in range(n_agents)]
        # self.var = [0.5 for i in range(n_agents)]

        # original, critic learning rate is 10 times larger compared to actor
        # self.critic_optimizer = [Adam(x.parameters(), lr=0.001) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=0.0001) for x in self.actors]

        self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr) for x in self.actors]

        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        self.episode_done = 0

    def load_model(self, filePath):
        if self.args.model_episode:
            path_flag = True
            # for idx in range(self.n_agents):
            #     path_flag = path_flag \
            #                 and (os.path.exists("trained_model/maddpg/actor["+ str(idx) + "]_"
            #                                     +str(self.args.model_episode)+".pth")) \
            #                 and (os.path.exists("trained_model/maddpg/critic["+ str(idx) + "]_"
            #                                     +str(self.args.model_episode)+".pth"))

            if path_flag:
                print("load model!")
                # for idx in range(self.n_agents):
                #     actor = torch.load("trained_model/maddpg/actor["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                #     critic = torch.load("trained_model/maddpg/critic["+ str(idx) + "]_"+str(self.args.model_episode)+".pth")
                #     self.actors[idx].load_state_dict(actor.state_dict())
                #     self.critics[idx].load_state_dict(critic.state_dict())
                for path_idx, path in enumerate(filePath):
                    self.actors[path_idx].load_state_dict(torch.load(path))



        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def save_model(self, episode, file_path):
        # if not os.path.exists("./trained_model_myenv/"):
        #     os.mkdir("./trained_model_myenv/")
        # if not os.path.exists("./trained_model/" + str(self.args.algo) + "/"):
        #     # os.mkdir(r"F:\githubClone\MAProj_myversion\algo/trained_model/" + str(self.args.algo))
        #     os.mkdir(r"D:\Multi_agent_AAC\old_framework_test\algo/trained_model/" + str(self.args.algo))
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        for i in range(self.n_agents):
            torch.save(self.actors[i].state_dict(), file_path + '/' +'episode_'+str(episode)+'_'+'agent_'+ str(i) + 'actor_net.pth')

    def update(self, i_episode):

        self.train_num = i_episode
        if self.train_num <= self.episodes_before_train:
            return None, None

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        for agent in range(self.n_agents):

            non_final_mask = BoolTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))  # create a boolean tensor, that has same length as the "batch.next_states", if an element is batch.next_state is not "None" then assign a True value, False otherwise.
            # state_batch: batch_size x n_agents x dim_obs
            state_batch = torch.stack(batch.states).type(FloatTensor)
            action_batch = torch.stack(batch.actions).type(FloatTensor)
            reward_batch = torch.stack(batch.rewards).type(FloatTensor)
            non_final_next_states = torch.stack([s for s in batch.next_states if s is not None]).type(FloatTensor)  # create a new tensor, but exclude the None values in the old tensor which is "batch.next_states"
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()

            current_Q = self.critics[agent](whole_state, whole_action)
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:, i,:]) for i in range(self.n_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())  # using () at outer most will leads to creation of a new tensor, (batch_size X agentNo X action_dim)

            target_Q = torch.zeros(self.batch_size).type(FloatTensor)
            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states), # .view(-1, self.n_agents * self.n_states)
                non_final_next_actions.view(-1, self.n_agents * self.n_actions)).squeeze() # .view(-1, self.n_agents * self.n_actions)

            reward_sum = sum([reward_batch[:,agent_idx] for agent_idx in range(self.n_agents)])
            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (reward_batch[:, agent].unsqueeze(1))
            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            self.critic_optimizer[agent].zero_grad()
            self.actors[agent].zero_grad()
            self.critics[agent].zero_grad()

            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            # ac = action_batch
            ac[:, agent, :] = action_i  # replacing every single element in "ac[:, agent, :]"
            whole_action = ac.view(self.batch_size, -1)

            actor_loss = -self.critics[agent](whole_state, whole_action).mean()
            # actor_loss += (action_i ** 2).mean() * 1e-3
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            torch.nn.utils.clip_grad_norm_(self.critics[agent].parameters(), 1)
            self.actor_optimizer[agent].step()
            # self.critic_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.train_num % 100 == 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def update_myown_ddpg(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, action, use_LSTM_flag, wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):
        self.train_num = i_episode
        if len(self.memory) <= self.batch_size:
            return None, None, single_eps_critic_cal_record

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        #Reset the recurrent layer's hidden state
        lstm_actor_hidden = None
        lstm_target_actor_hidden = None
        lstm_critic_hidden = None
        lstm_target_critic_hidden = None

        c_loss = []
        a_loss = []

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

        # use the stacked tensors
        # current_state in the form of list of length of agents in the environments, then, batchNo X individual Feature length
        # cur_state_list1 = [stacked_elem_0[:, i, :] for i in range(self.n_agents)]

        # for next state
        next_stacked_elem_0 = torch.stack([elem[0] for elem in batch.next_states]).to(device)
        next_stacked_elem_1 = torch.stack([elem[1] for elem in batch.next_states]).to(device)
        if full_observable_critic_flag == True:
            next_stacked_elem_0_combine = next_stacked_elem_0.view(self.batch_size, -1)
            next_stacked_elem_1_combine = next_stacked_elem_1.view(self.batch_size, -1)

        # for done
        dones_stacked = torch.stack([three_agent_dones for three_agent_dones in batch.dones]).to(device)

        for agent in range(self.n_agents):
            # whole_ownState = stacked_elem_0_combine  # own_state only

            # non_final_next_states_actorin = [next_stacked_elem_0]  # 2 portion available
            non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1]  # 2 portion available

            # configured for target Q

            whole_curren_action = action_batch.view(self.batch_size, -1)

            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], history_batch[:,:,i,:])[0] for i in range(self.n_agents)]
            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], agents_next_hidden_state[:,i,:])[0] for i in range(self.n_agents)]

            # non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][:,i,:], non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]

            # non_final_next_combine_actions = torch.stack(non_final_next_actions).view(self.batch_size, -1)

            # get current Q-estimate, using agent's critic network
            # current_Q = self.critics[agent](whole_state, whole_action, whole_agent_combine_gru)
            # current_Q = self.critics[agent](whole_state, whole_action, history_batch[:, :, agent, :])
            if use_GRU_flag:
                non_final_next_actions = [self.actors_target[i](
                    [non_final_next_states_actorin[0][:, i, :], non_final_next_states_actorin[1][:, i, :]],
                    agents_next_hidden_state[:, i, :])[0] for i in range(self.n_agents)]
                current_Q = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], action_batch[:, agent, :], agents_cur_hidden_state[:, agent, :])[0]
            elif use_LSTM_flag:
                # lstm_target_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_target_actor_hidden is None else lstm_target_actor_hidden
                # lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_critic_hidden is None else lstm_critic_hidden
                lstm_target_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)

                lstm_target_actor_hidden = (lstm_target_actor_hidden[0].detach(), lstm_target_actor_hidden[1].detach())
                lstm_critic_hidden = (lstm_critic_hidden[0].detach(), lstm_critic_hidden[1].detach())

                non_final_next_actions = [self.actors_target[i](
                    [non_final_next_states_actorin[0][:, i, :].unsqueeze(1), non_final_next_states_actorin[1][:, i, :].unsqueeze(1)],
                    lstm_target_actor_hidden) for i in range(self.n_agents)]
                lstm_target_actor_hidden = non_final_next_actions[agent][1]
                current_Q, lstm_critic_hidden = self.critics[agent]([stacked_elem_0[:, agent, :].unsqueeze(1), stacked_elem_1[:, agent, :].unsqueeze(1)],
                                                action_batch[:, agent, :].unsqueeze(1), lstm_critic_hidden)
            else:
                non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][:,i,:], non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]
                current_Q = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                action_batch[:, agent, :])

            if use_GRU_flag:
                next_target_critic_value = \
                self.critics_target[agent]([next_stacked_elem_0[:, agent, :], next_stacked_elem_1[:, agent, :]],
                                           non_final_next_actions[agent], agents_next_hidden_state[:, agent, :])[0].squeeze()
            elif use_LSTM_flag:
                # lstm_target_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_target_critic_hidden is None else lstm_target_critic_hidden
                lstm_target_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                lstm_target_critic_hidden = (lstm_target_critic_hidden[0].detach(), lstm_target_critic_hidden[1].detach())

                next_target_critic_value, lstm_target_critic_hidden = \
                    self.critics_target[agent]([next_stacked_elem_0[:, agent, :].unsqueeze(1),
                                                next_stacked_elem_1[:, agent, :].unsqueeze(1)],
                                               non_final_next_actions[agent][0], lstm_target_critic_hidden)
            else:
                next_target_critic_value = self.critics_target[agent]([next_stacked_elem_0[:,agent,:],
                                                                       next_stacked_elem_1[:,agent,:]], non_final_next_actions[agent]).squeeze()

            tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1 - dones_stacked[:, agent].unsqueeze(1).unsqueeze(1))
            reward_cal = reward_batch[:, agent].clone()
            target_Q = (reward_batch[:, agent].unsqueeze(1).unsqueeze(1)) + \
                       self.GAMMA * next_target_critic_value * (1 - dones_stacked[:, agent].unsqueeze(1).unsqueeze(1))
            if not use_LSTM_flag:
                target_Q = target_Q.unsqueeze(1)
            tar_Q_after_rew = target_Q.clone()

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            cal_loss_Q = loss_Q.clone()
            single_eps_critic_cal_record.append([tar_Q_before_rew.detach().cpu().numpy(),
                                                 reward_cal.detach().cpu().numpy(),
                                                 tar_Q_after_rew.detach().cpu().numpy(),
                                                 cal_loss_Q.detach().cpu().numpy(),
                                                 (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
                                                 (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
                                                 (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
                                                 (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max())])
            self.critic_optimizer[agent].zero_grad()
            # if use_LSTM_flag:
            #     loss_Q.backward(retain_graph=True)
            # else:
            loss_Q.backward()
            # self.has_gradients(self.critics[agent], agent, wandb)
            # for name, param in self.critics[agent].named_parameters():
            #     if param.grad is not None:
            #         wandb.log({f"actor/{agent}_/{name}_gradients_histogram": wandb.Histogram(param.grad.cpu().detach().numpy())})

            self.critic_optimizer[agent].step()

            if use_GRU_flag:
                action_i = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],agents_cur_hidden_state[:, agent, :])[0]
                actor_loss = - self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                     action_i, agents_cur_hidden_state[:, agent, :])[0].mean()
            elif use_LSTM_flag:
                # lstm_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_actor_hidden is None else lstm_actor_hidden
                # lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_critic_hidden is None else lstm_critic_hidden
                lstm_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)

                lstm_actor_hidden = (lstm_actor_hidden[0].detach(), lstm_actor_hidden[1].detach())
                lstm_critic_hidden = (lstm_critic_hidden[0].detach(), lstm_critic_hidden[1].detach())

                action_i, lstm_actor_hidden = self.actors[agent]([stacked_elem_0[:, agent, :].unsqueeze(1), stacked_elem_1[:, agent, :].unsqueeze(1)],
                                              lstm_actor_hidden)
                q, lstm_critic_hidden = self.critics[agent]([stacked_elem_0[:, agent, :].unsqueeze(1), stacked_elem_1[:, agent, :].unsqueeze(1)],
                                                     action_i, lstm_critic_hidden)
                actor_loss = -torch.mean(q)
            else:
                action_i = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]])
                actor_loss = - self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], action_i).mean()

            # actor_loss = -self.critics[agent](stacked_elem_0[:,agent,:], ac[:, agent, :], agents_cur_hidden_state[:, agent, :])[0].mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            # for name, param in self.actors[agent].named_parameters():
            #     if param.grad is not None:
            #         wandb.log({f"actor/{agent}_/{name}_gradients_histogram": wandb.Histogram(param.grad.cpu().detach().numpy())})
            # torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            # self.has_gradients(self.actors[agent], agent, wandb)  # Replace with your actor network variable
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # if total_step_count % UPDATE_EVERY == 0:  # every "UPDATE_EVERY" step, do a soft update
        #     for i in range(self.n_agents):
        #         print("all agents NN update at total step {}".format(total_step_count))
        #         soft_update(self.critics_target[i], self.critics[i], self.tau)
        #         soft_update(self.actors_target[i], self.actors[i], self.tau)

        if i_episode % UPDATE_EVERY == 0:  # perform a soft update at each step of an episode.
            for i in range(self.n_agents):
                # print("all agents NN update at episode {}".format(i_episode))
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss, single_eps_critic_cal_record

    def update_myown(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, action, wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):

        self.train_num = i_episode

        if len(self.memory) <= self.batch_size:
        # if True:
            return None, None, single_eps_critic_cal_record

        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

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

        # use the stacked tensors
        # current_state in the form of list of length of agents in the environments, then, batchNo X individual Feature length
        # cur_state_list1 = [stacked_elem_0[:, i, :] for i in range(self.n_agents)]

        # for next state
        next_stacked_elem_0 = torch.stack([elem[0] for elem in batch.next_states]).to(device)
        next_stacked_elem_1 = torch.stack([elem[1] for elem in batch.next_states]).to(device)
        if full_observable_critic_flag == True:
            next_stacked_elem_0_combine = next_stacked_elem_0.view(self.batch_size, -1)
            next_stacked_elem_1_combine = next_stacked_elem_1.view(self.batch_size, -1)

        # for done
        dones_stacked = torch.stack([three_agent_dones for three_agent_dones in batch.dones]).to(device)

        for agent in range(self.n_agents):
            # whole_ownState = stacked_elem_0_combine  # own_state only

            # non_final_next_states_actorin = [next_stacked_elem_0]  # 2 portion available
            non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1]  # 2 portion available

            # configured for target Q

            whole_curren_action = action_batch.view(self.batch_size, -1)

            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], history_batch[:,:,i,:])[0] for i in range(self.n_agents)]
            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], agents_next_hidden_state[:,i,:])[0] for i in range(self.n_agents)]

            # non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][:,i,:], non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]

            # non_final_next_combine_actions = torch.stack(non_final_next_actions).view(self.batch_size, -1)

            # get current Q-estimate, using agent's critic network
            # current_Q = self.critics[agent](whole_state, whole_action, whole_agent_combine_gru)
            # current_Q = self.critics[agent](whole_state, whole_action, history_batch[:, :, agent, :])
            if use_GRU_flag:
                non_final_next_actions = [self.actors_target[i](
                    [non_final_next_states_actorin[0][:, i, :], non_final_next_states_actorin[1][:, i, :]],
                    agents_next_hidden_state[:, i, :])[0] for i in range(self.n_agents)]
                current_Q = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], action_batch[:, agent, :], agents_cur_hidden_state[:, agent, :])[0]
            else:
                non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][:,i,:], non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]
                current_Q = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                action_batch[:, agent, :])

            # if full_observable_critic_flag:
            #     current_Q = self.critics[agent]([stacked_elem_0_combine, stacked_elem_1_combine], whole_curren_action)
            # else:
            #     current_Q = self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], action_batch[:,agent,:])

            # has_positive_values = (current_Q > 0).any()
            # if has_positive_values:
            #     print("true")
            with T.no_grad():
                # next_target_critic_value = self.critics_target[agent](next_stacked_elem_0_combine, non_final_next_actions.view(-1,self.n_agents * self.n_actions), whole_agent_combine_gru).squeeze()
                # next_target_critic_value = self.critics_target[agent](next_stacked_elem_0_combine, non_final_next_actions.view(-1,self.n_agents * self.n_actions), history_batch[:, :, agent, :]).squeeze()
                # if full_observable_critic_flag:
                #     next_target_critic_value = self.critics_target[agent](
                #         [next_stacked_elem_0_combine, next_stacked_elem_1_combine],
                #         non_final_next_combine_actions).squeeze()
                #
                # else:
                #     next_target_critic_value = self.critics_target[agent](
                #         [next_stacked_elem_0[:, agent, :], next_stacked_elem_1[:, agent, :]],
                #         non_final_next_actions[agent]).squeeze()
                if use_GRU_flag:
                    next_target_critic_value = \
                    self.critics_target[agent]([next_stacked_elem_0[:, agent, :], next_stacked_elem_1[:, agent, :]],
                                               non_final_next_actions[agent], agents_next_hidden_state[:, agent, :])[
                        0].squeeze()
                else:
                    next_target_critic_value = self.critics_target[agent]([next_stacked_elem_0[:,agent,:], next_stacked_elem_1[:,agent,:]], non_final_next_actions[agent]).squeeze()

                tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1-dones_stacked[:, agent])
                reward_cal = reward_batch[:, agent].clone()
                target_Q = (reward_batch[:, agent]) + (self.GAMMA * next_target_critic_value * (1-dones_stacked[:, agent]))
                target_Q = target_Q.unsqueeze(1)
                tar_Q_after_rew = target_Q.clone()

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            cal_loss_Q = loss_Q.clone()
            single_eps_critic_cal_record.append([tar_Q_before_rew.detach().cpu().numpy(),
                                                 reward_cal.detach().cpu().numpy(),
                                                 tar_Q_after_rew.detach().cpu().numpy(),
                                                 cal_loss_Q.detach().cpu().numpy(),
                                                 (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
                                                 (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
                                                 (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
                                                 (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max())])
            self.critic_optimizer[agent].zero_grad()
            # loss_Q.backward(retain_graph=True)
            loss_Q.backward()
            # self.has_gradients(self.critics[agent], agent, wandb)
            # for name, param in self.critics[agent].named_parameters():
            #     if param.grad is not None:
            #         wandb.log({f"actor/{agent}_/{name}_gradients_histogram": wandb.Histogram(param.grad.cpu().detach().numpy())})

            self.critic_optimizer[agent].step()

            if use_GRU_flag:
                action_i = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                              agents_cur_hidden_state[:, agent, :])[0]
            else:
                action_i = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]])

            ac = action_batch.clone()

            ac[:, agent, :] = action_i.squeeze(0)  # replace the actor from self.actors[agent] into action batch
            combine_action_action_replaced = ac.view(self.batch_size, -1)

            # actor_loss = -self.critics[agent](whole_state, whole_action_action_replaced, whole_hs).mean()
            # actor_loss = 3-self.critics[agent](whole_state, whole_action_action_replaced, whole_agent_combine_gru).mean()
            # if full_observable_critic_flag:
            #     actor_loss = 3 - self.critics[agent]([stacked_elem_0_combine, stacked_elem_1_combine], combine_action_action_replaced).mean()
            #     # actor_loss = - self.critics[agent]([stacked_elem_0_combine, stacked_elem_1_combine], combine_action_action_replaced).mean()
            # else:
            #     actor_loss = 3 - self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
            #                                          ac[:, agent, :]).mean()
            if use_GRU_flag:
                actor_loss = 3 - self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                     ac[:, agent, :], agents_cur_hidden_state[:, agent, :])[0].mean()
            else:
                actor_loss = 3-self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], ac[:, agent, :]).mean()

            # actor_loss = -self.critics[agent](stacked_elem_0[:,agent,:], ac[:, agent, :], agents_cur_hidden_state[:, agent, :])[0].mean()
            self.actor_optimizer[agent].zero_grad()
            actor_loss.backward()
            # for name, param in self.actors[agent].named_parameters():
            #     if param.grad is not None:
            #         wandb.log({f"actor/{agent}_/{name}_gradients_histogram": wandb.Histogram(param.grad.cpu().detach().numpy())})
            # torch.nn.utils.clip_grad_norm_(self.actors[agent].parameters(), 1)
            # self.has_gradients(self.actors[agent], agent, wandb)  # Replace with your actor network variable
            self.actor_optimizer[agent].step()

            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        # if total_step_count % UPDATE_EVERY == 0:  # every "UPDATE_EVERY" step, do a soft update
        #     for i in range(self.n_agents):
        #         print("all agents NN update at total step {}".format(total_step_count))
        #         soft_update(self.critics_target[i], self.critics[i], self.tau)
        #         soft_update(self.actors_target[i], self.actors[i], self.tau)

        if i_episode % UPDATE_EVERY == 0:  # perform a soft update at each step of an episode.
            for i in range(self.n_agents):
                # print("all agents NN update at episode {}".format(i_episode))
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss, single_eps_critic_cal_record

    def has_gradients(self, model, agent, wandb=None):
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                print(f"Numerical instability detected in {name}")
            if param.grad is None:
                print(f"No gradient for {name}")
            else:
                # print(f"Gradient for {name} is {param.grad.norm()}")
                # wandb.log(
                #     {f"actor/{agent}_/{name}_gradients_histogram": wandb.Histogram(param.grad.cpu().detach().numpy())})
                wandb.log({name: float(param.grad.norm())})
                # wandb.log({'agent' + str(idx): wandb.Histogram(param.grad.cpu().detach().numpy())})

    def choose_action(self, state, cur_total_step, cur_episode, step, total_training_steps, noise_start_level, actor_hiddens, lstm_hist, use_LSTM_flag, noisy=True, use_GRU_flag=False):
        # ------------- MADDPG_test_181123_10_10_54 version noise -------------------
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        noise_value = np.zeros(2)
        if lstm_hist is None:
            lstm_hist = self.init_hidden(self.actors[0].rnn_hidden_dim, device=device)

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
            elif use_LSTM_flag:
                act, lstm_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], lstm_hist)
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
            elif use_LSTM_flag:  # don't change act_hn in this loop
                pass
            else:
                act_hn[i, :] = torch.zeros(1, self.n_actions)
            self.actors[i].train()
        self.steps_done += 1
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
        if use_LSTM_flag:
            return actions.data.cpu().numpy(), noise_value, lstm_hist, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()
        else:
            return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        # return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        # return actions.data.cpu().numpy(), noise_value

    def get_custom_linear_scaling_factor(self, episode, eps_end, start_scale=1, end_scale=0.05):
        # Calculate the slope of the linear decrease only up to eps_end
        if episode <= eps_end:
            slope = (end_scale - start_scale) / (eps_end - 1)
            current_scale = start_scale + slope * (episode - 1)
        else:
            current_scale = end_scale
        return current_scale

    def linear_decay(self, cur_total_step, total_training_steps, start_scale=1, end_scale=0.03):
        current_scale = start_scale
        # Calculate the slope of the linear decrease only up to eps_end
        if cur_total_step == 0:
            return current_scale
        if current_scale > end_scale:
            eps_delta = (start_scale - end_scale) / total_training_steps
            current_scale = start_scale - eps_delta
        else:
            current_scale = end_scale
        return current_scale

    def init_hidden(self, hidden_size, batch_size=1, device='cpu'):
        # for lstm initialization is follow with (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(1, batch_size, hidden_size).to(device)
        c0 = torch.zeros(1, batch_size, hidden_size).to(device)
        return (h0, c0)