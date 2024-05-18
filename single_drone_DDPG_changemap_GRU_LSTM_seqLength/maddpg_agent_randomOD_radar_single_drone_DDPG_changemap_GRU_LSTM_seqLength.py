# from Nnetworks_MADDPGv3 import CriticNetwork_0724, ActorNetwork
from Nnetworks_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength import *
import torch
from copy import deepcopy
from torch.optim import Adam
from memory_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength import ReplayMemory, Experience
# from random_process_MADDPGv3_randomOD import OrnsteinUhlenbeckProcess
from torch.autograd import Variable
import os
import torch.nn as nn
import time
import numpy as np
import torch as T
from utils_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength import device
from Utilities_own_randomOD_radar_single_drone_DDPG_changemap_GRU_LSTM_seqLength import *
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
    def __init__(self, actor_dim, critic_dim, dim_act, actor_hidden_state_size, gru_history_length, n_agents, args, cr_lr, ac_lr, gamma, tau, full_observable_critic_flag, use_GRU_flag, use_attention_flag, attention_only, use_LSTM_flag, stacking, feature_matching):
        self.args = args
        self.mode = args.mode
        self.actors = []
        self.critics = []
        self.random_network = RandomNetwork(actor_dim, 0.1)
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
                if feature_matching:
                    self.actors = [GRU_batch_actor_TwoPortionFM(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
                    self.critics = [critic_single_obs_GRU_batch_twoPortionFM(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]
                else:
                    self.actors = [GRU_batch_actor_TwoPortion(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
                    self.critics = [critic_single_obs_GRU_batch_twoPortion(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for _ in range(n_agents)]

        elif use_LSTM_flag:
            self.actors = [LSTM_batch_actor_TwoPortion(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]  # use deterministic policy
            self.critics = [critic_single_obs_LSTM_batch_twoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                                    actor_hidden_state_size) for _ in range(n_agents)]

        elif attention_only:
            self.actors = [actor_TwoPortion_wATT(actor_dim, dim_act) for _ in
                           range(n_agents)]  # use deterministic policy
            self.critics = [critic_single_obs_TwoPortion_wATT(critic_dim, n_agents, dim_act) for _ in range(n_agents)]

        # elif stacking:
        elif stacking:
            self.actors = [GRU_LSTM_stacking_TwoPortion(actor_dim, dim_act, actor_hidden_state_size) for _ in range(n_agents)]
            self.critics = [
                critic_single_GRU_LSTM_stacking_TwoPortion(critic_dim, n_agents, dim_act, gru_history_length,
                                                                    actor_hidden_state_size) for
                _ in range(n_agents)]
        else:
            if feature_matching:
                self.actors = [ActorNetwork_TwoPortionFM(actor_dim, dim_act) for _ in range(n_agents)]  # use deterministic policy
                self.critics = [
                    critic_single_TwoPortionFM(critic_dim, n_agents, dim_act, gru_history_length, actor_hidden_state_size) for
                    _ in range(n_agents)]
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

        self.memory = ReplayMemory(args.memory_length, gru_history_length)
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

        # self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr, weight_decay=1e-4) for x in self.critics]
        # self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr, weight_decay=1e-4) for x in self.actors]

        self.critic_optimizer = [Adam(x.parameters(), lr=cr_lr, weight_decay=1e-5) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(), lr=ac_lr, weight_decay=1e-5) for x in self.actors]

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

    def update_myown_ddpg(self, i_episode, total_step_count, UPDATE_EVERY, single_eps_critic_cal_record, action, use_LSTM_flag, stacking, feature_matching, wandb=None, full_observable_critic_flag=False, use_GRU_flag=False):
        self.train_num = i_episode
        if use_LSTM_flag or use_GRU_flag or stacking:
            if len(self.memory) <= self.batch_size*self.memory.history_seq_length:
                return None, None, single_eps_critic_cal_record
        else:
            if len(self.memory) <= self.batch_size:
                return None, None, single_eps_critic_cal_record
        # print("------------------ Training starts ------------------------")
        BoolTensor = torch.cuda.BoolTensor if self.use_cuda else torch.BoolTensor
        FloatTensor = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor

        c_loss = []
        a_loss = []

        if use_LSTM_flag or use_GRU_flag or stacking:
            # sequence_indexes = torch.arange(0, self.memory.history_seq_length, 1)
            # size = len(self.memory)
            # size -= sequence_indexes[-1].item()
            # indexes = torch.randperm(size, dtype=torch.long)[:self.batch_size]
            # indexes = (sequence_indexes.repeat(indexes.shape[0], 1) + indexes.view(-1, 1)).view(-1)
            # transitions = self.memory.sample_by_index(indexes)
            transitions = self.memory.sample(self.batch_size*self.memory.history_seq_length)
        else:
            transitions = self.memory.sample(self.batch_size)

        # transitions = self.memory.sample(self.batch_size)
        batch = Experience(*zip(*transitions))

        action_batch = torch.stack(batch.actions).type(FloatTensor)
        reward_batch = torch.stack(batch.rewards).type(FloatTensor)
        # for done
        dones_stacked = torch.stack([three_agent_dones for three_agent_dones in batch.dones]).to(device)
        if use_GRU_flag:
            stacked_hidden = torch.empty((self.batch_size*self.memory.history_seq_length, *batch.gru_hidden[0].squeeze(0).shape), device=device) # avoid using stack
            for i, elem in enumerate(batch.gru_hidden):
                stacked_hidden[i] = elem.squeeze(0)
            stacked_hidden = stacked_hidden.contiguous().view(1, -1, self.memory.history_seq_length, stacked_hidden.shape[-1]).detach()  # (D * num_layers, N, L, Hout)

        if use_LSTM_flag:
            stacked_hidden = torch.stack([elem[0] for elem in batch.lstm_hidden]).to(device)  # lstm_hidden, (N, num_layer, L, Hout)
            stacked_cell = torch.stack([elem[1] for elem in batch.lstm_hidden]).to(device) # lstm_cell, (N, num_layer, L, Hout)
            stacked_hidden = stacked_hidden.view(1, -1, self.memory.history_seq_length, stacked_hidden.shape[-1])  # (D * num_layers, N, L, Hout)
            stacked_hidden = stacked_hidden.detach()
            cell_states = stacked_cell.view(1, -1, self.memory.history_seq_length, stacked_cell.shape[-1])  # (D * num_layers, N, L, Hcell)
            cell_states = cell_states.detach()

        if stacking:
            lstm_stacked_hidden = torch.stack([elem[0] for elem in batch.lstm_hidden]).to(device)  # lstm_hidden, (N, num_layer, L, Hout)
            lstm_stacked_cell = torch.stack([elem[1] for elem in batch.lstm_hidden]).to(device) # lstm_cell, (N, num_layer, L, Hout)
            lstm_stacked_hidden = lstm_stacked_hidden.view(1, -1, self.memory.history_seq_length, lstm_stacked_hidden.shape[-1])  # (D * num_layers, N, L, Hout)
            lstm_stacked_hidden = lstm_stacked_hidden.detach()
            lstm_stacked_cell = lstm_stacked_cell.view(1, -1, self.memory.history_seq_length, lstm_stacked_cell.shape[-1])  # (D * num_layers, N, L, Hcell)
            lstm_stacked_cell = lstm_stacked_cell.detach()

            gru_stacked_hidden = torch.empty((self.batch_size*self.memory.history_seq_length, *batch.gru_hidden[0].squeeze(0).shape), device=device) # avoid using stack
            for i, elem in enumerate(batch.gru_hidden):
                gru_stacked_hidden[i] = elem.squeeze(0)
            gru_stacked_hidden = gru_stacked_hidden.contiguous().view(1, -1, self.memory.history_seq_length, gru_stacked_hidden.shape[-1]).detach()  # (D * num_layers, N, L, Hout)


        if use_LSTM_flag or use_GRU_flag or stacking:
            # stack tensors only once
            stacked_elem_0 = torch.empty((self.batch_size * self.memory.history_seq_length, *batch.states[0][0].shape),
                                         device=device)
            stacked_elem_1 = torch.empty((self.batch_size * self.memory.history_seq_length, *batch.states[0][1].shape),
                                         device=device)
            for i, elem in enumerate(batch.states):
                stacked_elem_0[i] = elem[0]
                stacked_elem_1[i] = elem[1]
            stacked_elem_0 = stacked_elem_0.contiguous().view(-1, self.memory.history_seq_length, stacked_elem_0.shape[-1])
            stacked_elem_1 = stacked_elem_1.contiguous().view(-1, self.memory.history_seq_length, stacked_elem_1.shape[-1])
        else:
            stacked_elem_0 = torch.stack([elem[0] for elem in batch.states]).to(device)
            stacked_elem_1 = torch.stack([elem[1] for elem in batch.states]).to(device)

        if full_observable_critic_flag == True:
            stacked_elem_0_combine = stacked_elem_0.view(self.batch_size, -1)  # own_state only
            stacked_elem_1_combine = stacked_elem_1.view(self.batch_size, -1)  # own_state only

        # use the stacked tensors
        # current_state in the form of list of length of agents in the environments, then, batchNo X individual Feature length
        # cur_state_list1 = [stacked_elem_0[:, i, :] for i in range(self.n_agents)]


        if use_LSTM_flag or use_GRU_flag or stacking:
            # for next state
            next_stacked_elem_0 = torch.empty(
                (self.batch_size * self.memory.history_seq_length, *batch.next_states[0][0].shape), device=device)
            next_stacked_elem_1 = torch.empty(
                (self.batch_size * self.memory.history_seq_length, *batch.next_states[0][1].shape), device=device)
            for i, elem in enumerate(batch.next_states):
                next_stacked_elem_0[i] = elem[0]
                next_stacked_elem_1[i] = elem[1]
            next_stacked_elem_0 = next_stacked_elem_0.contiguous().view(-1, self.memory.history_seq_length, next_stacked_elem_0.shape[-1])
            next_stacked_elem_1 = next_stacked_elem_1.contiguous().view(-1, self.memory.history_seq_length, next_stacked_elem_1.shape[-1])
        else:
            next_stacked_elem_0 = torch.stack([elem[0] for elem in batch.next_states]).to(device)
            next_stacked_elem_1 = torch.stack([elem[1] for elem in batch.next_states]).to(device)



        if full_observable_critic_flag == True:
            next_stacked_elem_0_combine = next_stacked_elem_0.view(self.batch_size, -1)
            next_stacked_elem_1_combine = next_stacked_elem_1.view(self.batch_size, -1)

        for agent in range(self.n_agents):
            # whole_ownState = stacked_elem_0_combine  # own_state only

            # non_final_next_states_actorin = [next_stacked_elem_0]  # 2 portion available
            non_final_next_states_actorin = [next_stacked_elem_0, next_stacked_elem_1]  # 2 portion available

            # configured for target Q
            if self.memory.history_seq_length != 0:
                # whole_curren_action = action_batch.contiguous().view(len(self.memory.sampling_indexes), -1)
                whole_curren_action = action_batch.squeeze(1)
            else:
                whole_curren_action = action_batch.contiguous().view(self.batch_size, -1)

            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], history_batch[:,:,i,:])[0] for i in range(self.n_agents)]
            # non_final_next_actions = [self.actors_target[i](non_final_next_states_actorin[0][:,i,:], agents_next_hidden_state[:,i,:])[0] for i in range(self.n_agents)]

            # non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][:,i,:], non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]

            # non_final_next_combine_actions = torch.stack(non_final_next_actions).view(self.batch_size, -1)

            # get current Q-estimate, using agent's critic network
            # current_Q = self.critics[agent](whole_state, whole_action, whole_agent_combine_gru)
            # current_Q = self.critics[agent](whole_state, whole_action, history_batch[:, :, agent, :])
            with torch.no_grad():
                if use_GRU_flag:
                    non_final_next_actions = [self.actors_target[i](
                        [non_final_next_states_actorin[0], non_final_next_states_actorin[1]],
                        stacked_hidden, 1, dones_stacked) for i in range(self.n_agents)]
                elif use_LSTM_flag:
                    # lstm_target_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                    # lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                    # lstm_target_actor_hidden = (lstm_target_actor_hidden[0].detach(), lstm_target_actor_hidden[1].detach())
                    # lstm_critic_hidden = (lstm_critic_hidden[0].detach(), lstm_critic_hidden[1].detach())
                    non_final_next_actions = [self.actors_target[i](
                        [non_final_next_states_actorin[0], non_final_next_states_actorin[1]],
                        (stacked_hidden, cell_states), 1, dones_stacked) for i in range(self.n_agents)]  # target network flag is 1.
                elif stacking:
                    non_final_next_actions = [self.actors_target[i](
                        [non_final_next_states_actorin[0], non_final_next_states_actorin[1]],
                        gru_stacked_hidden, (lstm_stacked_hidden,lstm_stacked_cell), 1, dones_stacked) for i in range(self.n_agents)]
                else:
                    if feature_matching:
                        non_final_next_actions = [self.actors_target[i](
                            [non_final_next_states_actorin[0][:, i, :], non_final_next_states_actorin[1][:, i, :]])[0] for
                                                  i in range(self.n_agents)]

                    else:
                        non_final_next_actions = [self.actors_target[i]([non_final_next_states_actorin[0][:,i,:], non_final_next_states_actorin[1][:,i,:]]) for i in range(self.n_agents)]

                if use_GRU_flag:
                    next_target_critic_value, _ = \
                    self.critics_target[agent]([next_stacked_elem_0, next_stacked_elem_1],
                                               non_final_next_actions[agent][0], stacked_hidden, 1, dones_stacked)
                elif use_LSTM_flag:
                    # lstm_target_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_target_critic_hidden is None else lstm_target_critic_hidden
                    # lstm_target_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                    # lstm_target_critic_hidden = (lstm_target_critic_hidden[0].detach(), lstm_target_critic_hidden[1].detach())
                    next_target_critic_value, _ = \
                        self.critics_target[agent]([next_stacked_elem_0, next_stacked_elem_1],
                                                   non_final_next_actions[agent][0], (stacked_hidden, cell_states), 1, dones_stacked)
                elif stacking:
                    next_target_critic_value, _, _ = \
                        self.critics_target[agent]([next_stacked_elem_0, next_stacked_elem_1],
                                                   non_final_next_actions[agent][0], gru_stacked_hidden, (lstm_stacked_hidden,lstm_stacked_cell), 1, dones_stacked)
                else:
                    if feature_matching:
                        next_target_critic_value = self.critics_target[agent]([next_stacked_elem_0[:,agent,:],
                                                                               next_stacked_elem_1[:,agent,:]], non_final_next_actions[agent])[0].squeeze()
                    else:
                        next_target_critic_value = self.critics_target[agent]([next_stacked_elem_0[:,agent,:],
                                                                               next_stacked_elem_1[:,agent,:]], non_final_next_actions[agent]).squeeze()
                if use_LSTM_flag or use_GRU_flag or stacking:
                    tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1 - dones_stacked[:, agent].unsqueeze(1))
                    reward_cal = reward_batch[:, agent].clone()
                    target_Q = (reward_batch[:, agent].unsqueeze(1)) + \
                               self.GAMMA * next_target_critic_value * (
                                           1 - dones_stacked[:, agent].unsqueeze(1))
                else:
                    tar_Q_before_rew = self.GAMMA * next_target_critic_value * (1 - dones_stacked[:, agent].unsqueeze(1).unsqueeze(1))
                    reward_cal = reward_batch[:, agent].clone()
                    target_Q = (reward_batch[:, agent].unsqueeze(1)) + \
                               self.GAMMA * next_target_critic_value.unsqueeze(1) * (1 - dones_stacked[:, agent].unsqueeze(1))
                # if not use_GRU_flag and not use_LSTM_flag:
                #     target_Q = target_Q.unsqueeze(1)  # only run when both flag are false
                tar_Q_after_rew = target_Q.clone()
            if use_LSTM_flag:
                current_Q, _ = self.critics[agent]([stacked_elem_0, stacked_elem_1], action_batch,
                                                   (stacked_hidden, cell_states), 0, dones_stacked)
            elif use_GRU_flag:
                current_Q, _ = self.critics[agent]([stacked_elem_0, stacked_elem_1],
                                                action_batch, stacked_hidden, 0, dones_stacked)
            elif stacking:
                current_Q, _, _ = self.critics[agent]([stacked_elem_0, stacked_elem_1],
                                                action_batch, gru_stacked_hidden, (lstm_stacked_hidden,lstm_stacked_cell), 0, dones_stacked)
            else:
                if feature_matching:
                    current_Q, _ = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                    action_batch[:, agent, :])
                else:
                    current_Q = self.critics[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]],
                                                    action_batch[:, agent, :])

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            cal_loss_Q = loss_Q.clone()
            single_eps_critic_cal_record.append(None)
            # single_eps_critic_cal_record.append([tar_Q_before_rew.detach().cpu().numpy(),
            #                                      reward_cal.detach().cpu().numpy(),
            #                                      tar_Q_after_rew.detach().cpu().numpy(),
            #                                      cal_loss_Q.detach().cpu().numpy(),
            #                                      (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
            #                                      (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
            #                                      (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
            #                                      (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max())])
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
                action_i, _ = self.actors[agent]([stacked_elem_0, stacked_elem_1], stacked_hidden, 0, dones_stacked)
                q, _ = self.critics[agent]([stacked_elem_0, stacked_elem_1], action_i, stacked_hidden, 0, dones_stacked)
                actor_loss = -torch.mean(q)
            elif use_LSTM_flag:
                # lstm_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_actor_hidden is None else lstm_actor_hidden
                # lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device) if lstm_critic_hidden is None else lstm_critic_hidden
                # lstm_actor_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)
                # lstm_critic_hidden = self.init_hidden(self.actors[0].rnn_hidden_dim, batch_size=self.batch_size, device=device)

                # lstm_actor_hidden = (lstm_actor_hidden[0].detach(), lstm_actor_hidden[1].detach())
                # lstm_critic_hidden = (lstm_critic_hidden[0].detach(), lstm_critic_hidden[1].detach())

                action_i, _ = self.actors[agent]([stacked_elem_0, stacked_elem_1],
                                              (stacked_hidden, cell_states), 0, dones_stacked)
                q, _ = self.critics[agent]([stacked_elem_0, stacked_elem_1],
                                                     action_i, (stacked_hidden, cell_states), 0, dones_stacked)
                actor_loss = -torch.mean(q)
            elif stacking:
                action_i, _, _ = self.actors[agent]([stacked_elem_0, stacked_elem_1], gru_stacked_hidden, (lstm_stacked_hidden,lstm_stacked_cell), 0, dones_stacked)
                q, _, _ = self.critics[agent]([stacked_elem_0, stacked_elem_1], action_i, gru_stacked_hidden, (lstm_stacked_hidden,lstm_stacked_cell), 0, dones_stacked)
                actor_loss = -torch.mean(q)
            else:
                if feature_matching:
                    action_i, _ = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]])
                    actor_loss = - self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], action_i)[0].mean()

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

            if feature_matching:
                _, actor_features_random = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], use_random=True)
                _, actor_features_clean = self.actors[agent]([stacked_elem_0[:, agent, :], stacked_elem_1[:, agent, :]], use_random=False)
                fm_loss_actor = self.compute_feature_matching_loss(actor_features_random, actor_features_clean)

                _, critic_feature_random = self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], action_i, use_random=True)
                _, critic_feature_clean = self.critics[agent]([stacked_elem_0[:,agent,:], stacked_elem_1[:,agent,:]], action_i, use_random=False)
                fm_loss_critic = self.compute_feature_matching_loss(critic_feature_random, critic_feature_clean)
                # Total FM loss
                total_fm_loss = 0.002 * (fm_loss_actor + fm_loss_critic)

                self.actor_optimizer[agent].zero_grad()
                self.critic_optimizer[agent].zero_grad()
                total_fm_loss.backward()
                self.actor_optimizer[agent].step()
                self.critic_optimizer[agent].step()
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

    def compute_feature_matching_loss(self, random_features, clean_features):
        return nn.functional.mse_loss(random_features, clean_features)

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
            single_eps_critic_cal_record.append(None)
            # single_eps_critic_cal_record.append([tar_Q_before_rew.detach().cpu().numpy(),
            #                                      reward_cal.detach().cpu().numpy(),
            #                                      tar_Q_after_rew.detach().cpu().numpy(),
            #                                      cal_loss_Q.detach().cpu().numpy(),
            #                                      (tar_Q_before_rew.detach().cpu().numpy().min(), tar_Q_before_rew.detach().cpu().numpy().max()),
            #                                      (reward_cal.detach().cpu().numpy().min(), reward_cal.detach().cpu().numpy().max()),
            #                                      (tar_Q_after_rew.detach().cpu().numpy().min(), tar_Q_after_rew.detach().cpu().numpy().max()),
            #                                      (cal_loss_Q.detach().cpu().numpy().min(), cal_loss_Q.detach().cpu().numpy().max())])
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

    def choose_action(self, OU_noise, state, cur_total_step, cur_episode, step, total_training_steps, noise_start_level, actor_hiddens, lstm_hist, gru_hist, use_LSTM_flag, stacking, feature_matching, noisy=True, use_GRU_flag=False):
        # ------------- MADDPG_test_181123_10_10_54 version noise -------------------
        obs = torch.from_numpy(np.stack(state[0])).float().to(device)
        obs_grid = torch.from_numpy(np.stack(state[1])).float().to(device)
        noise_value = np.zeros(2)
        if use_LSTM_flag and lstm_hist is None:
            lstm_hist = self.init_hidden(self.actors[0].rnn_hidden_dim, device=device)
        if use_GRU_flag and gru_hist is None:
            gru_hist = self.init_hidden_gru(self.actors[0].rnn_hidden_dim, device=device)
        if stacking and lstm_hist is None and gru_hist is None:
            lstm_hist = self.init_hidden(512, device=device)
            gru_hist = self.init_hidden_gru(self.actors[0].rnn_hidden_dim, device=device)

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
            # self.actors[i].eval()
            if use_GRU_flag:
                act, gru_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], gru_hist)
            elif use_LSTM_flag:
                act, lstm_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], lstm_hist)
            elif stacking:
                act, gru_hist, lstm_hist = self.actors[i]([sb.unsqueeze(0).unsqueeze(0), sb_grid.unsqueeze(0).unsqueeze(0)], gru_hist, lstm_hist)
            else:
                if feature_matching:
                    act, _ = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)], use_random=False)
                else:
                    act = self.actors[i]([sb.unsqueeze(0), sb_grid.unsqueeze(0)])
            if noisy:
                noise_value = np.random.randn(2) * self.var[i]  # gaussian
                # noise_value = OU_noise() * self.var[i]  # OU-noise
                act += torch.from_numpy(noise_value).type(FloatTensor)
                # print("Episode {}, agent {}, noise level is {}".format(episode, i, self.var[i]))
                act = torch.clamp(act, -1.0, 1.0)  # when using stochastic policy, we are not require to clamp again.

            actions[i, :] = act
            if use_GRU_flag:
                pass
            elif use_LSTM_flag:  # don't change act_hn in this loop
                pass
            else:
                act_hn[i, :] = torch.zeros(1, self.n_actions)
            # self.actors[i].train()
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
        # if use_LSTM_flag:
        #     return actions.data.cpu().numpy(), noise_value, lstm_hist, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()
        # elif use_GRU_flag:
        #     return actions.data.cpu().numpy(), noise_value, gru_hist, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()
        # else:
        #     return actions.data.cpu().numpy(), noise_value, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()  # NOTE: tensor.data.cpu() is to make the tensor's "is_leaf" = True, this also prevent the error message on line "retain_graph=True"
        return actions.data.cpu().numpy(), noise_value, lstm_hist, gru_hist, gru_history_input.squeeze(0).data.cpu(), act_hn.data.cpu()
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

    def init_hidden_gru(self, hidden_size, batch_size=1, device='cpu'):
        # for gru initialization is follow with (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(1, batch_size, hidden_size).to(device)
        return h0