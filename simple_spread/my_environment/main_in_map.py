import argparse
import datetime
import os
import sys
sys.path.insert(0, 'F:\githubClone\Multi_agent_AAC\simple_spread\multiagent-particle-envs')
import time
import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from multiagent import scenarios
from multiagent.environment import MultiAgentEnv

from MADDPG import MADDPG
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
matplotlib.use('TkAgg')


def get_running_reward(reward_array: np.ndarray, window=100):
    """calculate the running reward, i.e. average of last `window` elements from rewards"""
    running_reward = np.zeros_like(reward_array)
    for i in range(window - 1):
        running_reward[i] = np.mean(reward_array[:i + 1])
    for i in range(window - 1, len(reward_array)):
        running_reward[i] = np.mean(reward_array[i - window + 1:i + 1])
    return running_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env', default='simple_spread', type=str, help='name of the environment',
                        choices=['simple_adversary', 'simple_crypto', 'simple_push', 'simple_reference',
                                 'simple_speaker_listener', 'simple_spread', 'simple_tag','simple_spread_with_obstacle','simple_spread_in_map',
                                 'simple_world_comm'])
    parser.add_argument('--episode-length', type=int, default=150, help='steps per episode')
    parser.add_argument('--episode-num', type=int, default=300000, help='total number of episode')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument('--buffer-capacity', default=int(1e6))
    parser.add_argument('--batch-size', default=1024)
    parser.add_argument('--actor-lr', type=float, default=1e-2, help='learning rate of actor')
    parser.add_argument('--critic-lr', type=float, default=1e-2, help='learning rate of critic')
    parser.add_argument('--steps-before-learn', type=int, default=5e4,
                        help='steps to be executed before agents start to learn')
    parser.add_argument('--learn-interval', type=int, default=100,
                        help='maddpg will only learn every this many steps')
    parser.add_argument('--save-interval', type=int, default=5000,
                        help='save model once every time this many episodes are completed')
    parser.add_argument('--tau', type=float, default=0.02, help='soft update parameter')
    args = parser.parse_args()
    start = time.time()

    # create folder to save result
    env_dir = os.path.join('results', args.env)
    if not os.path.exists(env_dir):
        os.makedirs(env_dir)
    total_files = len([file for file in os.listdir(env_dir)])
    res_dir = os.path.join(env_dir, f'{total_files + 1}')
    os.makedirs(res_dir)
    model_dir = os.path.join(res_dir, 'model')
    os.makedirs(model_dir)
    print('----------------常数6------取消合作模式，飞机都飞到目的地后停止，飞机碰撞不停，飞机不动给惩罚----------', model_dir)
    
    # create env
    scenario = scenarios.load(f'{args.env}.py').Scenario()  # created class "Scenario" in simple_spread.py
    world = scenario.make_world()  # call make_world() method in Scenario class in simple_spread.py
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation,None,scenario.is_done)

    # get dimension info about observation and action
    obs_dim_list = []
    for obs_space in env.observation_space:  # continuous observation
        obs_dim_list.append(obs_space.shape[0])  # Box
    act_dim_list = []
    for act_space in env.action_space:  # discrete action
        act_dim_list.append(act_space.n)  # Discrete

    maddpg = MADDPG(obs_dim_list, act_dim_list, args.buffer_capacity, args.actor_lr, args.critic_lr, res_dir)


    '''    
    pretrained_model = 'results/simple_spread_in_map/1119'
    data = torch.load(os.path.join(pretrained_model, 'model.pt'))
    for agent, actor_parameter in zip(maddpg.agents, data):
        agent.actor.load_state_dict(actor_parameter)
    print(f'MADDPG load model.pt from {pretrained_model}')
    '''



    total_step = 0
    total_reward = np.zeros((args.episode_num, env.n))  # reward of each agent in each episode
    for episode in range(args.episode_num):
        obs = env.reset()
        # record reward of each agent in this episode
        # episode_reward = np.zeros((args.episode_length, env.n))
        episode_reward = []
        # print('episode_reward: ',episode_reward,episode_reward.shape)

        for step in range(args.episode_length):  # interact with the env for an episode
            actions = maddpg.select_action(obs)
            step_start_time = time.time()
            next_obs, rewards, dones, infos = env.step(actions)
            print("one step time used is {} milliseconds".format((time.time()-step_start_time)*1000))
            #print(next_obs)
            #episode_reward[step] = rewards
            episode_reward.append(rewards)
            #env.render()
            #time.sleep(0.2)
            total_step += 1

            maddpg.add(obs, actions, rewards, next_obs, dones)
            # only start to learn when there are enough experiences to sample
            if total_step > args.steps_before_learn:
                if total_step % args.learn_interval == 0:
                    maddpg.learn(args.batch_size, args.gamma)
                    maddpg.update_target(args.tau)
                if episode % args.save_interval == 0:
                    torch.save([agent.actor.state_dict() for agent in maddpg.agents],
                               os.path.join(model_dir, f'model_{episode}.pt'))
            obs = next_obs

            '''
            如果有一架飞机坠机了，就停止
            '''
            if True in dones:
                break
            if np.sum(rewards) > 11.5:  # 3*(生存值3+到达奖励1)
                print(f'episode {episode + 1}: success')
                break


        # episode finishes
        # calculate cumulative reward of each agent in this episode
        episode_reward = np.array(episode_reward)

        cumulative_reward = episode_reward.sum(axis=0)
        total_reward[episode] = cumulative_reward
        len_episode_step = len(episode_reward)
        first_step_reward = episode_reward[0]
        last_step_reward = episode_reward[-1]

        if (episode+1) % 1000 ==0:
            print(f'episode {episode + 1}: finished at step {len_episode_step}, the first step reward: {first_step_reward}, the last step reward: {last_step_reward},  cumulative reward: {cumulative_reward}, '
                  f'sum reward: {sum(cumulative_reward)}')
        


        if (episode + 1) % 10000 == 0:
            # plot result
            fig, ax = plt.subplots()
            x = range(1, args.episode_num + 1)
            for agent in range(env.n):
                ax.plot(x, total_reward[:, agent], label=agent)
                ax.plot(x, get_running_reward(total_reward[:, agent]))
            ax.legend()
            ax.set_xlabel('episode')
            ax.set_ylabel('reward')
            title = f'training result of maddpg solve {args.env}'
            ax.set_title(title)
            plt.savefig(os.path.join(res_dir, title))

            print(f'training , time spent: {datetime.timedelta(seconds=int(time.time() - start))}')

    # all episodes performed, training finishes
    # save agent parameters
    torch.save([agent.actor.state_dict() for agent in maddpg.agents], os.path.join(res_dir, 'model.pt'))
    # save training reward
    np.save(os.path.join(res_dir, 'rewards.npy'), total_reward)

    # plot result
    fig, ax = plt.subplots()
    x = range(1, args.episode_num + 1)
    for agent in range(env.n):
        ax.plot(x, total_reward[:, agent], label=agent)
        ax.plot(x, get_running_reward(total_reward[:, agent]))
    ax.legend()
    ax.set_xlabel('episode')
    ax.set_ylabel('reward')
    title = f'training result of maddpg solve {args.env}'
    ax.set_title(title)
    plt.savefig(os.path.join(res_dir, title))

    print(f'training finishes, time spent: {datetime.timedelta(seconds=int(time.time() - start))}')




