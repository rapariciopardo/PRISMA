import tensorflow as tf
import numpy as np
import pandas as pd
import time, datetime

from .learner import DQN_AGENT
from .replay_buffer import ReplayBuffer
from .models import DQN_buffer_model
from .utils import save_model, load_model, LinearSchedule


def save_info_df(csv_filename, episode_info_df, t, num_episodes, train_delay, train_loss, episode_reward, val_delay, val_loss, exploration_value, episode_mean_td_error):
    """Stores the information about the episode in a csv file.

    Args:
        csv_filename (str): name of the csv file.
        episode_info_df (DataFrame): previous episodes dataframe to append.
        t (int): iteration number.
        num_episodes (int): number of episode.
        train_delay (float): train average arrived packet delay.
        train_loss (float): train loss.
        episode_reward (float): episode reward.
        val_delay (float): validation average arrived packet delay.
        val_loss (float): validation loss.
        exploration_value (float): exploration step.
        episode_mean_td_error (float): average episode td error.

    Returns:
        DataFrame: episode dataframe.
    """
    
    info_df = pd.DataFrame({
        'steps': [t],
        'num_episodes': [num_episodes],
        "train_delay": train_delay,
        "train_loss": train_loss,
        'episode_reward': [episode_reward],
        "val_delay": val_delay,
        "val_loss" :val_loss,
        'episode_mean_td_error': [episode_mean_td_error / t],


    })
    episode_info_df = episode_info_df.append(info_df)

    episode_info_df.to_csv(csv_filename)

    return episode_info_df



def update_save_plot_info_df(csv_filename, episode_info_df, info, greedy_info, oracle_info, t, episode_reward, exploration_value, number_of_requests):
    info_df = pd.DataFrame({
        't': [t],
        'total_utility': [info['total_utility']],
        'number_of_skipped_requests': [info['number_of_skipped_requests'] / number_of_requests],
        'number_of_accepted_requests': [info['number_of_accepted_requests'] / number_of_requests],
        'number_of_blocked_requests': [info['number_of_blocked_requests'] / number_of_requests],
        
        'greedy_total_utility': [greedy_info['total_utility']],
        'greedy_number_of_skipped_requests': [greedy_info['number_of_skipped_requests'] / number_of_requests],
        'greedy_number_of_accepted_requests': [greedy_info['number_of_accepted_requests'] / number_of_requests],
        'greedy_number_of_blocked_requests': [greedy_info['number_of_blocked_requests'] / number_of_requests],


    })

    for action in range(len(info['actions_statistics'])):
        info_df[f'action {action} statistics'] = [info['actions_statistics'][action] / number_of_requests]

    for action in range(len(greedy_info['actions_statistics'])):
        info_df[f'greedy_action {action} statistics'] = [
            greedy_info['actions_statistics'][action] / number_of_requests]

    info_df["episode_reward"] = [episode_reward]
    info_df["exploring"] = [exploration_value]

    episode_info_df = episode_info_df.append(info_df)

    episode_info_df.to_csv(csv_filename)

    return episode_info_df


def log_env_info(info, number_of_requests, prefix=''):
    print(prefix + 'total utility', info['total_utility'] )
                          
    print(prefix + 'number of skipped requests',
                          info['number_of_skipped_requests'] / number_of_requests)
    print(prefix + 'number of accepted requests',
                          info['number_of_accepted_requests'] / number_of_requests)
    print(prefix + 'number of blocked requests',
                          info['number_of_blocked_requests'] / number_of_requests)

    for a in range(len(info['actions_statistics'])):
        print(prefix + 'action {} statistics'.format(a),
                              info['actions_statistics'][a] / number_of_requests)


def dqn_decision(obs, n, env, agents):
    action = np.argmin(agents[n](obs))
    return action



def dqn_step_decorator(obs, n, env, agents, stochastic=True, update_eps=-1):
    return agents[n].step(obs, stochastic, update_eps)[0].numpy()
        
     
def learn(env,
          validation_env,
          csv_results_file = "ma_dqn_results.csv",
          seed=None,
          lr=5e-4,
          total_timesteps=100000,
          buffer_size=50000,
          exploration_fraction=0.1,
          exploration_initial_eps=1.00,
          exploration_final_eps=0.02,
          train_freq=1,
          batch_size=32,
          print_freq=100,
          checkpoint_freq=10000,
          checkpoint_path=None,
          storing_starts=1500,
          learning_starts=1000,
          gamma=1.0,
          target_network_update_freq=500,
          load_path=None,
          model_path="saved_model",
          **network_kwargs
            ):
    """

    env: gym.Env
        environment to train on
    network: string or a function
        neural network to use as a q function approximator. If string, has to be one of the names of registered models in baselines.common.models
        (mlp, cnn, conv_only). If a function, should take an observation tensor and return a latent variable tensor, which
        will be mapped to the Q function heads (see build_q_func in baselines.deepq.models for details on that)
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate for adam optimizer
    total_timesteps: int
        number of env steps to optimizer for
    buffer_size: int
        size of the replay buffer
    exploration_fraction: float
        fraction of entire training period over which the exploration rate is annealed
    exploration_final_eps: float
        final value of random action probability
    train_freq: int
        update the model every `train_freq` steps.
        set to None to disable printing
    batch_size: int
        size of a batched sampled from replay buffer for training
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    checkpoint_freq: int
        how often to save the model. This is so that the best version is restored
        at the end of the training. If you do not wish to restore the best version at
        the end of the training set this variable to None.
    learning_starts: int
        how many steps of the model to collect transitions for before learning starts
    gamma: float
        discount factor
    target_network_update_freq: int
        update the target network every `target_network_update_freq` steps.
    prioritized_replay: True
        if True prioritized replay buffer will be used.
    prioritized_replay_alpha: float
        alpha parameter for prioritized replay buffer
    prioritized_replay_beta0: float
        initial value of beta for prioritized replay buffer
    prioritized_replay_beta_iters: int
        number of iterations over which beta will be annealed from initial value
        to 1.0. If set to None equals to total_timesteps.
    prioritized_replay_eps: float
        epsilon to add to the TD errors when updating priorities.
    callback: (locals, globals) -> None
        function called at every steps with state of the algorithm.
        If callback returns true training stops.
    load_path: str
        path to load the model from. (default: None)
    **network_kwargs
        additional keyword arguments to pass to the network builder.

    Returns
    -------
    act: ActWrapper
        Wrapper over act function. Adds ability to save it and load it.
        See header of baselines/deepq/categorical.py for details on the act function.
    """
    # fix the seed
    tf.random.set_seed(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)

    # define the agents
    agent = [None]*env.number_of_nodes
    for node in range(env.number_of_nodes):
        agent[node] = DQN_AGENT(
            q_func=DQN_AGENT,
            observation_shape=env.observation_space[node].shape,
            num_actions=env.action_space[node].n,
            num_nodes=env.number_of_nodes,
            input_size_splits = [1, env.obs_action_history_length,
                                 env.obs_number_of_future_destinations,
                                 env.obs_num_max_queue_node,
                                 env.obs_use_buffer_pkts_num * len(list(env.G.neighbors(node))),
                                 env.obs_use_buffer_lengths * len(list(env.G.neighbors(node)))
                                 ],
            lr=lr,
            gamma=gamma
        )

    # load the models
    if load_path is not None:
        loaded_models = load_model(load_path)
        if loaded_models is not None:
            for i in range(len(agent)):
                agent[i].q_network.set_weights(loaded_models[i].get_weights())
                # agent[i].update_target()
            print("Restoring from {}".format(load_path))

    # Create the schedule for exploration.
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=exploration_initial_eps,
                                 final_p=exploration_final_eps)

    # for node in range(env.number_of_nodes):
    #     agent[node].update_target()

    episode_rewards = [0.0]
    env.reset()
    episode_info_df = pd.DataFrame() 
    episode_mean_td_error = [0.0]
    # Create the replay buffer
    replay_buffer = [ReplayBuffer(buffer_size) for n in range(env.number_of_nodes)]
    beta_schedule = None
     
    # start episode   
    start_time = time.time()
    episode_time = 0
    print("begin training: ", csv_results_file)
    for t in range(total_timesteps):
        update_eps = tf.constant(exploration.value(t))

        # to change
        obses, obses_t, rewards, actions, flags, done, target_update_instructions, _ = env.step(dqn_step_decorator, agents=agent, stochastic=True, update_eps=update_eps)

        ### sync neighbors target q netwroks
        for element in target_update_instructions:
            neighbor_idx = list(env.G.neighbors(element[0])).index(element[1])
            agent[element[0]].sync_neighbor_target_q_network(agent[element[1]].q_network, neighbor_idx)
            
            
        ### Storing the states in the replay buffer
        for n in range(env.number_of_nodes):
            if len(obses[n]) == 0:
                continue
            for obs_idx in range(len(obses[n])):

                replay_buffer[n].add(np.array(obses[n][obs_idx], dtype=np.float).squeeze(),
                                    actions[n][obs_idx], 
                                    rewards[n][obs_idx],
                                    np.array(obses_t[n][obs_idx], dtype=np.float).squeeze(), 
                                    flags[n][obs_idx])
                

        episode_rewards[-1] += np.sum(np.array(rewards).sum())
        if done:
            delay_per_packet = env.get_delay_per_packet()
            packet_loss = env.get_loss_ratio()
            buffered_packets = sum(np.sum(env._get_all_out_buffer_lengths()))
            arrived_packets = env.num_rvd_packets
            obses = env.reset()
            episode_rewards.append(0.0)
            episode_mean_td_error.append(0.0)


        if t > learning_starts and t % train_freq == 0:
            for n in range(env.number_of_nodes):
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if len(replay_buffer[n])==0:
                    continue
                obses_t, actions_t, rewards, next_obses, dones = replay_buffer[n].sample(batch_size)
                weights, _ = np.ones(batch_size, dtype=np.float32), None
                neighbors = list(env.G.neighbors(n))
                targets_t = []
                action_indices_all = []
                for indx in range(len(neighbors)):
                    next_hop_ = neighbors[indx]
                    filtered_indices = np.where(np.array(list(env.G.neighbors(next_hop_)))!=n)[0]
                    action_indices = np.where(actions_t == indx)[0]
                    action_indices_all.append(action_indices)
                    if len(action_indices):
                        targets_t.append(agent[n].get_neighbor_target_value(indx, rewards[action_indices], tf.constant(
                            np.array(np.vstack(next_obses[action_indices]), dtype=np.float)), dones[action_indices], filtered_indices))
                # for indx in range(batch_size):
                #     next_hop_ = neighbors[actions_t[indx]]
                #     filtered_indices = np.where(np.array(list(env.G.neighbors(next_hop_)))!=n)[0]
                #     targets_t.append(agent[next_hop_].get_target_value(rewards[indx], tf.constant(next_obses[indx]), dones[indx]))
                action_indices_all = np.concatenate(action_indices_all)
                obses_t = tf.constant(obses_t[action_indices_all,])
                actions_t = tf.constant(actions_t[action_indices_all], shape=(batch_size))
                targets_t = tf.constant(tf.concat(targets_t, axis=0), shape=(batch_size))
                weights = tf.constant(weights)
                td_errors = agent[n].train(obses_t, actions_t, targets_t, weights)

                # obses_t = tf.constant(obses_t, dtype=tf.float32)
                # actions = tf.constant(actions, shape=(batch_size))
                # targets_t = tf.constant(targets, shape=(batch_size), dtype=tf.float32)

                # weights = tf.constant(weights)
                # td_errors = agent[n].train(obses_t, actions, targets, weights)
                # print(env.num_rvd_packets, np.mean(td_errors))       
                episode_mean_td_error[-1] += np.mean(td_errors)


        ### Update target network periodically.
        # if t > learning_starts and t % target_network_update_freq == 0:
        #     for n in range(env.number_of_nodes):
        #         agent[n].update_target()

        mean_100ep_reward = round(np.mean(episode_rewards[-101:-2]), 1)
        num_episodes = len(episode_rewards)-1
        if done and print_freq is not None and len(episode_rewards) % print_freq == 0:
            validation_env.reset()
            delay_val = []
            lost_val = []
            while (validation_env.current_episode+1) % 2 != 0:
                    _, _, _, _, _, done_val, _, _ = validation_env.step(dqn_step_decorator, agents=agent, stochastic=False, update_eps=-1)
                    if done_val:
                        delay_val.append(validation_env.get_delay_per_packet())
                        lost_val.append(validation_env.get_loss_ratio())
                        arrived_packets_val = validation_env.num_rvd_packets
                        buffered_packets_val = sum(np.sum(validation_env._get_all_out_buffer_lengths()))
                        validation_env.reset()
                        
            episode_time = str(datetime.timedelta(seconds= time.time() - start_time))
            start_time = time.time()
            print("steps", t)
            print("episodes", num_episodes)
            print("mean 100 episode reward", mean_100ep_reward)
            print("episode_rewards", episode_rewards[-2])
            print("average train delay", delay_per_packet)
            print("nb arrived packets train", arrived_packets)
            print("nb buffered packets train", buffered_packets)
            print("average train loss", packet_loss)
            print("% epsilon", int(100 * update_eps))
            print("% time spent exploring", int(100 * exploration.value(t)))
            print("average validation delay", round(np.mean(delay_val),3))
            print("average validation loss", round(np.mean(lost_val), 5))
            print("nb arrived packets val", arrived_packets_val)
            print("nb buffered packets val", buffered_packets_val)
            print("time spend for the episode", episode_time)
            episode_info_df = save_info_df(csv_results_file, 
                                           episode_info_df, 
                                           t, 
                                           len(episode_rewards),
                                           delay_per_packet, 
                                           packet_loss, 
                                           episode_rewards[-2], 
                                           round(np.mean(delay_val),3),  
                                           round(np.mean(lost_val), 5), 
                                           exploration.value(t), 
                                           episode_mean_td_error[-1],
                                )
            
            save_model(agent, model_path, t, len(episode_rewards))
            
    return agent
