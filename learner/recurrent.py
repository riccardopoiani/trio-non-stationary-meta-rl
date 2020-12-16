from functools import reduce

import numpy as np
import torch

from ppo.algo.ppo import PPO
from ppo.envs import get_vec_envs_multi_task
from ppo.model import MLPBase, Policy, MLPRL2FeatureExtractor, RL2Base
from ppo.storage import RolloutStorage


class RL2:

    def __init__(self,
                 hidden_size,
                 use_elu,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr,
                 eps,
                 max_grad_norm,
                 action_space,
                 obs_shape,
                 use_obs_env,
                 num_processes,
                 gamma,
                 device,
                 num_steps,
                 action_dim,
                 state_dim,
                 use_gae,
                 gae_lambda,
                 use_proper_time_limits,
                 use_xavier,
                 use_huber_loss,
                 use_rms_rew,
                 use_rms_rew_in_policy,
                 use_rms_act,
                 use_rms_state,
                 use_extractor,
                 state_extractor_dim,
                 action_extractor_dim,
                 reward_extractor_dim,
                 done_extractor_dim,
                 use_done,
                 latent_dim):
        """
        :param hidden_size: hidden sizes of the layer of the policy network
        :param use_elu: if True ELU activation will be used in the policy network, else Tanh will be used
        :param clip_param: clip parameter to be used in PPO
        :param ppo_epoch: number of PPO epochs per training iteration
        :param num_mini_batch: number of mini batches that will be used in PPO
        :param value_loss_coef: value loss coefficient used in PPO updates
        :param entropy_coef: entropy coefficient used in PPO
        :param lr: learning rate of Adam optimizer used for policy training
        :param eps: epsilon parameter of Adam optimizer used for policy training
        :param max_grad_norm: maximum gradient norm used in PPO
        :param action_space: action space of the environment that the agent will be trained on
        :param obs_shape: shape of the observation
        :param use_obs_env: whether to use or not environment observation. This is False in MAB environments
        :param num_processes: number of parallel processes that will collect data to update the policy
        :param gamma: discount factor of RL problem
        :param device: device on which the code will run
        :param num_steps: number of steps that will be taken before updating the policy in each process
        :param action_dim: action dimension
        :param state_dim: state dimension
        :param use_gae: True if Generalized Advantage Estimation should be used
        :param gae_lambda: lambda parameter of Generalized Advantage Estimatino
        :param use_proper_time_limits: whether to use proper time limits or not. If False, time limits will be
        considered at the same way of terminal states
        :param use_xavier:if True xavier init will be used for the Policy network; if False orthogonal init will
        be used
        :param use_huber_loss: whether to use a Huber loss in RL training or not
        :param use_extractor: True if a more complex network that uses feature extraction layers and smoothers
        should be used
        :param use_rms_rew: True if reward should be smoothed in RL training
        :param use_rms_rew_in_policy: True if reward should be smoothed when they are fed to the policy
        :param use_rms_act: True if actions should be smoothed when they are fed to the policy
        :param use_rms_state: True if state should be smoothed when it is fed to the policy
        :param state_extractor_dim: dimension of the feature extraction layer that concerns the state
        :param action_extractor_dim: dimension of the feature extraction layer that concerns the action
        :param reward_extractor_dim: dimension of the feature extraction layer that concerns the reward
        :param done_extractor_dim: dimension of the feature extraction layer that concerns the done signal
        :param use_done: True if Policy should be conditioned on done signals
        """
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.use_obs_env = use_obs_env
        self.num_processes = num_processes
        self.gamma = gamma
        self.device = device
        self.num_steps = num_steps
        self.action_dim = action_dim

        self.use_done = use_done
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

        self.use_rms_rew = use_rms_rew

        if use_extractor:
            base = MLPRL2FeatureExtractor
            self.actor_critic = Policy(self.obs_shape,
                                       self.action_space,
                                       base=base,
                                       base_kwargs={
                                           'recurrent': True,
                                           'hidden_size': hidden_size,
                                           'use_elu': use_elu,
                                           'use_xavier': use_xavier,
                                           'state_dim': state_dim,
                                           'state_extractor_dim': state_extractor_dim,
                                           'action_dim': action_dim,
                                           'action_extractor_dim': action_extractor_dim,
                                           'reward_extractor_dim': reward_extractor_dim,
                                           'done_extractor_dim': done_extractor_dim,
                                           'has_done': use_done,
                                           'norm_state': use_rms_state,
                                           'norm_action': use_rms_act,
                                           'norm_reward': use_rms_rew_in_policy
                                       })
        else:
            """
            base = MLPBase
            self.actor_critic = Policy(self.obs_shape,
                                       self.action_space, base=base,
                                       base_kwargs={'recurrent': True,
                                                    'hidden_size': hidden_size,
                                                    'use_elu': use_elu,
                                                    'use_xavier': use_xavier})
            """
            base = RL2Base
            self.actor_critic = Policy(self.obs_shape,
                                       self.action_space, base=base,
                                       base_kwargs={
                                           'latent_dim': latent_dim,
                                           'hidden_size': hidden_size,
                                           'use_elu': use_elu,
                                           'ext_hidden_sizes': None,
                                           'use_xavier': use_xavier,
                                           'use_env_obs': use_obs_env,
                                           'state_dim': state_dim
                                       })

        self.agent = PPO(self.actor_critic,
                         clip_param,
                         ppo_epoch,
                         num_mini_batch,
                         value_loss_coef,
                         entropy_coef,
                         lr=lr,
                         eps=eps,
                         max_grad_norm=max_grad_norm,
                         use_clipped_value_loss=True,
                         use_huber_loss=use_huber_loss)
        self.envs = None
        self.eval_envs = None

    def train(self, n_iter, env_name, seed, task_generator,
              eval_interval, num_test_processes, num_random_task_to_eval,
              task_len, prior_task_sequences=None, log_dir=".", verbose=True):

        eval_list = []
        test_list = []

        for i in range(n_iter):
            self.train_iter(env_name=env_name, seed=seed, task_generator=task_generator, log_dir=log_dir,
                            task_len=task_len)
            if i % 10 == 0:
                print("Iteration {} / {}".format(i, n_iter))

            if i % eval_interval == 0:
                e = self.evaluate(num_task_to_evaluate=num_random_task_to_eval, task_generator=task_generator,
                                  log_dir=log_dir, seed=seed, env_name=env_name)
                eval_list.append(e)

                e = self.meta_test(prior_task_sequences, task_generator, num_test_processes, env_name, seed, log_dir,
                                   task_len=task_len)
                test_list.append(e)

        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()

        return eval_list, test_list

    def build_obs(self, obs, reward, action, is_init, use_obs_env, done, num_processes):
        if is_init:
            done = torch.zeros(num_processes, 1)
            reward = torch.zeros(num_processes, 1)
            action = torch.zeros(num_processes, self.action_dim)
        else:
            done = torch.FloatTensor([[1.0] if _done else [0.0] for _done in done])
            action = action.float()

        if use_obs_env:
            if self.use_done:
                new_obs = torch.cat([done, action, reward, obs], 1)
            else:
                new_obs = torch.cat([action, reward, obs], 1)
        else:
            if self.use_done:
                new_obs = torch.cat([done, action, reward], 1)
            else:
                new_obs = torch.cat([action, reward], 1)

        return new_obs

    def train_iter(self, env_name, seed, task_generator, log_dir, task_len):
        envs_kwargs, prev_task, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name=env_name,
                                            seed=seed,
                                            num_processes=self.num_processes,
                                            gamma=self.gamma,
                                            log_dir=log_dir,
                                            device=self.device,
                                            allow_early_resets=True,
                                            env_kwargs_list=envs_kwargs,
                                            normalize_rew=self.use_rms_rew,
                                            envs=self.envs,
                                            num_frame_stack=None)

        obs = self.envs.reset()
        obs = self.build_obs(obs=obs, reward=None, action=None, is_init=True, use_obs_env=self.use_obs_env,
                             num_processes=self.num_processes, done=None)

        rollouts_multi_task = RolloutStorage(self.num_steps, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)

        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        keep_init_if_done_masks = torch.FloatTensor([[1.0] for _ in range(self.num_processes)])
        bad_masks = torch.FloatTensor([[1.0] or _ in range(self.num_processes)])
        for step in range(self.num_steps):
            # Sample actions
            with torch.no_grad():
                if task_len == 1 or step == 0:
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                        rollouts_multi_task.masks[step]
                    )
                else:
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                        rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                        keep_init_if_done_masks)

            # Observe reward and next obs
            obs, (reward, reward_norm), done, infos = self.envs.step(action)
            obs = self.build_obs(obs=obs, reward=reward, action=action, is_init=False, use_obs_env=self.use_obs_env,
                                 num_processes=self.num_processes, done=done)

            # If done then clean the history of observations.
            if self.use_done:
                rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                           action_log_prob, value, reward_norm, keep_init_if_done_masks, bad_masks)
            else:
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                           action_log_prob, value, reward_norm, masks, bad_masks)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                rollouts_multi_task.obs[-1], rollouts_multi_task.recurrent_hidden_states[-1],
                rollouts_multi_task.masks[-1]).detach()

        rollouts_multi_task.compute_returns(next_value, self.use_gae, self.gamma,
                                            self.gae_lambda, self.use_proper_time_limits)

        self.agent.update(rollouts_multi_task)

        rollouts_multi_task.after_update()

    def evaluate(self, num_task_to_evaluate, task_generator, log_dir, seed, env_name):
        assert num_task_to_evaluate % self.num_processes == 0

        n_iter = num_task_to_evaluate // self.num_processes
        r_epi_list = []

        for _ in range(n_iter):
            envs_kwargs, prev_task, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            self.eval_envs = get_vec_envs_multi_task(env_name=env_name,
                                                     seed=seed,
                                                     num_processes=self.num_processes,
                                                     gamma=self.gamma,
                                                     log_dir=log_dir,
                                                     device=self.device,
                                                     allow_early_resets=True,
                                                     env_kwargs_list=envs_kwargs,
                                                     envs=None,
                                                     normalize_rew=self.use_rms_rew,
                                                     num_frame_stack=None)

            eval_episode_rewards = []

            obs = self.eval_envs.reset()
            obs = self.build_obs(obs=obs, reward=None, action=None, is_init=True, use_obs_env=self.use_obs_env,
                                 num_processes=self.num_processes, done=None)

            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

            already_ended = torch.zeros(self.num_processes, dtype=torch.bool)
            while len(eval_episode_rewards) < self.num_processes:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)

                # Observe reward and next obs
                obs, (reward, reward_norm), done, infos = self.eval_envs.step(action)
                obs = self.build_obs(obs=obs, reward=reward, action=action, is_init=False, use_obs_env=self.use_obs_env,
                                     num_processes=self.num_processes, done=done)

                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device)

                for i, info in enumerate(infos):
                    if 'episode' in info.keys() and not already_ended[i]:
                        total_epi_reward = info['episode']['r']
                        eval_episode_rewards.append(total_epi_reward)
                already_ended = already_ended | done

            r_epi_list.append(eval_episode_rewards)

        r_epi_list = reduce(list.__add__, r_epi_list)
        print("Evaluation using {} tasks. Mean reward: {}".format(num_task_to_evaluate, np.mean(r_epi_list)))
        return np.mean(r_epi_list)

    def meta_test(self, prior_task_sequences, task_generator, num_test_processes, env_name, seed, log_dir,
                  task_len):

        result_all = []

        for sequence in prior_task_sequences:
            sequence_rewards = []
            for prior in sequence:
                start_task = True
                prev_episodes_hidden_states = torch.zeros(
                    num_test_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
                task_r = []
                for _ in range(task_len):
                    kwargs = task_generator.sample_task_from_prior(prior)
                    self.eval_envs = get_vec_envs_multi_task(env_name=env_name,
                                                             seed=seed,
                                                             num_processes=num_test_processes,
                                                             gamma=self.gamma,
                                                             log_dir=log_dir,
                                                             device=self.device,
                                                             allow_early_resets=True,
                                                             env_kwargs_list=[kwargs for _ in
                                                                              range(num_test_processes)],
                                                             normalize_rew=self.use_rms_rew,
                                                             envs=None,
                                                             num_frame_stack=None)
                    eval_episode_rewards = []

                    obs = self.eval_envs.reset()

                    obs = self.build_obs(obs=obs, reward=None, action=None, is_init=True, use_obs_env=self.use_obs_env,
                                         num_processes=num_test_processes, done=None)

                    if start_task:
                        eval_recurrent_hidden_states = torch.zeros(
                            num_test_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
                        eval_masks = torch.zeros(num_test_processes, 1, device=self.device)
                    else:
                        eval_recurrent_hidden_states = prev_episodes_hidden_states
                        eval_masks = torch.ones(num_test_processes, 1, device=self.device)
                    start_task = False

                    already_ended = torch.zeros(num_test_processes, dtype=torch.bool)

                    while len(eval_episode_rewards) < num_test_processes:
                        with torch.no_grad():
                            _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                                obs,
                                eval_recurrent_hidden_states,
                                eval_masks,
                                deterministic=False)

                        # Observe reward and next obs
                        obs, (reward, reward_norm), done, infos = self.eval_envs.step(action)
                        obs = self.build_obs(obs=obs, reward=reward, action=action, is_init=False,
                                             use_obs_env=self.use_obs_env, num_processes=num_test_processes,
                                             done=done)

                        eval_masks = torch.tensor(
                            [[0.0] if done_ else [1.0] for done_ in done],
                            dtype=torch.float32,
                            device=self.device)

                        for i, info in enumerate(infos):
                            if 'episode' in info.keys() and not already_ended[i]:
                                total_epi_reward = info['episode']['r']
                                eval_episode_rewards.append(total_epi_reward)
                                prev_episodes_hidden_states[i] = eval_recurrent_hidden_states[i].clone().detach()
                                task_r.append(total_epi_reward)
                        already_ended = already_ended | done
                    # print(np.mean(eval_episode_rewards))
                    # sequence_rewards.append(np.mean(eval_episode_rewards))
                sequence_rewards.append(np.mean(task_r))
            print("Sequence results {}".format(sequence_rewards))
            result_all.append(sequence_rewards)

        return result_all
