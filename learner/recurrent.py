import torch
import numpy as np
from functools import reduce

from ppo_a2c.algo.ppo import PPO
from ppo_a2c.envs import get_vec_envs_multi_task
from ppo_a2c.model import MLPBase, Policy
from ppo_a2c.storage import RolloutStorage


class RL2:

    def __init__(self, hidden_size, use_elu, clip_param, ppo_epoch, num_mini_batch, value_loss_coef,
                 entropy_coef, lr, eps, max_grad_norm, action_space, obs_shape, use_obs_env,
                 num_processes, gamma, device, num_steps, action_dim, use_gae, gae_lambda,
                 use_proper_time_limits):
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.use_obs_env = use_obs_env
        self.num_processes = num_processes
        self.gamma = gamma
        self.device = device
        self.num_steps = num_steps
        self.action_dim = action_dim

        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

        base = MLPBase
        self.actor_critic = Policy(self.obs_shape,
                                   self.action_space, base=base,
                                   base_kwargs={'recurrent': True,
                                                'hidden_size': hidden_size,
                                                'use_elu': use_elu})

        self.agent = PPO(self.actor_critic,
                         clip_param,
                         ppo_epoch,
                         num_mini_batch,
                         value_loss_coef,
                         entropy_coef,
                         lr=lr,
                         eps=eps,
                         max_grad_norm=max_grad_norm,
                         use_clipped_value_loss=True)
        self.envs = None
        self.eval_envs = None

    def train(self, n_iter, env_name, seed, task_generator,
              eval_interval, num_test_processes, num_random_task_to_eval,
              prior_task_sequences=None, log_dir=".", verbose=True):

        eval_list = []
        test_list = []

        for i in range(n_iter):
            self.train_iter(env_name=env_name, seed=seed, task_generator=task_generator, log_dir=log_dir)
            if i % 10 == 0:
                print("Iteration {} / {}".format(i, n_iter))

            if i % eval_interval == 0:
                # if verbose:
                #    print("Iteration {} / {}".format(i, n_iter))
                e = self.evaluate(num_task_to_evaluate=num_random_task_to_eval, task_generator=task_generator,
                                  log_dir=log_dir, seed=seed, env_name=env_name)
                eval_list.append(e)

                e = self.meta_test(prior_task_sequences, task_generator, num_test_processes, env_name, seed, log_dir)
                test_list.append(e)

        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()

        return eval_list, test_list

    def build_obs(self, obs, reward, action, is_init, use_obs_env, num_processes):
        if is_init:
            reward = torch.zeros(num_processes, 1)
            action = torch.zeros(num_processes, self.action_dim)

        if use_obs_env:
            new_obs = torch.cat([action, reward, obs], 1)
        else:
            new_obs = torch.cat([action, reward], 1)

        return new_obs

    def train_iter(self, env_name, seed, task_generator, log_dir):
        envs_kwargs, prev_task, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            True, envs_kwargs, self.envs, num_frame_stack=None)

        obs = self.envs.reset()
        obs = self.build_obs(obs=obs, reward=None, action=None, is_init=True, use_obs_env=self.use_obs_env,
                             num_processes=self.num_processes)

        rollouts_multi_task = RolloutStorage(self.num_steps, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)

        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        for step in range(self.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                    rollouts_multi_task.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = self.envs.step(action)
            obs = self.build_obs(obs=obs, reward=reward, action=action, is_init=False, use_obs_env=self.use_obs_env,
                                 num_processes=self.num_processes)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                       action_log_prob, value, reward, masks, bad_masks)

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
            self.envs = get_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                                True, envs_kwargs, self.envs, num_frame_stack=None)

            eval_episode_rewards = []

            obs = self.envs.reset()
            obs = self.build_obs(obs=obs, reward=None, action=None, is_init=True, use_obs_env=self.use_obs_env,
                                 num_processes=self.num_processes)

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
                obs, reward, done, infos = self.envs.step(action)
                obs = self.build_obs(obs=obs, reward=reward, action=action, is_init=False, use_obs_env=self.use_obs_env,
                                     num_processes=self.num_processes)

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

    def meta_test(self, prior_task_sequences, task_generator, num_test_processes, env_name, seed, log_dir):

        result_all = []

        for sequence in prior_task_sequences:
            sequence_rewards = []
            for prior in sequence:
                kwargs = task_generator.sample_task_from_prior(prior)
                temp = [kwargs for _ in range(num_test_processes)]
                self.eval_envs = get_vec_envs_multi_task(env_name, seed, num_test_processes, self.gamma, log_dir,
                                                         self.device, True, temp, self.eval_envs, num_frame_stack=None)

                eval_episode_rewards = []

                obs = self.eval_envs.reset()
                obs = self.build_obs(obs=obs, reward=None, action=None, is_init=True, use_obs_env=self.use_obs_env,
                                     num_processes=num_test_processes)

                eval_recurrent_hidden_states = torch.zeros(
                    num_test_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
                eval_masks = torch.zeros(num_test_processes, 1, device=self.device)
                already_ended = torch.zeros(num_test_processes, dtype=torch.bool)
                while len(eval_episode_rewards) < num_test_processes:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks,
                            deterministic=False)

                    # Observe reward and next obs
                    obs, reward, done, infos = self.eval_envs.step(action)
                    obs = self.build_obs(obs=obs, reward=reward, action=action, is_init=False,
                                         use_obs_env=self.use_obs_env, num_processes=num_test_processes)

                    eval_masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done],
                        dtype=torch.float32,
                        device=self.device)

                    for i, info in enumerate(infos):
                        if 'episode' in info.keys() and not already_ended[i]:
                            total_epi_reward = info['episode']['r']
                            eval_episode_rewards.append(total_epi_reward)
                    already_ended = already_ended | done

                sequence_rewards.append(np.mean(eval_episode_rewards))

            result_all.append(sequence_rewards)

        return result_all
