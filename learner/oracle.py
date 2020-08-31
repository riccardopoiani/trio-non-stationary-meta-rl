import numpy as np
import torch

from functools import reduce

from utilities.observation_utils import oracle_augment_obs
from ppo_a2c.algo.ppo import PPO
from ppo_a2c.envs import get_vec_envs_multi_task
from ppo_a2c.model import MLPBase, Policy
from ppo_a2c.storage import RolloutStorage


class OracleAgent:
    """
    Oracle multi-task agent that knows the latent parametrization of the task
    """

    def __init__(self, action_space, device, gamma, num_steps, num_processes,
                 clip_param, ppo_epoch, num_mini_batch, value_loss_coef,
                 entropy_coef, lr, eps, max_grad_norm, use_linear_lr_decay, use_gae, gae_lambda,
                 use_proper_time_limits, obs_shape, latent_dim,
                 recurrent_policy, hidden_size, use_elu):
        self.device = device
        self.gamma = gamma
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape  # env shape + obs shape
        self.latent_dim = latent_dim

        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

        base = MLPBase
        self.action_space = action_space

        self.actor_critic = Policy(self.obs_shape,
                                   action_space, base=base,
                                   base_kwargs={'recurrent': recurrent_policy,
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

    def train(self, training_iter, env_name, seed, task_generator,
              num_update_per_meta_training_iter, eval_interval, num_task_to_eval, log_dir=".",
              use_env_obs=False):
        eval_list = []

        for j in range(training_iter):
            envs_kwargs, curr_latent = task_generator.sample_task(self.num_processes)

            envs = get_vec_envs_multi_task(env_name,
                                           seed,
                                           self.num_processes,
                                           self.gamma,
                                           log_dir,
                                           self.device,
                                           False,
                                           envs_kwargs,
                                           num_frame_stack=None)
            obs = envs.reset()
            obs = oracle_augment_obs(obs=obs, latent=curr_latent,
                                     latent_dim=self.latent_dim, use_env_obs=use_env_obs)

            rollouts_multi_task = RolloutStorage(self.num_steps, self.num_processes,
                                                 self.obs_shape, self.action_space,
                                                 self.actor_critic.recurrent_hidden_state_size)

            rollouts_multi_task.obs[0].copy_(obs)
            rollouts_multi_task.to(self.device)

            for _ in range(num_update_per_meta_training_iter):
                # Collect observations and store them into the storage
                for step in range(self.num_steps):
                    # Sample actions
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                            rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                            rollouts_multi_task.masks[step])

                    # Obser reward and next obs
                    obs, reward, done, infos = envs.step(action)
                    obs = oracle_augment_obs(obs=obs, latent=curr_latent,
                                             latent_dim=self.latent_dim, use_env_obs=use_env_obs)

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

            if eval_interval is not None and j % eval_interval == 0 and j > 1:
                e = self.evaluate(num_task_to_eval, task_generator, env_name, seed, log_dir,
                                  use_env_obs)
                eval_list.append(e)

        return eval_list

    def evaluate(self, num_task_to_evaluate, task_generator, env_name, seed, log_dir,
                 use_env_obs):
        assert num_task_to_evaluate % self.num_processes == 0

        print("Evaluation...")

        n_iter = num_task_to_evaluate // self.num_processes
        r_epi_list = []

        for _ in range(n_iter):
            envs_kwargs, curr_latent = task_generator.sample_task(self.num_processes)

            eval_envs = get_vec_envs_multi_task(env_name,
                                                seed + self.num_processes,
                                                self.num_processes,
                                                None,
                                                log_dir,
                                                self.device,
                                                False,
                                                envs_kwargs,
                                                num_frame_stack=None)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            obs = oracle_augment_obs(obs=obs, latent=curr_latent,
                                     latent_dim=self.latent_dim, use_env_obs=use_env_obs)
            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

            while len(eval_episode_rewards) < 10:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=True)

                # Observe reward and next obs
                obs, _, done, infos = eval_envs.step(action)
                obs = oracle_augment_obs(obs=obs, latent=curr_latent,
                                         latent_dim=self.latent_dim, use_env_obs=use_env_obs)

                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device)

                for info in infos:
                    if 'episode' in info.keys():
                        total_epi_reward = info['episode']['r']
                        eval_episode_rewards.append(total_epi_reward)

            r_epi_list.append(eval_episode_rewards)
            eval_envs.close()

        r_epi_list = reduce(list.__add__, r_epi_list)
        print("Evaluation using {} tasks. Mean reward: {}".format(n_iter, np.mean(r_epi_list)))
        return np.mean(r_epi_list)
