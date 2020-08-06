from functools import reduce

import numpy as np
import torch

from active_learning.observation_utils import augment_obs_posterior, get_posterior
from network.vae_utils import loss_inference_closed_form
from ppo_a2c.algo.ppo import PPO
from ppo_a2c.envs import make_vec_envs_multi_task
from ppo_a2c.model import MLPBase, Policy
from ppo_a2c.storage import RolloutStorage


def _rescale_action(action, max_new, min_new):
    return (max_new - min_new) / (1 - (-1)) * (action - 1) + 100


class PosteriorMTAgent:

    def __init__(self, action_space, device, gamma, num_steps, num_processes,
                 clip_param, ppo_epoch, num_mini_batch, value_loss_coef,
                 entropy_coef, lr, eps, max_grad_norm, use_linear_lr_decay, use_gae, gae_lambda,
                 use_proper_time_limits, obs_shape, latent_dim,
                 recurrent_policy, hidden_size, use_elu,
                 variational_model, vae_optim, rescale_obs, max_old, min_old, vae_min_seq, vae_max_seq,
                 max_action, min_action):
        # General parameters
        self.device = device
        self.gamma = gamma
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape  # env shape + obs shape
        self.latent_dim = latent_dim
        self.action_space = action_space

        # Rescale information
        self.rescale_obs = rescale_obs
        self.max_old = max_old
        self.min_old = min_old

        self.max_action = max_action
        self.min_action = min_action

        # Variational inference parameters
        self.vae_min_seq = vae_min_seq
        self.vae_max_seq = vae_max_seq
        self.vae = variational_model
        self.vae_optim = vae_optim

        # PPO parameters
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

        base = MLPBase
        self.actor_critic = Policy(self.obs_shape,
                                   self.action_space, base=base,
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
              eval_interval, num_task_to_eval, init_vae_steps,
              num_vae_steps, log_dir=".", use_env_obs=False, verbose=True):
        eval_list = []
        vae_list = []

        self.vae_step(init_vae_steps, use_env_obs, task_generator, env_name, seed, log_dir, verbose,
                      init_vae=True)

        # self.vae = torch.load("notebooks/inference_2")

        for k in range(training_iter):
            # Variational training step
            res_vae = self.vae_step(num_vae_steps, use_env_obs, task_generator, env_name, seed, log_dir,
                                    verbose=verbose, init_vae=False)
            vae_list.append(res_vae)

            # Optimal policy training step
            envs_kwargs, prev_task, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            False, envs_kwargs, num_frame_stack=None)
            self.multi_task_policy_step(envs, prev_task, prior, use_env_obs)

            # Evaluation
            if eval_interval is not None and k % eval_interval == 0 and k > 1:
                print("Epoch {} / {}".format(k, training_iter))
                e = self.evaluate(num_task_to_eval, task_generator, log_dir, seed, use_env_obs, env_name)
                eval_list.append(e)

        return eval_list, vae_list

    def vae_step(self, num_vae_steps, use_env_obs, task_generator, env_name, seed, log_dir, verbose, init_vae):
        train_loss = 0
        mse_train_loss = 0
        kdl_train_loss = 0

        for vae_step in range(num_vae_steps):
            envs_kwargs, prev_task, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            False, envs_kwargs, num_frame_stack=None)
            # Data structure for the loss function
            prior = torch.empty(self.num_processes, 4)
            mu_prior = torch.empty(self.num_processes, 2)
            logvar_prior = torch.empty(self.num_processes, 2)

            for t_idx in range(self.num_processes):
                prior[t_idx] = prior_list[t_idx].reshape(1, 4).squeeze(0).clone().detach()
                mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
                logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()

            # Sample data under the current policy
            obs = envs.reset()
            obs = augment_obs_posterior(obs, self.latent_dim, prior, use_env_obs, rescale_obs=self.rescale_obs,
                                        max_old=self.max_old, min_old=self.min_old)
            rollouts_multi_task = RolloutStorage(self.num_steps, self.num_processes,
                                                 self.obs_shape, self.action_space,
                                                 self.actor_critic.recurrent_hidden_state_size)
            rollouts_multi_task.obs[0].copy_(obs)
            rollouts_multi_task.to(self.device)

            num_data_context = torch.randint(low=self.vae_min_seq, high=self.vae_max_seq, size=(1,)).item()
            context = torch.empty(self.num_processes, num_data_context, 2)

            if init_vae:
                for step in range(num_data_context):
                    with torch.no_grad():
                        _, action, _, _ = self.actor_critic.act(
                            rollouts_multi_task.obs[0], rollouts_multi_task.recurrent_hidden_states[0],
                            rollouts_multi_task.masks[0])
                    _, reward, _, _ = envs.step(action)
                    vae_action = _rescale_action(action, max_new=self.max_action, min_new=self.min_action)
                    context[:, step, 0] = vae_action.squeeze(1)
                    context[:, step, 1] = reward.squeeze(1)
            else:
                for step in range(num_data_context):
                    use_prev_state = True if step > 0 else 0

                    # Sample context under
                    with torch.no_grad():
                        value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                            rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                            rollouts_multi_task.masks[step])

                    obs, reward, done, infos = envs.step(action)
                    posterior = get_posterior(self.vae, action, reward, prior, prev_task, max_action=self.max_action,
                                              min_action=self.min_action, use_prev_state=use_prev_state)
                    obs = augment_obs_posterior(obs, self.latent_dim, posterior,
                                                use_env_obs, rescale_obs=self.rescale_obs,
                                                max_old=self.max_old, min_old=self.min_old)
                    vae_action = _rescale_action(action, max_new=self.max_action, min_new=self.min_action)
                    context[:, step, 0] = vae_action.squeeze(1)
                    context[:, step, 1] = reward.squeeze(1)

                    # If done then clean the history of observations.
                    masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                    bad_masks = torch.FloatTensor(
                        [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                    rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                               action_log_prob, value, reward, masks, bad_masks)

            # Now that data have been collected, we train the variational model
            self.vae_optim.zero_grad()
            z_hat, mu_hat, logvar_hat = self.vae(context, torch.tensor(prev_task), prior)

            loss, mse, kdl = loss_inference_closed_form(new_tasks, mu_hat, logvar_hat, mu_prior, logvar_prior, None,
                                                        verbose)
            loss.backward()
            train_loss += loss.item()
            mse_train_loss += mse
            kdl_train_loss += kdl
            self.vae_optim.step()

            if verbose and vae_step % 100 == 0:
                print("Vae step {}/{}, mse {}, kdl {}".format(vae_step, num_vae_steps, mse, kdl))

        return train_loss / num_vae_steps, mse_train_loss / num_vae_steps, kdl_train_loss / num_vae_steps

    def multi_task_policy_step(self, envs, prev_task, prior, use_env_obs):
        # Multi-task learning with posterior mean
        obs = envs.reset()
        obs = augment_obs_posterior(obs, self.latent_dim, prior,
                                    use_env_obs, rescale_obs=self.rescale_obs,
                                    max_old=self.max_old, min_old=self.min_old)

        rollouts_multi_task = RolloutStorage(self.num_steps, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)

        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        # Collect observations and store them into the storage
        for step in range(self.num_steps):
            use_prev_state = True if step > 0 else False

            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                    rollouts_multi_task.masks[step])

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            posterior = get_posterior(self.vae, action, reward, prior, prev_task, max_action=self.max_action,
                                      min_action=self.min_action, use_prev_state=use_prev_state)
            obs = augment_obs_posterior(obs, self.latent_dim, posterior,
                                        use_env_obs, rescale_obs=self.rescale_obs,
                                        max_old=self.max_old, min_old=self.min_old)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                       action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = self.actor_critic.get_value(
                rollouts_multi_task.obs[-1], rollouts_multi_task.recurrent_hidden_states[-1],
                rollouts_multi_task.masks[-1]).detach()

        rollouts_multi_task.compute_returns(next_value, self.use_gae, 1,
                                            self.gae_lambda, self.use_proper_time_limits)

        self.agent.update(rollouts_multi_task)

        rollouts_multi_task.after_update()

    def evaluate(self, num_task_to_evaluate, task_generator, log_dir, seed, use_env_obs, env_name):
        assert num_task_to_evaluate % self.num_processes == 0

        print("Evaluation...")

        n_iter = num_task_to_evaluate // self.num_processes
        r_epi_list = []

        for _ in range(n_iter):
            envs_kwargs, prev_task, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            eval_envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                                 False, envs_kwargs, num_frame_stack=None)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            obs = augment_obs_posterior(obs, self.latent_dim, prior,
                                        use_env_obs, rescale_obs=self.rescale_obs,
                                        max_old=self.max_old, min_old=self.min_old)

            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

            use_prev_state = False
            while len(eval_episode_rewards) < self.num_processes:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)

                # Observe reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                posterior = get_posterior(self.vae, action, reward, prior, prev_task,
                                          min_action=self.min_action, max_action=self.max_action,
                                          use_prev_state=use_prev_state)
                obs = augment_obs_posterior(obs, self.latent_dim, posterior,
                                            use_env_obs, rescale_obs=self.rescale_obs,
                                            max_old=self.max_old, min_old=self.min_old)

                use_prev_state = True
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
        print("Evaluation using {} tasks. Mean reward: {}".format(num_task_to_evaluate, np.mean(r_epi_list)))
        return np.mean(r_epi_list)

    def test_task_sequence(self, env_name, seed, log_dir, envs_kwargs_list):
        num_tasks = len(envs_kwargs_list)

        for kwargs in envs_kwargs_list:
            # Create task
            eval_envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                                 False, kwargs, num_frame_stack=None)

