import torch
import numpy as np
from functools import reduce

from active_learning.observation_utils import oracle_augment_obs, al_augment_obs, get_posterior_no_prev, \
    rescale_posterior, rescale_latent
from active_learning.training_utils import get_reward, get_mse
from network.vae_utils import loss_inference_closed_form
from ppo_a2c.algo.ppo import PPO
from ppo_a2c.envs import make_vec_envs_multi_task
from ppo_a2c.model import MLPBase, Policy
from ppo_a2c.storage import RolloutStorage


class FixedIDAgent:

    def __init__(self, action_space, device, gamma, num_steps, num_processes,
                 clip_param, ppo_epoch, num_mini_batch, value_loss_coef,
                 entropy_coef, lr, eps, max_grad_norm, use_linear_lr_decay, use_gae, gae_lambda,
                 use_proper_time_limits, obs_shape_opt, latent_dim,
                 recurrent_policy, hidden_size, use_elu,
                 variational_model, vae_optim, rescale_obs, max_old, min_old, vae_min_seq, vae_max_seq,
                 max_action, min_action,
                 recurrent_policy_id, hidden_size_id, use_elu_id, clip_param_id, ppo_epoch_id, value_loss_coef_id,
                 lr_id, eps_id, max_grad_norm_id, num_mini_batch_id, entropy_coef_id, obs_shape_id, num_steps_id,
                 gamma_identification):
        # General parameters
        self.device = device
        self.gamma = gamma
        self.gamma_identification = gamma_identification
        self.num_steps = num_steps
        self.num_steps_id = num_steps_id
        self.num_processes = num_processes
        self.obs_shape_opt = obs_shape_opt  # env shape + obs shape
        self.obs_shape_id = obs_shape_id
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

        # Optimal policy
        base_optimal = MLPBase
        self.actor_critic_optimal = Policy(self.obs_shape_opt,
                                           self.action_space, base=base_optimal,
                                           base_kwargs={'recurrent': recurrent_policy,
                                                        'hidden_size': hidden_size,
                                                        'use_elu': use_elu})

        self.agent_optimal = PPO(self.actor_critic_optimal,
                                 clip_param,
                                 ppo_epoch,
                                 num_mini_batch,
                                 value_loss_coef,
                                 entropy_coef,
                                 lr=lr,
                                 eps=eps,
                                 max_grad_norm=max_grad_norm,
                                 use_clipped_value_loss=True)

        # Identification policy
        base_identification = MLPBase
        self.actor_critic_identification = Policy(self.obs_shape_id, self.action_space, base=base_identification,
                                                  base_kwargs={'recurrent': recurrent_policy_id,
                                                               'hidden_size': hidden_size_id,
                                                               'use_elu': use_elu_id})
        self.agent_identification = PPO(self.actor_critic_identification,
                                        clip_param_id,
                                        ppo_epoch_id,
                                        num_mini_batch_id,
                                        value_loss_coef_id,
                                        entropy_coef_id,
                                        lr=lr_id,
                                        eps=eps_id,
                                        max_grad_norm=max_grad_norm_id,
                                        use_clipped_value_loss=True)

    def train(self, training_iter_id, training_iter_optimal, env_name, seed, task_generator,
              eval_interval, num_random_task_to_eval,
              num_vae_steps, num_test_processes, max_id_iteration, gp_list=None, sw_size=None,
              test_kwargs=None, init_prior_test=None,
              log_dir=".", use_env_obs=False, verbose=True):
        if verbose:
            print("Training VI...")
        # train variational network with data-loader
        self.train_vae(num_vae_steps=num_vae_steps, task_generator=task_generator,
                       verbose=verbose)

        if verbose:
            print("Training optimal policy")
        # train optimal policy
        self.train_optimal(n_iter=training_iter_optimal, task_generator=task_generator,
                           env_name=env_name, seed=seed, log_dir=log_dir, use_env_obs=use_env_obs,
                           eval_interval=eval_interval, num_task_to_eval=num_random_task_to_eval)

        if verbose:
            print("Training identification policy")
        # train identification policy
        e, t = self.train_identification(n_iter=training_iter_id, task_generator=task_generator,
                                         eval_interval=eval_interval, seed=seed, log_dir=log_dir,
                                         num_task_to_eval=num_random_task_to_eval,
                                         num_test_processes=num_test_processes,
                                         init_prior_test=init_prior_test, gp_list=gp_list, sw_size=sw_size,
                                         test_kwargs=test_kwargs, use_env_obs=use_env_obs,
                                         max_id_iteration=max_id_iteration, env_name=env_name)
        return e, t

    def train_vae(self, num_vae_steps, task_generator, verbose):
        train_loss = 0
        mse_train_loss = 0
        kdl_train_loss = 0

        for vae_step in range(num_vae_steps):
            data, prev_task, prior_list, new_tasks = task_generator.sample_pair_tasks_data_loader(self.num_processes)

            prior = torch.empty(self.num_processes, self.latent_dim * 2)
            mu_prior = torch.empty(self.num_processes, self.latent_dim)
            logvar_prior = torch.empty(self.num_processes, self.latent_dim)

            for t_idx in range(self.num_processes):
                prior[t_idx] = prior_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()
                mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
                logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()

            num_data_context = torch.randint(low=self.vae_min_seq, high=self.vae_max_seq, size=(1,)).item()
            idx = torch.randperm(self.vae_max_seq)
            ctx_idx = idx[0:num_data_context]

            context = torch.empty(self.num_processes, num_data_context, 2)

            for t_idx in range(self.num_processes):
                # Creating context to be fed to the network
                batch = data[t_idx][0]['train']
                batch = torch.cat([batch[0], batch[1]], dim=1)
                context[t_idx] = batch[ctx_idx]

            self.vae_optim.zero_grad()
            z_hat, mu_hat, logvar_hat = self.vae(context, prior)

            loss, kdl, mse = loss_inference_closed_form(new_tasks, mu_hat, logvar_hat, mu_prior, logvar_prior,
                                                        vae_step, verbose)
            loss.backward()

            train_loss += loss.item()
            mse_train_loss += mse
            kdl_train_loss += kdl
            self.vae_optim.step()
        if num_vae_steps > 0:
            return train_loss / num_vae_steps, mse_train_loss / num_vae_steps, kdl_train_loss / num_vae_steps
        else:
            return 0, 0, 0

    def train_optimal(self, n_iter, task_generator, env_name, seed, log_dir, use_env_obs, eval_interval,
                      num_task_to_eval, verbose=True):
        eval_opt = []
        for curr_iter in range(n_iter):
            # Sample task and create envs
            envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            False, envs_kwargs, num_frame_stack=None)

            # Multi-task learning with posterior mean
            obs = envs.reset()
            obs = oracle_augment_obs(obs=obs, latent=new_tasks, latent_dim=self.latent_dim,
                                     use_env_obs=use_env_obs)
            if self.rescale_obs:
                obs = rescale_latent(self.num_processes, obs, self.latent_dim, self.max_old, self.min_old)

            rollouts_multi_task = RolloutStorage(self.num_steps, self.num_processes,
                                                 self.obs_shape_opt, self.action_space,
                                                 self.actor_critic_optimal.recurrent_hidden_state_size)

            rollouts_multi_task.obs[0].copy_(obs)
            rollouts_multi_task.to(self.device)

            # Collect observations and store them into the storage
            for step in range(self.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic_optimal.act(
                        rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                        rollouts_multi_task.masks[step])

                # Observe reward and next obs
                obs, reward, done, infos = envs.step(action)
                obs = oracle_augment_obs(obs, new_tasks, self.latent_dim, use_env_obs)
                if self.rescale_obs:
                    obs = rescale_latent(self.num_processes, obs, self.latent_dim, self.max_old, self.min_old)

                # If done then clean the history of observations.
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
                rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                           action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic_optimal.get_value(
                    rollouts_multi_task.obs[-1], rollouts_multi_task.recurrent_hidden_states[-1],
                    rollouts_multi_task.masks[-1]).detach()

            rollouts_multi_task.compute_returns(next_value, self.use_gae, self.gamma,
                                                self.gae_lambda, self.use_proper_time_limits)

            self.agent_optimal.update(rollouts_multi_task)

            rollouts_multi_task.after_update()

            if curr_iter % eval_interval == 0:
                if verbose:
                    print("Epoch {} / {}".format(curr_iter, n_iter))
                e = self.evaluate_optimal_policy(task_generator=task_generator, seed=seed, env_name=env_name,
                                                 log_dir=log_dir, use_env_obs=use_env_obs,
                                                 num_task_to_eval=num_task_to_eval)
                eval_opt.append(e)
        return eval_opt

    def evaluate_optimal_policy(self, task_generator, num_task_to_eval,
                                seed=0, env_name='gauss-v0', log_dir=".", use_env_obs=False):
        r_epi_list = []

        n_iter = num_task_to_eval // self.num_processes

        for _ in range(n_iter):
            envs_kwargs, prev_task, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            eval_envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir,
                                                 self.device,
                                                 False, envs_kwargs, num_frame_stack=None)

            eval_episode_rewards = []

            obs = eval_envs.reset()
            obs = oracle_augment_obs(obs, new_tasks, self.latent_dim, use_env_obs)
            if self.rescale_obs:
                obs = rescale_latent(self.num_processes, obs, self.latent_dim, self.max_old, self.min_old)

            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, self.actor_critic_optimal.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

            while len(eval_episode_rewards) < self.num_processes:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = self.actor_critic_optimal.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)

                # Observe reward and next obs
                obs, reward, done, infos = eval_envs.step(action)
                obs = oracle_augment_obs(obs, new_tasks, self.latent_dim, use_env_obs)
                if self.rescale_obs:
                    obs = rescale_latent(self.num_processes, obs, self.latent_dim, self.max_old, self.min_old)

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
        print("Evaluation using {} tasks. Mean reward: {}".format(self.num_processes * n_iter, np.mean(r_epi_list)))
        return np.mean(r_epi_list)

    def train_identification(self, n_iter, task_generator, env_name, seed, log_dir,
                             eval_interval, num_task_to_eval, gp_list, sw_size,
                             test_kwargs, max_id_iteration, num_test_processes, use_env_obs,
                             init_prior_test):
        eval_list = []
        test_list = []

        for k in range(n_iter):
            envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)

            envs = make_vec_envs_multi_task(env_name,
                                            seed,
                                            self.num_processes,
                                            None,
                                            log_dir,
                                            self.device,
                                            False,
                                            envs_kwargs,
                                            num_frame_stack=None)

            obs = envs.reset()
            obs = al_augment_obs(obs=obs, latent_dim=self.latent_dim, posterior=prior_list,
                                 prior=prior_list,
                                 rescale_obs=self.rescale_obs, max_old=self.max_old,
                                 min_old=self.min_old)

            rollouts = RolloutStorage(self.num_steps_id, self.num_processes,
                                      self.obs_shape_id, self.action_space,
                                      self.actor_critic_identification.recurrent_hidden_state_size)

            rollouts.obs[0].copy_(obs)
            rollouts.to(self.device)

            use_prev_state = False

            # Collect observations and store them into the storage
            for step in range(self.num_steps_id):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = self.actor_critic_identification.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                # Observe reward and next obs
                obs, reward, done, infos = envs.step(action)
                posterior = get_posterior_no_prev(self.vae, action, reward, prior_list,
                                                  use_prev_state=use_prev_state, max_action=self.max_action,
                                                  min_action=self.min_action)
                use_prev_state = True

                reward = get_reward(posterior, new_tasks, self.latent_dim, self.num_processes)

                obs = al_augment_obs(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                     prior=prior_list,
                                     rescale_obs=self.rescale_obs, max_old=self.max_old,
                                     min_old=self.min_old)

                # If done then clean the history of observations.
                if done.any():
                    use_prev_state = False
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])

                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = self.actor_critic_identification.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            rollouts.compute_returns(next_value, self.use_gae, self.gamma_identification,
                                     self.gae_lambda, self.use_proper_time_limits)

            _, _, _ = self.agent_identification.update(rollouts)

            rollouts.after_update()

            if eval_interval is not None and k % eval_interval == 0 and k > 1:
                e = self.evaluate_identification(num_task_to_eval=num_task_to_eval, task_generator=task_generator
                                                 , seed=seed, env_name=env_name, log_dir=log_dir)
                eval_list.append(e)

                e = self.meta_test(gp_list=gp_list, sw_size=sw_size, env_name=env_name, seed=seed,
                                   log_dir=log_dir, envs_kwargs_list=test_kwargs,
                                   init_prior=init_prior_test, use_env_obs=use_env_obs,
                                   num_eval_processes=num_test_processes, max_id_iteration=max_id_iteration)
                test_list.append(e)

        return eval_list, test_list

    def evaluate_identification(self, num_task_to_eval, task_generator, seed, env_name, log_dir):
        n_iter = num_task_to_eval // self.num_processes
        reward_iter = torch.zeros(self.num_processes)
        r_list = []
        mse_list_at_horizon = []
        mse_list_at_10 = []
        mse_list_at_50 = []
        mse_list_at_0 = []

        for _ in range(n_iter):
            envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)

            eval_envs = make_vec_envs_multi_task(env_name,
                                                 seed,
                                                 self.num_processes,
                                                 None,
                                                 log_dir,
                                                 self.device,
                                                 False,
                                                 envs_kwargs,
                                                 num_frame_stack=None)

            obs = eval_envs.reset()

            obs = al_augment_obs(obs=obs, latent_dim=self.latent_dim, posterior=prior_list,
                                 prior=prior_list,
                                 rescale_obs=self.rescale_obs, max_old=self.max_old,
                                 min_old=self.min_old)

            epi_done = []

            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, self.actor_critic_identification.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

            t = 0

            use_prev_state = False
            while len(epi_done) < self.num_processes:
                with torch.no_grad():
                    _, action, _, eval_recurrent_hidden_states = self.actor_critic_identification.act(
                        obs,
                        eval_recurrent_hidden_states,
                        eval_masks,
                        deterministic=False)

                # Observe reward and next obs
                if t == 0:
                    mse_list_at_0.append(get_mse(prior_list, new_tasks, self.latent_dim, self.num_processes).item())

                obs, reward, done, infos = eval_envs.step(action)
                posterior = get_posterior_no_prev(self.vae, action, reward, prior_list,
                                                  use_prev_state=use_prev_state, max_action=self.max_action,
                                                  min_action=self.min_action)
                reward = get_reward(posterior, new_tasks, self.latent_dim, self.num_processes)

                obs = al_augment_obs(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                     prior=prior_list,
                                     rescale_obs=self.rescale_obs, max_old=self.max_old,
                                     min_old=self.min_old)
                use_prev_state = True
                reward_iter += reward.squeeze()
                t = t + 1

                if t == 10:
                    mse_list_at_10.append(get_mse(posterior, new_tasks, self.latent_dim, self.num_processes).item())
                if t == 50:
                    mse_list_at_50.append(get_mse(posterior, new_tasks, self.latent_dim, self.num_processes).item())

                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device)

                for info in infos:
                    if 'episode' in info.keys():
                        epi_done.append(True)

            mse_list_at_horizon.append(get_mse(posterior, new_tasks, self.latent_dim, self.num_processes).item())
            r_list.append(reward_iter.mean().item())
            eval_envs.close()

        mean_mse_0 = np.mean(mse_list_at_0)
        mean_mse_horizon = np.mean(mse_list_at_horizon)
        mean_mse_10 = np.mean(mse_list_at_10)
        mean_mse_50 = np.mean(mse_list_at_50)
        mean_r = np.mean(r_list)
        print("Evaluation using {} tasks. Mean reward: {}. Mean MSE: {:.2f} || {:.2f} || {:.2f} || {:.2f}".
              format(n_iter * self.num_processes, mean_r, mean_mse_0, mean_mse_10, mean_mse_50, mean_mse_horizon))
        return mean_r, r_list

    def meta_test(self, gp_list, sw_size, env_name, seed, log_dir, envs_kwargs_list, init_prior, use_env_obs,
                  num_eval_processes, max_id_iteration):
        print("Meta-testing...")

        num_tasks = len(envs_kwargs_list)

        eval_episode_rewards = []

        prior = init_prior
        posterior_history = torch.empty(num_tasks, num_eval_processes, 2 * self.latent_dim)

        for t, kwargs in enumerate(envs_kwargs_list):
            # Task creation
            temp = [kwargs for _ in range(num_eval_processes)]
            eval_envs = make_vec_envs_multi_task(env_name, seed, num_eval_processes, self.gamma, log_dir, self.device,
                                                 False, temp, num_frame_stack=None)

            obs = eval_envs.reset()
            obs = al_augment_obs(obs, self.latent_dim, prior, prior,
                                 rescale_obs=self.rescale_obs,
                                 max_old=self.max_old, min_old=self.min_old)

            eval_recurrent_hidden_states = torch.zeros(
                num_eval_processes, self.actor_critic_identification.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(num_eval_processes, 1, device=self.device)

            use_prev_state = False

            task_epi_rewards = []

            iteration = 0
            while len(task_epi_rewards) < num_eval_processes:
                if iteration < max_id_iteration:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = self.actor_critic_identification.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks,
                            deterministic=False)

                    # Observe reward and next obs
                    obs, reward, done, infos = eval_envs.step(action)
                    posterior = get_posterior_no_prev(self.vae, action, reward, prior,
                                                      min_action=self.min_action, max_action=self.max_action,
                                                      use_prev_state=use_prev_state)
                    obs = al_augment_obs(obs, self.latent_dim, posterior,
                                         prior, rescale_obs=self.rescale_obs,
                                         max_old=self.max_old, min_old=self.min_old)
                else:
                    if iteration == max_id_iteration:
                        eval_recurrent_hidden_states = torch.zeros(
                            num_eval_processes, self.actor_critic_optimal.recurrent_hidden_state_size,
                            device=self.device)
                        eval_masks = torch.zeros(num_eval_processes, 1, device=self.device)
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = self.actor_critic_optimal.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks,
                            deterministic=False
                        )
                    obs, reward, done, infos = eval_envs.step(action)
                    obs = oracle_augment_obs(obs=obs, latent=posterior[:, 0:self.latent_dim],
                                             latent_dim=self.latent_dim, use_env_obs=use_env_obs)

                use_prev_state = True
                eval_masks = torch.tensor(
                    [[0.0] if done_ else [1.0] for done_ in done],
                    dtype=torch.float32,
                    device=self.device)

                for info in infos:
                    if 'episode' in info.keys():
                        total_epi_reward = info['episode']['r']
                        task_epi_rewards.append(total_epi_reward)

            eval_episode_rewards.append(np.mean(task_epi_rewards))
            eval_envs.close()

            # Retrieve new prior for the identified model so far
            posterior_history[t, :, :] = posterior
            x = np.atleast_2d(np.arange(t + 1)).T
            for dim in range(self.latent_dim):
                for proc in range(num_eval_processes):
                    if t > sw_size:
                        gp_list[dim][proc].fit(x[-sw_size:],
                                               np.atleast_2d(
                                                   posterior_history[t + 1 - sw_size:t + 1, proc, dim].numpy()).T)
                    else:
                        gp_list[dim][proc].fit(x,
                                               np.atleast_2d(posterior_history[0:t + 1, proc, dim].numpy()).T)

            prior = []
            for proc in range(num_eval_processes):
                prior_proc = torch.empty(self.latent_dim, 2)
                for dim in range(self.latent_dim):
                    x_points = np.atleast_2d(np.array([t + 1])).T
                    y_pred, sigma = gp_list[dim][proc].predict(x_points, return_std=True)
                    prior_proc[dim, 0] = y_pred[0, 0]
                    prior_proc[dim, 1] = sigma[0]
                prior.append(prior_proc)

        print("Reward : {}".format(np.mean(eval_episode_rewards)))
        return eval_episode_rewards
