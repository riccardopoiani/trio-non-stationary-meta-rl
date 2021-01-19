from functools import reduce

import numpy as np
import torch

from inference.inference_utils import loss_inference_closed_form
from ppo.algo.ppo import PPO
from ppo.envs import get_vec_envs_multi_task
from ppo.model import MLPBase, Policy, MLPFeatureExtractor
from ppo.storage import RolloutStorage
from utilities.observation_utils import augment_obs_posterior, get_posterior


class BayesAgent:

    def __init__(self,
                 action_space,
                 device,
                 gamma,
                 num_steps,
                 num_processes,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr,
                 eps,
                 max_grad_norm,
                 use_linear_lr_decay,
                 use_gae,
                 gae_lambda,
                 use_proper_time_limits,
                 obs_shape,
                 latent_dim,
                 recurrent_policy,
                 hidden_size,
                 use_elu,
                 variational_model,
                 vae_optim,
                 vae_min_seq,
                 vae_max_seq,
                 max_sigma,
                 use_decay_kld,
                 decay_kld_rate,
                 env_dim,
                 action_dim,
                 min_sigma,
                 use_xavier,
                 use_rms_latent,
                 use_rms_obs,
                 use_rms_rew,
                 decouple_rms,
                 use_feature_extractor,
                 state_extractor_dim,
                 latent_extractor_dim,
                 uncertainty_extractor_dim,
                 use_huber_loss,
                 detach_every):
        """
        :param action_space: action space of the environment that the agent will learn to solve
        :param device: device that will be used to run the code
        :param gamma: discount factor of the problem
        :param num_steps: number of steps that the agent will collect before updating
        :param num_processes: number of processes that will run in parallel before each update. This
        corresponds also to the number of batches for training variational inference.
        :param clip_param: clip parameter of PPO algorithm
        :param ppo_epoch: number of PPO epochs update for each training iteration
        :param num_mini_batch: number of mini batches that will be used in PPO
        :param value_loss_coef: value loss coefficient for PPO algorithm
        :param entropy_coef: entropy coefficient of PPO algorithm
        :param lr: learning rate for Adam optimizer used in PPO
        :param eps: epsilon parameter for Adam optimizer used in PPO
        :param max_grad_norm: maximum gradient norm used in PPO
        :param use_linear_lr_decay: whether to use a linear decay of the learning rate. Unsupported at the moment
        :param use_gae: whether to use or not Generalized Advantage Estimation in PPO updates
        :param gae_lambda: lambda parameter of Generalized Advantage Estimation in PPO updates
        :param use_proper_time_limits: whether to use proper time limits or not. If False, time limits will be
        considered at the same way of terminal states
        :param obs_shape: shape of the observation. This contains environment state,
        latent mean estimation and uncertainty about the latent space
        :param latent_dim: dimension of the latent space
        :param recurrent_policy: whether to use a recurrent policy
        :param hidden_size: size of the hidden layers of the networks
        :param use_elu: True if hidden layers of policy networks should use ELU actiovation function. If False
        is used, Tanh will be used
        :param variational_model: variational inference network
        :param vae_optim: optimizer for the variational inference network
        :param vae_min_seq: minimum number of samples per batch that will be used to train VAE
        :param vae_max_seq: maximum number of samples per batch that will be used to train VAE
        :param max_sigma: maximum std that can be set at meta-test time at the end of each task for the prediction
        of the prior of the next task
        :param min_sigma: minimum std that can be set at meta-test time at the end of each task for the prediction
        of the prior of the next task
        :param use_decay_kld: True if the weight of KLD loss used in inference training should decrease as the
        number of samples increase
        :param decay_kld_rate: Initial weight of KLD loss used in inference training when the network as seen
        only 1 sample. This should be considered only in the case in which use_decay_kld is True
        :param env_dim: state dimension of environment observation
        :param action_dim: action dimension
        :param use_xavier: if True xavier init will be used for the Policy network; if False orthogonal init will
        be used
        :param use_rms_latent: True if latent space should be smoothed when the input is fed to the policy
        :param use_rms_obs: True if env. state observation should be smoothed when the input is
        :param use_rms_rew: True if reward should be smoothed in PPO updates
        :param decouple_rms: True if two different smoothers should be used for the latent space (1 for the mean
        estimation and 1 for the variance) should be used in policy input
        :param use_feature_extractor: whether to use or not a more complex policy network that can use smoother
        and feature extractors layers
        :param state_extractor_dim: dimension of the state feature extractor to be used at the beginning of the policy
        :param latent_extractor_dim: dimension of the latent mean estimation feature extraction layer to be
        used at the beginning of the policy
        :param uncertainty_extractor_dim: dimension of the latent variance feature extraction layer to be used
        at the beginning of the policy
        :param use_huber_loss: whether to use a Huber loss in RL training or not
        :param detach_every: if it is not None, than VAE back-propagation through time will stop after this number
        of steps
        """
        # General parameters
        self.device = device
        self.gamma = gamma
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_shape = obs_shape  # env shape + obs shape + 1 if use time True
        self.latent_dim = latent_dim
        self.action_space = action_space

        self.env_dim = env_dim
        self.action_dim = action_dim

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

        # Observation smoother
        self.use_rms_rew = use_rms_rew

        # Variational inference parameters
        self.vae_min_seq = vae_min_seq
        self.vae_max_seq = vae_max_seq
        self.vae = variational_model
        self.vae_optim = vae_optim
        self.use_decay_kld = use_decay_kld
        self.decay_kld_rate = decay_kld_rate
        self.detach_every = detach_every

        # PPO parameters
        self.use_linear_lr_decay = use_linear_lr_decay
        self.use_gae = use_gae
        self.gae_lambda = gae_lambda
        self.use_proper_time_limits = use_proper_time_limits

        if use_feature_extractor:
            base = MLPFeatureExtractor
            self.actor_critic = Policy(self.obs_shape,
                                       self.action_space, base=base,
                                       base_kwargs={'hidden_size': hidden_size,
                                                    'use_elu': use_elu,
                                                    'use_xavier': use_xavier,
                                                    'state_dim': self.env_dim,
                                                    'latent_dim': self.latent_dim,
                                                    'latent_extractor_dim': latent_extractor_dim,
                                                    'state_extractor_dim': state_extractor_dim,
                                                    'has_uncertainty': True,
                                                    'uncertainty_extractor_dim': uncertainty_extractor_dim,
                                                    'norm_state': use_rms_obs,
                                                    'norm_latent': use_rms_latent,
                                                    'decouple_latent_rms': decouple_rms
                                                    })
        else:
            base = MLPBase
            self.actor_critic = Policy(self.obs_shape,
                                       self.action_space, base=base,
                                       base_kwargs={
                                           'recurrent': recurrent_policy,
                                           'hidden_size': hidden_size,
                                           'use_elu': use_elu,
                                           'use_xavier': use_xavier
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

    def train(self, training_iter, env_name, seed, task_generator,
              eval_interval, num_random_task_to_eval, init_vae_steps, task_len,
              num_test_processes, prior_sequences=None, gp_list_sequences=None, sw_size=None,
              init_prior_test_sequences=None,
              log_dir=".", use_env_obs=False, verbose=True,
              vae_smart=False):
        assert len(prior_sequences) == len(init_prior_test_sequences)

        eval_list = []
        test_list = []
        vae_list = []

        for k in range(init_vae_steps):
            res_vae = self.vae_step(use_env_obs=use_env_obs,
                                    task_generator=task_generator, env_name=env_name, seed=seed, log_dir=log_dir,
                                    verbose=verbose, init_vae=False, epoch=k)
            vae_list.append(res_vae)

        for k in range(training_iter):
            # Variational training step
            if vae_smart:
                if np.random.rand() < 0.5:
                    res_vae = self.vae_step_wrong_prior(use_env_obs=use_env_obs, task_generator=task_generator,
                                                        env_name=env_name, seed=seed, log_dir=log_dir,
                                                        verbose=verbose, epoch=k + init_vae_steps)
                else:
                    res_vae = self.vae_step(use_env_obs=use_env_obs, task_generator=task_generator, env_name=env_name,
                                            seed=seed, log_dir=log_dir, verbose=verbose, init_vae=False,
                                            epoch=k + init_vae_steps)
            else:
                res_vae = self.vae_step(use_env_obs=use_env_obs,
                                        task_generator=task_generator, env_name=env_name, seed=seed, log_dir=log_dir,
                                        verbose=verbose, init_vae=False, epoch=k + init_vae_steps)
            vae_list.append(res_vae)

            # Optimal policy training step
            envs_kwargs, _, prior, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            self.envs = get_vec_envs_multi_task(env_name=env_name,
                                                seed=seed,
                                                num_processes=self.num_processes,
                                                gamma=self.gamma,
                                                log_dir=log_dir,
                                                device=self.device,
                                                allow_early_resets=True,
                                                env_kwargs_list=envs_kwargs,
                                                envs=self.envs,
                                                normalize_rew=self.use_rms_rew,
                                                num_frame_stack=None)
            self.multi_task_policy_step(prior, use_env_obs)

            # Evaluation
            if eval_interval is not None and k % eval_interval == 0 and k > 1:
                print("Epoch {} / {}".format(k, training_iter))

                e = self.evaluate(num_random_task_to_eval, task_generator, log_dir, seed, use_env_obs, env_name)
                eval_list.append(e)
                e = self.meta_test_sequences(gp_list_sequences=gp_list_sequences,
                                             sw_size=sw_size,
                                             env_name=env_name,
                                             seed=seed,
                                             log_dir=log_dir,
                                             prior_sequences=prior_sequences,
                                             init_prior_sequences=init_prior_test_sequences,
                                             use_env_obs=use_env_obs,
                                             num_eval_processes=num_test_processes,
                                             task_generator=task_generator,
                                             task_len=task_len,
                                             store_history=False)
                test_list.append(e)

        final_meta_sequence_result = self.meta_test_sequences(gp_list_sequences=gp_list_sequences,
                                                              sw_size=sw_size,
                                                              env_name=env_name,
                                                              seed=seed,
                                                              log_dir=log_dir,
                                                              prior_sequences=prior_sequences,
                                                              init_prior_sequences=init_prior_test_sequences,
                                                              use_env_obs=use_env_obs,
                                                              num_eval_processes=num_test_processes,
                                                              task_generator=task_generator,
                                                              task_len=task_len,
                                                              store_history=True)

        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()

        return eval_list, vae_list, test_list, final_meta_sequence_result

    def vae_step_wrong_prior(self, use_env_obs, task_generator, env_name, seed, log_dir, verbose,
                             epoch):
        train_loss = 0
        mse_train_loss = 0
        kdl_train_loss = 0

        envs_kwargs, prev_task, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name=env_name,
                                            seed=seed,
                                            num_processes=self.num_processes,
                                            gamma=self.gamma,
                                            log_dir=log_dir,
                                            device=self.device,
                                            allow_early_resets=True,
                                            env_kwargs_list=envs_kwargs,
                                            envs=self.envs,
                                            normalize_rew=self.use_rms_rew,
                                            num_frame_stack=None)
        _, _, prior_list_policy, _ = task_generator.sample_pair_tasks(self.num_processes)

        # Data structure for the loss function
        prior = torch.empty(self.num_processes, self.latent_dim * 2)
        mu_prior = torch.empty(self.num_processes, self.latent_dim)
        logvar_prior = torch.empty(self.num_processes, self.latent_dim)
        prior_policy = torch.empty(self.num_processes, self.latent_dim * 2)

        for t_idx in range(self.num_processes):
            prior[t_idx] = prior_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()
            mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
            logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()
            prior_policy[t_idx] = prior_list_policy[t_idx].reshape(1, self.latent_dim * 2).squeeze(
                0).clone().detach()

        # Sample data under the current policy
        obs = self.envs.reset()

        obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=prior_policy,
                                    use_env_obs=use_env_obs, is_prior=True)

        rollouts_multi_task = RolloutStorage(self.vae_max_seq, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)
        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        num_data_context = torch.randint(low=self.vae_min_seq, high=self.vae_max_seq, size=(1,)).item()
        context = torch.empty(self.num_processes, num_data_context, 1 + self.env_dim + self.action_dim)

        for step in range(num_data_context):
            use_prev_state = True if step > 0 else 0

            # Sample context under
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                    rollouts_multi_task.masks[step])

            obs, (reward, reward_norm), done, infos = self.envs.step(action)

            if use_env_obs:
                context[:, step, 1 + self.action_dim:] = obs

            posterior = get_posterior(vi=self.vae, action=action, reward=reward, prior=prior,
                                      use_prev_state=use_prev_state,
                                      env_obs=obs, use_env_obs=use_env_obs)
            obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                        is_prior=False, use_env_obs=use_env_obs)

            context[:, step, 0:self.action_dim] = action
            context[:, step, self.action_dim] = reward.squeeze(1)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                       action_log_prob, value, reward, masks, bad_masks)

        # Now that data have been collected, we train the variational model
        self.vae_optim.zero_grad()
        z_hat, mu_hat, logvar_hat = self.vae(context, prior, detach_every=self.detach_every)

        loss, kdl, mse = loss_inference_closed_form(z=new_tasks,
                                                    mu_hat=mu_hat,
                                                    logvar_hat=logvar_hat,
                                                    mu_prior=mu_prior,
                                                    logvar_prior=logvar_prior,
                                                    n_samples=num_data_context,
                                                    use_decay=self.use_decay_kld,
                                                    decay_param=self.decay_kld_rate,
                                                    epoch=epoch,
                                                    verbose=verbose
                                                    )
        loss.backward()
        train_loss += loss.item()
        mse_train_loss += mse
        kdl_train_loss += kdl
        self.vae_optim.step()

        return train_loss, mse_train_loss, kdl_train_loss

    def vae_step(self, use_env_obs, task_generator, env_name, seed, log_dir, verbose, init_vae, epoch):
        train_loss = 0
        mse_train_loss = 0
        kdl_train_loss = 0

        envs_kwargs, prev_task, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name=env_name,
                                            seed=seed,
                                            num_processes=self.num_processes,
                                            gamma=self.gamma,
                                            log_dir=log_dir,
                                            device=self.device,
                                            allow_early_resets=True,
                                            env_kwargs_list=envs_kwargs,
                                            envs=self.envs,
                                            normalize_rew=self.use_rms_rew,
                                            num_frame_stack=None)
        # Data structure for the loss function
        prior = torch.empty(self.num_processes, self.latent_dim * 2)
        mu_prior = torch.empty(self.num_processes, self.latent_dim)
        logvar_prior = torch.empty(self.num_processes, self.latent_dim)

        for t_idx in range(self.num_processes):
            prior[t_idx] = prior_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()
            mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
            logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()

        # Sample data under the current policy
        obs = self.envs.reset()

        obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=prior, use_env_obs=use_env_obs,
                                    is_prior=True)

        rollouts_multi_task = RolloutStorage(self.vae_max_seq, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)
        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        num_data_context = torch.randint(low=self.vae_min_seq, high=self.vae_max_seq, size=(1,)).item()
        context = torch.empty(self.num_processes, num_data_context, 1 + self.env_dim + self.action_dim)

        for step in range(num_data_context):
            use_prev_state = True if step > 0 else 0

            # Sample context under
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                    rollouts_multi_task.masks[step])

            obs, (reward, reward_norm), done, infos = self.envs.step(action)
            if use_env_obs:
                context[:, step, 1 + self.action_dim:] = obs
            posterior = get_posterior(vi=self.vae, action=action, reward=reward, prior=prior,
                                      use_prev_state=use_prev_state,
                                      use_env_obs=use_env_obs, env_obs=obs)
            obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                        use_env_obs=use_env_obs,
                                        is_prior=False)
            context[:, step, 0:self.action_dim] = action
            context[:, step, self.action_dim] = reward.squeeze(1)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                       action_log_prob, value, reward, masks, bad_masks)

        # Now that data have been collected, we train the variational model
        self.vae_optim.zero_grad()
        z_hat, mu_hat, logvar_hat = self.vae(context, prior, detach_every=self.detach_every)

        loss, kdl, mse = loss_inference_closed_form(z=new_tasks,
                                                    mu_hat=mu_hat,
                                                    logvar_hat=logvar_hat,
                                                    mu_prior=mu_prior,
                                                    logvar_prior=logvar_prior,
                                                    n_samples=num_data_context,
                                                    use_decay=self.use_decay_kld,
                                                    decay_param=self.decay_kld_rate,
                                                    epoch=epoch,
                                                    verbose=verbose
                                                    )
        loss.backward()
        train_loss += loss.item()
        mse_train_loss += mse
        kdl_train_loss += kdl
        self.vae_optim.step()

        return train_loss, mse_train_loss, kdl_train_loss

    def multi_task_policy_step(self, prior, use_env_obs):
        # Multi-task learning with posterior mean
        obs = self.envs.reset()

        obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=prior,
                                    use_env_obs=use_env_obs, is_prior=True)

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
            obs, (reward, reward_norm), done, infos = self.envs.step(action)
            posterior = get_posterior(vi=self.vae, action=action, reward=reward, prior=prior,
                                      use_prev_state=use_prev_state,
                                      use_env_obs=use_env_obs, env_obs=obs)

            obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                        use_env_obs=use_env_obs, is_prior=False)

            # If done then clean the history of observations.
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

    def evaluate(self, num_task_to_evaluate, task_generator, log_dir, seed, use_env_obs, env_name):
        assert num_task_to_evaluate % self.num_processes == 0

        print("Evaluation...")

        n_iter = num_task_to_evaluate // self.num_processes
        r_epi_list = []

        for num_iteration in range(n_iter):
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

            obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=prior,
                                        use_env_obs=use_env_obs, is_prior=True)

            eval_recurrent_hidden_states = torch.zeros(
                self.num_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
            eval_masks = torch.zeros(self.num_processes, 1, device=self.device)

            use_prev_state = False
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
                posterior = get_posterior(vi=self.vae, action=action, reward=reward, prior=prior,
                                          use_prev_state=use_prev_state, use_env_obs=use_env_obs,
                                          env_obs=obs)
                obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                            use_env_obs=use_env_obs, is_prior=False)

                use_prev_state = True
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

    def meta_test_sequences(self, gp_list_sequences, sw_size, env_name, seed, log_dir, prior_sequences,
                            init_prior_sequences, use_env_obs, num_eval_processes, task_generator,
                            task_len, store_history=False, verbose=False, t_id=0):
        r_all_true_sigma = []
        r_all_false_sigma = []
        r_all_real_prior = []
        r_all_no_tracking = []

        prediction_mean_true = []
        posterior_history_true = []
        prediction_mean_false = []
        posterior_history_false = []

        for seq_idx, s in enumerate(prior_sequences):
            env_kwargs_list = [task_generator.sample_task_from_prior(s[i]) for i in range(len(s))]

            r, posterior_history, prediction_mean = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size,
                                                                            env_name, seed, log_dir,
                                                                            env_kwargs_list,
                                                                            init_prior_sequences[seq_idx],
                                                                            use_env_obs, num_eval_processes,
                                                                            task_len=task_len,
                                                                            use_true_sigma=True,
                                                                            use_real_prior=False,
                                                                            verbose=True)
            r_all_true_sigma.append(r)
            prediction_mean_true.append(prediction_mean)
            posterior_history_true.append(posterior_history)

            r, posterior_history, prediction_mean = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size,
                                                                            env_name, seed, log_dir,
                                                                            env_kwargs_list,
                                                                            init_prior_sequences[seq_idx],
                                                                            use_env_obs, num_eval_processes,
                                                                            task_len=task_len,
                                                                            use_true_sigma=False,
                                                                            use_real_prior=False)
            r_all_false_sigma.append(r)
            posterior_history_false.append(posterior_history)
            prediction_mean_false.append(prediction_mean)

            r, _, _ = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size, env_name, seed, log_dir,
                                              env_kwargs_list, init_prior_sequences[seq_idx],
                                              use_env_obs, num_eval_processes,
                                              task_len=task_len, use_true_sigma=None,
                                              use_real_prior=True, true_prior_sequence=s)
            r_all_real_prior.append(r)

            r, _, _ = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size, env_name, seed, log_dir,
                                              env_kwargs_list, init_prior_sequences[seq_idx],
                                              use_env_obs, num_eval_processes,
                                              task_len=task_len,
                                              use_true_sigma=None,
                                              use_real_prior=False,
                                              no_tracking=True)
            r_all_no_tracking.append(r)

        if store_history:
            return r_all_true_sigma, r_all_false_sigma, r_all_real_prior, r_all_no_tracking, posterior_history_true, prediction_mean_true, \
                   posterior_history_false, prediction_mean_false
        else:
            return r_all_true_sigma, r_all_false_sigma, r_all_real_prior, r_all_no_tracking

    def test_task_sequence(self, gp_list, sw_size, env_name, seed, log_dir, envs_kwargs_list, init_prior, use_env_obs,
                           num_eval_processes, use_true_sigma, use_real_prior, true_prior_sequence=None, verbose=False,
                           task_len=1, no_tracking=False):
        num_tasks = len(envs_kwargs_list)

        eval_episode_rewards = []
        prediction_mean = []

        prior = init_prior
        posterior_history = torch.empty(num_tasks, num_eval_processes, 2 * self.latent_dim)

        for t, kwargs in enumerate(envs_kwargs_list):
            use_prev_state = False
            task_r = []
            for _ in range(task_len):
                # Task creation
                self.eval_envs = get_vec_envs_multi_task(env_name=env_name,
                                                         seed=seed,
                                                         num_processes=num_eval_processes,
                                                         gamma=self.gamma,
                                                         log_dir=log_dir,
                                                         device=self.device,
                                                         allow_early_resets=True,
                                                         env_kwargs_list=[kwargs for _ in range(num_eval_processes)],
                                                         envs=None,
                                                         normalize_rew=self.use_rms_rew,
                                                         num_frame_stack=None)
                obs = self.eval_envs.reset()

                if not use_prev_state:
                    obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=prior,
                                                use_env_obs=use_env_obs, is_prior=True)
                else:
                    obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim,
                                                posterior=posterior_history[t, :, :],
                                                use_env_obs=use_env_obs, is_prior=False)

                eval_recurrent_hidden_states = torch.zeros(
                    num_eval_processes, self.actor_critic.recurrent_hidden_state_size, device=self.device)
                eval_masks = torch.zeros(num_eval_processes, 1, device=self.device)

                task_epi_rewards = []
                already_ended = torch.zeros(num_eval_processes, dtype=torch.bool)
                while len(task_epi_rewards) < num_eval_processes:
                    with torch.no_grad():
                        _, action, _, eval_recurrent_hidden_states = self.actor_critic.act(
                            obs,
                            eval_recurrent_hidden_states,
                            eval_masks,
                            deterministic=False)

                    # Observe reward and next obs
                    obs, (reward, reward_norm), done, infos = self.eval_envs.step(action)
                    posterior = get_posterior(vi=self.vae, action=action, reward=reward, prior=prior,
                                              use_prev_state=use_prev_state, use_env_obs=use_env_obs,
                                              env_obs=obs)

                    obs = augment_obs_posterior(obs=obs, latent_dim=self.latent_dim, posterior=posterior,
                                                use_env_obs=use_env_obs, is_prior=False)

                    use_prev_state = True
                    eval_masks = torch.tensor(
                        [[0.0] if done_ else [1.0] for done_ in done],
                        dtype=torch.float32,
                        device=self.device)

                    for i, info in enumerate(infos):
                        if 'episode' in info.keys() and not already_ended[i]:
                            posterior_history[t, i, :] = posterior[i].clone().detach()
                            total_epi_reward = info['episode']['r']
                            task_epi_rewards.append(total_epi_reward)
                            task_r.append(total_epi_reward)
                    already_ended = already_ended | done

            eval_episode_rewards.append(np.mean(task_r))

            # Retrieve new prior for the identified model so far
            if no_tracking:
                prior = init_prior
            elif use_real_prior and t + 1 < len(true_prior_sequence):
                prior = [true_prior_sequence[t + 1] for _ in range(num_eval_processes)]
            elif not use_real_prior:
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
                curr_pred_all = [[] for _ in range(self.latent_dim)]
                for proc in range(num_eval_processes):
                    prior_proc = torch.empty(2, self.latent_dim)

                    for dim in range(self.latent_dim):
                        x_points = np.atleast_2d(np.array([t + 1])).T
                        y_pred, sigma = gp_list[dim][proc].predict(x_points, return_std=True)
                        prior_proc[0, dim] = y_pred[0, 0]
                        if use_true_sigma:
                            if self.min_sigma[dim] > sigma[0]:
                                prior_proc[1, dim] = self.min_sigma[dim] ** 2
                            elif self.max_sigma[dim] < sigma[0]:
                                prior_proc[1, dim] = self.max_sigma[dim] ** 2
                            else:
                                prior_proc[1, dim] = sigma[0] ** 2
                        else:
                            prior_proc[1, dim] = self.max_sigma[dim] ** 2
                        curr_pred_all[dim].append(y_pred[0][0])
                    prior.append(prior_proc)

                curr_pred_all = np.array(curr_pred_all)
                prediction_mean.append(np.mean(curr_pred_all, 1))

        return eval_episode_rewards, posterior_history, prediction_mean
