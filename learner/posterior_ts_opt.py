import torch
import numpy as np

from functools import reduce

from ppo_a2c.algo.ppo import PPO
from ppo_a2c.model import MLPBase, Policy
from ppo_a2c.storage import RolloutStorage
from utilities.observation_utils import get_posterior, augment_obs_optimal, augment_obs_oracle
from inference.inference_utils import loss_inference_closed_form
from ppo_a2c.envs import get_vec_envs_multi_task


def _rescale_action(action, max_new, min_new):
    if max_new is not None or min_new is not None:
        return (max_new - min_new) / (1 - (-1)) * (action - 1) + max_new
    else:
        return action


class PosteriorOptTSAgent:

    def __init__(self,
                 vi,
                 vi_optim,
                 num_steps,
                 num_processes,
                 device,
                 gamma,
                 latent_dim,
                 use_env_obs,
                 min_action,
                 max_action,
                 max_sigma,
                 action_space,
                 obs_shape,
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
                 recurrent_policy,
                 hidden_size,
                 use_elu,
                 rescale_obs,
                 max_old,
                 min_old,
                 use_decay_kld,
                 decay_kld_rate,
                 env_dim,
                 action_dim,
                 min_sigma,
                 vae_max_steps):
        # Inference inference
        self.vi = vi
        self.vi_optim = vi_optim
        self.use_decay_kld = use_decay_kld
        self.decay_kld_rate = decay_kld_rate
        self.vae_max_steps = vae_max_steps

        # General settings
        self.use_env_obs = use_env_obs
        self.env_dim = env_dim
        self.action_dim = action_dim

        self.min_action = min_action
        self.max_action = max_action

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

        # General
        self.num_processes = num_processes
        self.device = device
        self.gamma = gamma
        self.action_space = action_space
        self.obs_shape = obs_shape
        self.rescale_obs = rescale_obs
        self.max_old = max_old
        self.min_old = min_old

        # Env
        self.num_steps = num_steps
        self.latent_dim = latent_dim

        # Optimal policy
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

        self.envs = None
        self.eval_envs = None

    def train(self, n_train_iter, init_vae_steps, eval_interval, task_generator, env_name, seed, log_dir, verbose,
              num_random_task_to_evaluate, gp_list_sequences, sw_size, prior_sequences,
              init_prior_sequences, num_eval_processes, vae_smart,
              task_len):
        eval_list = []
        test_list = []
        vi_loss = []

        for i in range(init_vae_steps):
            self.train_iter_vae(task_generator, env_name, seed, log_dir, verbose, i)

        for i in range(n_train_iter):
            if vae_smart:
                if np.random.rand() < 0.5:
                    loss = self.train_iter_vae_smart(task_generator, env_name, seed, log_dir, verbose,
                                                     i + init_vae_steps)
                else:
                    loss = self.train_iter_vae(task_generator, env_name, seed, log_dir, verbose, i + init_vae_steps)
            else:
                loss = self.train_iter_vae(task_generator, env_name, seed, log_dir, verbose, i + init_vae_steps)

            vi_loss.append(loss)

            self.train_multi_task_iter(task_generator, env_name, seed, log_dir)

            if i % eval_interval == 0:
                print("Iteration {} / {}".format(i, n_train_iter))
                e = self.evaluate(num_task_to_evaluate=num_random_task_to_evaluate,
                                  task_generator=task_generator, env_name=env_name,
                                  seed=seed, log_dir=log_dir)
                eval_list.append(e)

                e = self.meta_test_sequences(gp_list_sequences=gp_list_sequences,
                                             sw_size=sw_size, prior_sequences=prior_sequences,
                                             num_eval_processes=num_eval_processes,
                                             init_prior_sequences=init_prior_sequences,
                                             task_generator=task_generator,
                                             log_dir=log_dir, seed=seed, env_name=env_name,
                                             store_history=False,
                                             task_len=task_len)
                test_list.append(e)

        final_meta_test = self.meta_test_sequences(gp_list_sequences=gp_list_sequences,
                                                   sw_size=sw_size, prior_sequences=prior_sequences,
                                                   num_eval_processes=num_eval_processes,
                                                   init_prior_sequences=init_prior_sequences,
                                                   task_generator=task_generator,
                                                   log_dir=log_dir, seed=seed, env_name=env_name,
                                                   store_history=True,
                                                   task_len=task_len)
        self.envs.close()
        if self.eval_envs is not None:
            self.eval_envs.close()

        return vi_loss, eval_list, test_list, final_meta_test

    def train_multi_task_iter(self, task_generator, env_name, seed, log_dir):
        envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            True, envs_kwargs, self.envs, num_frame_stack=None)

        obs = self.envs.reset()
        obs = augment_obs_oracle(obs=obs, latent_dim=self.latent_dim, tasks=new_tasks,
                                 use_env_obs=self.use_env_obs, rescale_obs=self.rescale_obs,
                                 max_old=self.max_old, min_old=self.min_old)

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
            obs = augment_obs_oracle(obs, self.latent_dim, new_tasks,
                                     self.use_env_obs, rescale_obs=self.rescale_obs,
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

        rollouts_multi_task.compute_returns(next_value, self.use_gae, self.gamma,
                                            self.gae_lambda, self.use_proper_time_limits)

        self.agent.update(rollouts_multi_task)

        rollouts_multi_task.after_update()

    def train_iter_vae_smart(self, task_generator, env_name, seed, log_dir, verbose, epoch):
        envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            True, envs_kwargs, self.envs, num_frame_stack=None)

        _, _, prior_policy_list, _ = task_generator.sample_pair_tasks(self.num_processes)

        # Data structure for the loss function
        prior = torch.empty(self.num_processes, self.latent_dim * 2)
        mu_prior = torch.empty(self.num_processes, self.latent_dim)
        logvar_prior = torch.empty(self.num_processes, self.latent_dim)
        prior_policy = torch.empty(self.num_processes, self.latent_dim * 2)

        for t_idx in range(self.num_processes):
            prior[t_idx] = prior_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()
            mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
            logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()
            prior_policy[t_idx] = prior_policy_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()

        obs = self.envs.reset()
        obs = augment_obs_optimal(obs, self.latent_dim, prior_policy, self.use_env_obs,
                                  is_prior=True, rescale_obs=self.rescale_obs,
                                  max_old=self.max_old, min_old=self.min_old)
        rollouts_multi_task = RolloutStorage(self.vae_max_steps, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)
        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        num_data_context = torch.randint(low=1, high=self.vae_max_steps, size=(1,)).item()
        context = torch.empty(self.num_processes, num_data_context, 1 + self.env_dim + self.action_dim)

        for step in range(num_data_context):
            use_prev_state = True if step > 0 else False

            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                    rollouts_multi_task.masks[step])

            obs, reward, done, infos = self.envs.step(action)

            if self.use_env_obs:
                context[:, step, 1+self.action_dim:] = obs

            posterior = get_posterior(action=action, reward=reward, prior=prior,
                                      max_action=self.max_action, min_action=self.min_action,
                                      use_prev_state=use_prev_state, vi=self.vi,
                                      env_obs=obs, use_env_obs=self.use_env_obs)
            obs = augment_obs_optimal(obs, self.latent_dim, posterior, self.use_env_obs, False,
                                      self.rescale_obs, self.max_old, self.min_old)

            vae_action = _rescale_action(action, self.max_action, self.min_action)
            context[:, step, 0:self.action_dim] = vae_action
            context[:, step, self.action_dim] = reward.squeeze(1)

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                       action_log_prob, value, reward, masks, bad_masks)

        self.vi_optim.zero_grad()
        z_hat, mu_hat, logvar_hat = self.vi(context, prior)

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
        self.vi_optim.step()

        return loss.item()

    def train_iter_vae(self, task_generator, env_name, seed, log_dir, verbose, epoch):
        envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        self.envs = get_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            True, envs_kwargs, self.envs, num_frame_stack=None)

        # Data structure for the loss function
        prior = torch.empty(self.num_processes, self.latent_dim * 2)
        mu_prior = torch.empty(self.num_processes, self.latent_dim)
        logvar_prior = torch.empty(self.num_processes, self.latent_dim)

        for t_idx in range(self.num_processes):
            prior[t_idx] = prior_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()
            mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
            logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()

        obs = self.envs.reset()
        obs = augment_obs_optimal(obs, self.latent_dim, prior, self.use_env_obs,
                                  is_prior=True, rescale_obs=self.rescale_obs,
                                  max_old=self.max_old, min_old=self.min_old)
        rollouts_multi_task = RolloutStorage(self.vae_max_steps, self.num_processes,
                                             self.obs_shape, self.action_space,
                                             self.actor_critic.recurrent_hidden_state_size)
        rollouts_multi_task.obs[0].copy_(obs)
        rollouts_multi_task.to(self.device)

        num_data_context = torch.randint(low=1, high=self.vae_max_steps, size=(1,)).item()
        context = torch.empty(self.num_processes, num_data_context, 1 + self.env_dim + self.action_dim)

        for step in range(num_data_context):
            use_prev_state = True if step > 0 else False

            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                    rollouts_multi_task.obs[step], rollouts_multi_task.recurrent_hidden_states[step],
                    rollouts_multi_task.masks[step])

            obs, reward, done, infos = self.envs.step(action)

            if self.use_env_obs:
                context[:, step, 1+self.action_dim:] = obs

            posterior = get_posterior(action=action, reward=reward, prior=prior,
                                      max_action=self.max_action, min_action=self.min_action,
                                      use_prev_state=use_prev_state, vi=self.vi,
                                      env_obs=obs, use_env_obs=self.use_env_obs)
            obs = augment_obs_optimal(obs, self.latent_dim, posterior, self.use_env_obs, False,
                                      self.rescale_obs, self.max_old, self.min_old)

            vae_action = _rescale_action(action, self.max_action, self.min_action)
            context[:, step, 0:self.action_dim] = vae_action
            context[:, step, self.action_dim] = reward.squeeze(1)

            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_multi_task.insert(obs, recurrent_hidden_states, action,
                                       action_log_prob, value, reward, masks, bad_masks)

        self.vi_optim.zero_grad()
        z_hat, mu_hat, logvar_hat = self.vi(context, prior)

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
        self.vi_optim.step()

        return loss.item()

    def evaluate(self, num_task_to_evaluate, task_generator, env_name, seed, log_dir):
        assert num_task_to_evaluate % self.num_processes == 0

        n_iter_eval = num_task_to_evaluate // self.num_processes

        r_epi_list = []

        for _ in range(n_iter_eval):
            eval_episode_rewards = []

            envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            self.envs = get_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                                True, envs_kwargs, self.envs, num_frame_stack=None)

            obs = self.envs.reset()
            obs = augment_obs_optimal(obs, self.latent_dim, prior_list,
                                      self.use_env_obs, is_prior=True, rescale_obs=self.rescale_obs,
                                      max_old=self.max_old, min_old=self.min_old)
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
                obs, reward, done, infos = self.envs.step(action)
                posterior = get_posterior(self.vi, action, reward, prior_list,
                                          min_action=self.min_action, max_action=self.max_action,
                                          use_prev_state=use_prev_state, use_env_obs=self.use_env_obs,
                                          env_obs=obs)
                obs = augment_obs_optimal(obs, self.latent_dim, posterior, self.use_env_obs,
                                          False, self.rescale_obs, self.max_old, self.min_old)
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
                            init_prior_sequences, num_eval_processes, task_generator, store_history,
                            task_len):
        r_all_true_sigma = []
        r_all_false_sigma = []
        r_all_real_prior = []

        prediction_mean_true = []
        posterior_history_true = []
        prediction_mean_false = []
        posterior_history_false = []

        for seq_idx, s in enumerate(prior_sequences):
            env_kwargs_list = [task_generator.sample_task_from_prior(s[i]) for i in range(len(s))]
            p = [init_prior_sequences[seq_idx][0] for _ in range(num_eval_processes)]

            r, posterior_history, prediction_mean = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size,
                                                                            env_name, seed, log_dir,
                                                                            env_kwargs_list, p,
                                                                            num_eval_processes,
                                                                            use_true_sigma=True,
                                                                            use_real_prior=False,
                                                                            task_len=task_len)
            print("Using GP True sigma{}".format(np.mean(r)))
            r_all_true_sigma.append(r)
            prediction_mean_true.append(prediction_mean)
            posterior_history_true.append(posterior_history)

            r, posterior_history, prediction_mean = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size,
                                                                            env_name, seed, log_dir,
                                                                            env_kwargs_list, p,
                                                                            num_eval_processes,
                                                                            use_true_sigma=False,
                                                                            use_real_prior=False,
                                                                            task_len=task_len)
            print("Using GP False sigma {}".format(np.mean(r)))
            r_all_false_sigma.append(r)
            posterior_history_false.append(posterior_history)
            prediction_mean_false.append(prediction_mean)

            r, _, _ = self.test_task_sequence(gp_list_sequences[seq_idx],
                                              sw_size,
                                              env_name,
                                              seed,
                                              log_dir,
                                              env_kwargs_list, p,
                                              num_eval_processes,
                                              use_true_sigma=None,
                                              use_real_prior=True,
                                              true_prior_sequence=s,
                                              task_len=task_len)
            r_all_real_prior.append(r)
            print("Using real prior {}".format(np.mean(r)))

        if store_history:
            return r_all_true_sigma, r_all_false_sigma, r_all_real_prior, posterior_history_true, prediction_mean_true, \
                   posterior_history_false, prediction_mean_false
        else:
            return r_all_true_sigma, r_all_false_sigma, r_all_real_prior

    def test_task_sequence(self, gp_list, sw_size, env_name, seed, log_dir, envs_kwargs_list, init_prior,
                           num_eval_processes, use_true_sigma, use_real_prior, task_len,
                           true_prior_sequence=None):
        print("Meta-testing...")

        num_tasks = len(envs_kwargs_list)

        eval_episode_rewards = []
        prediction_mean = []

        prior = init_prior
        posterior_history = torch.empty(num_tasks, num_eval_processes, 2 * self.latent_dim)

        for t, kwargs in enumerate(envs_kwargs_list):
            use_prev_state = False
            for _ in range(task_len):
                # Task creation
                temp = [kwargs for _ in range(num_eval_processes)]
                self.eval_envs = get_vec_envs_multi_task(env_name, seed, num_eval_processes, self.gamma, log_dir,
                                                         self.device,
                                                         True, temp, self.eval_envs, num_frame_stack=None)

                obs = self.eval_envs.reset()
                if not use_prev_state:
                    obs = augment_obs_optimal(obs, self.latent_dim, prior,
                                              self.use_env_obs, is_prior=True, rescale_obs=self.rescale_obs,
                                              max_old=self.max_old, min_old=self.min_old)
                else:
                    obs = augment_obs_optimal(obs, self.latent_dim, posterior_history[t, :, :],
                                              self.use_env_obs, is_prior=False, rescale_obs=self.rescale_obs,
                                              max_old=self.max_old, min_old=self.min_old)
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

                    obs, reward, done, infos = self.eval_envs.step(action)
                    posterior = get_posterior(self.vi, action, reward, prior,
                                              min_action=self.min_action, max_action=self.max_action,
                                              use_prev_state=use_prev_state, use_env_obs=self.use_env_obs,
                                              env_obs=obs)
                    obs = augment_obs_optimal(obs, self.latent_dim, posterior,
                                              self.use_env_obs, is_prior=False, rescale_obs=self.rescale_obs,
                                              max_old=self.max_old, min_old=self.min_old)

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
                    already_ended = already_ended | done

                eval_episode_rewards.append(np.mean(task_epi_rewards))

            # Retrieve new prior for the identified model so far
            if use_real_prior and t + 1 < len(true_prior_sequence):
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
                curr_pred = []
                for proc in range(num_eval_processes):
                    prior_proc = torch.empty(2, self.latent_dim)
                    for dim in range(self.latent_dim):
                        x_points = np.atleast_2d(np.array([t + 1])).T
                        y_pred, sigma = gp_list[dim][proc].predict(x_points, return_std=True)
                        prior_proc[0, dim] = y_pred[0, 0]
                        if use_true_sigma:
                            if self.min_sigma[dim] > sigma[0]:
                                prior_proc[1, dim] = self.min_sigma[dim] ** 2
                            else:
                                prior_proc[1, dim] = sigma[0]
                        else:
                            prior_proc[1, dim] = self.max_sigma[dim] ** 2
                        curr_pred.append(y_pred[0][0])
                    prior.append(prior_proc)
                prediction_mean.append(np.mean(curr_pred))

        return eval_episode_rewards, posterior_history, prediction_mean
