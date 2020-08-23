import torch
import numpy as np

from functools import reduce
from active_learning.observation_utils import get_posterior_no_prev
from network.vae_utils import loss_inference_closed_form
from ppo_a2c.envs import make_vec_envs_multi_task


class PosteriorTSAgent:

    def __init__(self, vi, vi_optim, num_steps, num_processes, device, gamma, latent_dim,
                 use_env_obs, min_action, max_action, max_sigma):
        # Inference network
        self.vi = vi
        self.vi_optim = vi_optim

        self.use_env_obs = use_env_obs

        self.min_action = min_action
        self.max_action = max_action

        self.max_sigma = max_sigma

        # General
        self.num_processes = num_processes
        self.device = device
        self.gamma = gamma

        # Env
        self.num_steps = num_steps
        self.latent_dim = latent_dim

    def train(self, n_train_iter, eval_interval, task_generator, env_name, seed, log_dir, verbose,
              num_random_task_to_evaluate, gp_list_sequences, sw_size, prior_sequences,
              init_prior_sequences, num_eval_processes, use_true_sigma):
        eval_list = []
        test_list = []
        vi_loss = []

        for i in range(n_train_iter):
            loss = self.train_iter(task_generator, env_name, seed, log_dir, verbose)
            vi_loss.append(loss)

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
                                             use_true_sigma=use_true_sigma, task_generator=task_generator,
                                             log_dir=log_dir, seed=seed, env_name=env_name)
                test_list.append(e)

        return vi_loss, eval_list, test_list

    def train_iter(self, task_generator, env_name, seed, log_dir, verbose):
        envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
        envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                        False, envs_kwargs, num_frame_stack=None)

        # Data structure for the loss function
        prior = torch.empty(self.num_processes, self.latent_dim * 2)
        mu_prior = torch.empty(self.num_processes, self.latent_dim)
        logvar_prior = torch.empty(self.num_processes, self.latent_dim)

        for t_idx in range(self.num_processes):
            prior[t_idx] = prior_list[t_idx].reshape(1, self.latent_dim * 2).squeeze(0).clone().detach()
            mu_prior[t_idx] = prior_list[t_idx][0].clone().detach()
            logvar_prior[t_idx] = prior_list[t_idx][1].clone().detach().log()

        envs.reset()

        posterior = torch.tensor([prior_list[i].flatten().tolist() for i in range(self.num_processes)])

        num_data_context = torch.randint(low=1, high=self.num_steps, size=(1,)).item()
        context = torch.empty(self.num_processes, num_data_context, 2)

        for step in range(num_data_context):
            use_prev_state = True if step > 0 else False

            vae_action, env_action = self.pull_action(posterior, self.num_processes)
            obs, reward, done, infos = envs.step(env_action)
            posterior = get_posterior_no_prev(action=env_action, reward=reward, prior=prior,
                                              max_action=self.max_action, min_action=self.min_action,
                                              use_prev_state=use_prev_state, vi=self.vi)
            context[:, step, 0] = vae_action.squeeze(1)
            context[:, step, 1] = reward.squeeze(1)

        self.vi_optim.zero_grad()
        z_hat, mu_hat, logvar_hat = self.vi(context, prior)

        loss, kdl, mse = loss_inference_closed_form(new_tasks, mu_hat, logvar_hat, mu_prior, logvar_prior, None,
                                                    verbose)
        loss.backward()
        self.vi_optim.step()

        return loss

    def pull_action(self, posterior, num_processes):
        vae_action = torch.normal(posterior[:, 0], posterior[:, 1]).reshape(num_processes, 1)
        env_action = (1 - (-1)) / (self.max_action - self.min_action) * (vae_action - self.max_action) + 1
        return vae_action, env_action

    def evaluate(self, num_task_to_evaluate, task_generator, env_name, seed, log_dir):
        assert num_task_to_evaluate % self.num_processes == 0

        n_iter_eval = num_task_to_evaluate // self.num_processes

        r_epi_list = []

        for _ in range(n_iter_eval):
            eval_episode_rewards = []

            envs_kwargs, _, prior_list, new_tasks = task_generator.sample_pair_tasks(self.num_processes)
            envs = make_vec_envs_multi_task(env_name, seed, self.num_processes, self.gamma, log_dir, self.device,
                                            False, envs_kwargs, num_frame_stack=None)

            envs.reset()

            posterior = torch.tensor([prior_list[i].flatten().tolist() for i in range(self.num_processes)])

            use_prev_state = False
            while len(eval_episode_rewards) < self.num_processes:
                _, env_action = self.pull_action(posterior, self.num_processes)

                # Observe reward and next obs
                obs, reward, done, infos = envs.step(env_action)
                posterior = get_posterior_no_prev(self.vi, env_action, reward, prior_list,
                                                  min_action=self.min_action, max_action=self.max_action,
                                                  use_prev_state=use_prev_state)
                use_prev_state = True

                for info in infos:
                    if 'episode' in info.keys():
                        total_epi_reward = info['episode']['r']
                        eval_episode_rewards.append(total_epi_reward)

            r_epi_list.append(eval_episode_rewards)
            envs.close()

        r_epi_list = reduce(list.__add__, r_epi_list)
        print("Evaluation using {} tasks. Mean reward: {}".format(num_task_to_evaluate, np.mean(r_epi_list)))
        return np.mean(r_epi_list)

    def meta_test_sequences(self, gp_list_sequences, sw_size, env_name, seed, log_dir, prior_sequences,
                            init_prior_sequences, num_eval_processes, task_generator,
                            use_true_sigma=False):
        r_all = []
        r_all_real_prior = []
        for seq_idx, s in enumerate(prior_sequences):
            env_kwargs_list = [task_generator.sample_task_from_prior(s[i]) for i in range(len(s))]
            p = [init_prior_sequences[seq_idx][0] for _ in range(num_eval_processes)]

            r, _, _ = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size, env_name, seed, log_dir,
                                              env_kwargs_list, p,
                                              num_eval_processes, use_true_sigma, False)
            print("Using GP {}".format(np.mean(r)))
            r_all.append(np.mean(r))

            r, _, _ = self.test_task_sequence(gp_list_sequences[seq_idx], sw_size, env_name, seed, log_dir,
                                              env_kwargs_list, p,
                                              num_eval_processes, use_true_sigma,
                                              True, s)
            r_all_real_prior.append(np.mean(r))
            print("Using real prior {}".format(np.mean(r)))

        return r_all, r_all_real_prior

    def test_task_sequence(self, gp_list, sw_size, env_name, seed, log_dir, envs_kwargs_list, init_prior,
                           num_eval_processes, use_true_sigma, use_real_prior, true_prior_sequence=None):
        print("Meta-testing...")

        num_tasks = len(envs_kwargs_list)

        eval_episode_rewards = []
        prediction_mean = []

        prior = init_prior
        posterior_history = torch.empty(num_tasks, num_eval_processes, 2 * self.latent_dim)

        for t, kwargs in enumerate(envs_kwargs_list):
            # Task creation
            temp = [kwargs for _ in range(num_eval_processes)]
            eval_envs = make_vec_envs_multi_task(env_name, seed, num_eval_processes, self.gamma, log_dir, self.device,
                                                 False, temp, num_frame_stack=None)

            eval_envs.reset()

            posterior = torch.tensor([prior[i].flatten().tolist() for i in range(num_eval_processes)])

            use_prev_state = False
            task_epi_rewards = []

            while len(task_epi_rewards) < num_eval_processes:
                _, env_action = self.pull_action(posterior, num_eval_processes)

                obs, reward, done, infos = eval_envs.step(env_action)
                posterior = get_posterior_no_prev(self.vi, env_action, reward, prior,
                                                  min_action=self.min_action, max_action=self.max_action,
                                                  use_prev_state=use_prev_state)

                use_prev_state = True
                for info in infos:
                    if 'episode' in info.keys():
                        total_epi_reward = info['episode']['r']
                        task_epi_rewards.append(total_epi_reward)

            eval_episode_rewards.append(np.mean(task_epi_rewards))
            eval_envs.close()

            # Retrieve new prior for the identified model so far
            if use_real_prior and t + 1 < len(true_prior_sequence):
                prior = [true_prior_sequence[t + 1] for _ in range(num_eval_processes)]
            else:
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
                curr_pred = []
                for proc in range(num_eval_processes):
                    prior_proc = torch.empty(2, self.latent_dim)
                    for dim in range(self.latent_dim):
                        x_points = np.atleast_2d(np.array([t + 1])).T
                        y_pred, sigma = gp_list[dim][proc].predict(x_points, return_std=True)
                        prior_proc[0, dim] = y_pred[0, 0]
                        if use_true_sigma:
                            prior_proc[1, dim] = sigma[0]
                        else:
                            prior_proc[1, dim] = self.max_sigma
                        curr_pred.append(y_pred[0][0])
                    prior.append(prior_proc)
                prediction_mean.append(np.mean(curr_pred))

        return eval_episode_rewards, posterior_history, prediction_mean
