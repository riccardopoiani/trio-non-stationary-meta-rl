from functools import reduce
import torch
import torch.nn.functional as F
import numpy as np
from baselines.common.running_mean_std import RunningMeanStd

from ppo_a2c.envs import make_vec_envs_multi_task


def sample_task(task_gen, min_mean, max_mean, min_std, max_std, n_batches=10, test_perc=0, batch_size=128):
    a = 1
    m = (min_mean - max_mean) * torch.rand(1) + max_mean
    s = (min_std - max_std) * torch.rand(1) + max_std

    data = task_gen.get_mixed_data_loader(amplitude=a,
                                          mean=m,
                                          std=s,
                                          num_batches=n_batches,
                                          test_perc=test_perc,
                                          batch_size=batch_size)
    return data, m, s


def sample_prior_dist(dim, mu_min, mu_max, var_min, var_max):
    mu_l = []
    var_l = []
    for i in range(dim):
        mu = (mu_min - mu_max) * torch.rand(1) + mu_max
        var = (var_min - var_max) * torch.rand(1) + var_max

        mu_l.append(mu)
        var_l.append(var)

    return mu_l, var_l


def get_mse(posterior, offset_star, latent_dim, num_processes):
    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_processes)])
    return F.mse_loss(posterior[:, 0:latent_dim], offset_star)


def get_reward(posterior, offset_star, latent_dim, num_processes):
    like = get_like(posterior, offset_star, latent_dim, num_processes)
    return like


def get_posterior(variational_model, action, reward, prior, prev_latent_space, num_processes, use_prev_state=True):
    """
    Feed the variational model with the actual reward to identifiy the latent space
    and get the current reward using the posterior and the true task
    """
    num_proc = action.shape[0]
    prev_latent_space = torch.tensor(prev_latent_space)
    flatten_prior = torch.tensor([prior[i].flatten().tolist() for i in range(num_processes)])

    # To feed VI, i need (n_batch, 1, 2)
    context = torch.empty(num_proc, 1, 2)
    for i in range(num_proc):
        t = (100 - (-100)) / (1 - (-1)) * (action[i] - 1) + 100
        context[i] = torch.cat([t, reward[i]])

    posterior = variational_model(context=context, prev_z=prev_latent_space, prior=flatten_prior,
                                  use_prev_state=use_prev_state)
    posterior = posterior[1:]
    posterior = torch.cat([posterior[0].detach(), posterior[1].detach()], 1)
    return posterior


def rescale_latent(num_proc, old_var, latent_dim):
    rescaled_latent = []

    max_old = [40, 35]
    min_old = [-40, 15]

    for i in range(num_proc):
        new = []
        for j in range(latent_dim):
            t = (1 - (-1)) / (max_old[j] - min_old[j]) * (old_var[i][j] - max_old[j]) + 1
            if t > 1:
                print("Exceeding max in latent dim {}".format(j))
                t = 1.
            elif t < -1:
                print("Exceeding min in latent dim {}".format(j))
                t = -1
            new.append(t)
        rescaled_latent.append(new)
    return rescaled_latent


def rescale_posterior(num_proc, old_var, latent_dim):
    rescaled_posterior = []

    max_old = [100, 50, 20, 20]
    min_old = [-100, 0, 0, 0]

    for i in range(num_proc):
        new = []
        for j in range(latent_dim * 2):
            t = (1 - (-1)) / (max_old[j] - min_old[j]) * (old_var[i][j] - max_old[j]) + 1
            if t > 1:
                print("Exceeding max in posterior dim {}".format(j))
                t = 1.
            elif t < -1:
                print("Exceeding min in posterior dim {}".format(j))
                t = -1
            new.append(t)
        rescaled_posterior.append(new)

    return torch.tensor(rescaled_posterior)


def al_augment_obs(obs, prev_latent, latent_dim, env_obs_shape, posterior, prior, rescale_obs=True):
    num_proc = obs.shape[0]
    new_obs = torch.empty((num_proc, 2 * latent_dim))

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    else:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    if rescale_obs:
        posterior = rescale_posterior(num_proc, posterior, latent_dim)

    for i in range(num_proc):
        new_obs[i] = posterior[i]

    return new_obs


def get_like(posterior, offset_star, latent_dim, num_processes):
    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_processes)])
        posterior[:, latent_dim:] = posterior[:, latent_dim:].log()

    likelihood = torch.sum(
        -(1 / (2 * posterior[:, latent_dim:].exp())) * (posterior[:, 0:latent_dim] - offset_star).pow(2),
        1).unsqueeze(1)
    return likelihood


def identification_evaluate(actor_critic, vi, env_name, seed, num_processes, eval_log_dir, device,
                            num_task_to_evaluate, latent_dim, env_obs_shape, param,
                            prior_dist, n_tasks, max_horizon=150
                            ):
    assert num_task_to_evaluate % num_processes == 0

    n_iter = num_task_to_evaluate // num_processes
    reward_iter = torch.zeros(num_processes, max_horizon)
    r_list = []
    mse_list_at_horizon = []
    mse_list_at_10 = []
    mse_list_at_50 = []
    mse_list_at_0 = []

    for i in range(n_iter):
        prev_task_idx = torch.randint(low=0, high=n_tasks, size=(num_processes,))
        prev_task = [param[prev_task_idx[i]] for i in range(num_processes, )]

        prior_idx = torch.randint(low=0, high=n_tasks, size=(num_processes,))
        prior = [prior_dist[prior_idx[i]].clone().detach() for i in range(num_processes)]

        # Sample current task from the prior
        mu = [prior[i][0].clone().detach() for i in range(num_processes)]
        var = [prior[i][1].clone().detach() for i in range(num_processes)]

        offset_param = [torch.normal(mu[i], var[i]).tolist() for i in range(num_processes)]
        offset_param = torch.tensor(offset_param)

        # Modify the prior
        for i in range(num_processes):
            prior[i][0, :] = prior[i][0, :] + torch.tensor(prev_task[i])

        mu = [mu[i] + torch.tensor(prev_task[i]) for i in range(num_processes)]
        new_tasks = offset_param + torch.tensor(prev_task)

        # Sample new task
        envs_kwargs = [{'amplitude': 1,
                        'mean': new_tasks[i][0].item(),
                        'std': new_tasks[i][1].item(),
                        'noise_std': 0.001,
                        'scale_reward': False} for i in range(num_processes)]

        obs_rms_eval = ObsSmootherTemp(obs_shape=(7,))

        eval_envs = make_vec_envs_multi_task(env_name,
                                             seed,
                                             num_processes,
                                             None,
                                             eval_log_dir,
                                             device,
                                             False,
                                             envs_kwargs,
                                             num_frame_stack=None)

        obs = eval_envs.reset()
        obs = al_augment_obs(obs, prev_task, latent_dim, env_obs_shape, prior, prior)

        obs = obs_rms_eval.step(obs)

        epi_done = []

        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)

        t = 0

        use_prev_state = False
        while len(epi_done) < num_processes:
            with torch.no_grad():
                _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                    obs,
                    eval_recurrent_hidden_states,
                    eval_masks,
                    deterministic=False)

            # Observ reward and next obs
            if t == 0:
                mse_list_at_0.append(get_mse(prior, new_tasks, 2, num_processes).item())

            obs, reward, done, infos = eval_envs.step(action)
            posterior = get_posterior(vi, action, reward, prior, prev_task, num_processes,
                                      use_prev_state=use_prev_state)
            reward = get_reward(posterior, new_tasks, latent_dim, num_processes)
            obs = al_augment_obs(obs, prev_task, latent_dim, env_obs_shape, posterior, prior)
            obs = obs_rms_eval.step(obs)
            use_prev_state = True
            reward_iter[:, t] = reward.squeeze()
            t = t + 1

            if t == 10:
                mse_list_at_10.append(get_mse(posterior, new_tasks, 2, num_processes).item())
            if t == 50:
                mse_list_at_50.append(get_mse(posterior, new_tasks, 2, num_processes).item())

            eval_masks = torch.tensor(
                [[0.0] if done_ else [1.0] for done_ in done],
                dtype=torch.float32,
                device=device)

            for info in infos:
                if 'episode' in info.keys():
                    epi_done.append(True)

        mse_list_at_horizon.append(get_mse(posterior, new_tasks, 2, num_processes).item())
        r_list.append(reward_iter)
        eval_envs.close()

    mean_mse_0 = np.mean(mse_list_at_0)
    mean_mse_horizon = np.mean(mse_list_at_horizon)
    mean_mse_10 = np.mean(mse_list_at_10)
    mean_mse_50 = np.mean(mse_list_at_50)
    mean_r = np.mean(reduce(list.__add__, [torch.sum(elem, 1).tolist() for elem in r_list]))
    print("Evaluation using {} tasks. Mean reward: {}. Mean MSE: {:.2f} || {:.2f} || {:.2f} || {:.2f}".
          format(n_iter * num_processes, mean_r, mean_mse_0, mean_mse_10, mean_mse_50, mean_mse_horizon))
    return mean_r, r_list


class ObsSmootherTemp:
    def __init__(self, obs_shape, clipob=10., epsilon=1e-8):
        self.clipob = clipob
        self.epsilon = epsilon
        self.ob_rms = RunningMeanStd(shape=obs_shape)

    def step(self, obs):
        self.epsilon = self.epsilon
        # obs = obs.numpy()
        # self.ob_rms.update(obs)
        # obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        # return torch.tensor(obs, dtype=torch.float32)
        return obs
