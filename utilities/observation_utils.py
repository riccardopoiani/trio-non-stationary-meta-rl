import torch


def augment_obs_time(obs, time, rescale_time, max_time=None):
    num_proc = obs.shape[0]

    new_obs = torch.tensor((num_proc, obs.shape[1] + 1))

    new_obs[:, 0:-1] = obs

    if rescale_time:
        new_obs[:, -1] = time / max_time
    else:
        new_obs[:, -1] = time

    return new_obs


def augment_obs_oracle(obs, latent_dim, tasks, use_env_obs, rescale_obs=True, max_old=None, min_old=None):
    if rescale_obs:
        tasks = rescale_latent(obs.shape[0], tasks, latent_dim, max_old, min_old)
    if use_env_obs:
        return torch.cat([obs, tasks], 1)
    return tasks


def augment_obs_optimal(obs, latent_dim, posterior, use_env_obs, is_prior,
                        rescale_obs=True, max_old=None, min_old=None):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    elif not is_prior:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    posterior_sample = torch.normal(posterior[:, 0:latent_dim], posterior[:, latent_dim:].sqrt())

    if rescale_obs:
        posterior_sample = rescale_latent(num_proc, posterior_sample, latent_dim, max_old=max_old, min_old=min_old)

    if use_env_obs:
        new_obs = torch.cat([obs, posterior_sample], 1)
        return new_obs

    return posterior_sample


def augment_obs_posterior(obs, latent_dim, posterior, use_env_obs, is_prior, rescale_obs=True, max_old=None,
                          min_old=None):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    elif not is_prior:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    if rescale_obs:
        posterior = rescale_posterior(num_proc, posterior, latent_dim, max_old=max_old, min_old=min_old)

    if use_env_obs:
        new_obs = torch.cat([obs, posterior], 1)
    else:
        new_obs = posterior.clone().detach()
    return new_obs


def get_posterior_no_prev(vi, action, reward, prior, max_action, min_action, env_obs, use_env_obs, use_prev_state=True):
    """
    Feed the variational model with the actual reward to identify the latent space
    and get the current reward using the posterior and the true task
    """
    num_proc = action.shape[0]
    flatten_prior = torch.tensor([prior[i].flatten().tolist() for i in range(num_proc)])

    # To feed VI, i need (n_batch, 1, 2)
    if max_action is not None or min_action is not None:
        t = (max_action - min_action) / (1 - (-1)) * (action - 1) + max_action
    else:
        t = action

    context = torch.empty(num_proc, 1, 1 + env_obs.shape[1] + action.shape[1]) if use_env_obs \
        else torch.empty(num_proc, 1, 1 + action.shape[1])

    if use_env_obs:
        context[:, 0, :] = torch.cat([t.float(), reward, env_obs], 1)
    else:
        context[:, 0, :] = torch.cat([t.float(), reward], 1)

    res = vi(context=context, prior=flatten_prior, use_prev_state=use_prev_state)
    res = res[1:]
    res = torch.cat([res[0].detach(), res[1].detach()], 1)
    return res


def rescale_posterior(num_proc, old_var, latent_dim, max_old, min_old, verbose=False):
    new_p = torch.empty(num_proc, latent_dim * 2)
    for j in range(latent_dim * 2):
        new_p[:, j] = (1 - (-1)) / (max_old[j] - min_old[j]) * (old_var[:, j] - max_old[j]) + 1
    return new_p


def rescale_latent(num_proc, old_var, latent_dim, max_old, min_old):
    new_l = torch.empty(num_proc, latent_dim)
    for j in range(latent_dim):
        new_l[:, j] = (1 - (-1)) / (max_old[j] - min_old[j]) * (old_var[:, j] - max_old[j]) + 1
    return new_l
