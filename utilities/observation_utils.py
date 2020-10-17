import torch


def augment_obs_oracle(obs, tasks, use_env_obs, rms=None):
    t = rms.step(tasks) if rms is not None else tasks

    if use_env_obs:
        return torch.cat([obs, t], 1)
    return tasks


def augment_obs_optimal(obs, latent_dim, posterior, use_env_obs, is_prior, rms=None):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    elif not is_prior:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    posterior_sample = torch.normal(posterior[:, 0:latent_dim], posterior[:, latent_dim:].sqrt())

    if rms is not None:
        posterior_sample = rms.step(posterior_sample)

    if use_env_obs:
        new_obs = torch.cat([obs, posterior_sample], 1)
        return new_obs

    return posterior_sample


def augment_obs_posterior(obs, latent_dim, posterior, use_env_obs, is_prior, rms=None):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    elif not is_prior:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    if rms is not None:
        posterior = rms.step(posterior)

    if use_env_obs:
        new_obs = torch.cat([obs, posterior], 1)
    else:
        new_obs = posterior.clone().detach()

    return new_obs


def get_posterior(vi, action, reward, prior, env_obs, use_env_obs, use_prev_state=True):
    num_proc = action.shape[0]
    flatten_prior = torch.tensor([prior[i].flatten().tolist() for i in range(num_proc)])

    # To feed VI, i need (n_batch, 1, 2)
    context = torch.empty(num_proc, 1, 1 + env_obs.shape[1] + action.shape[1]) if use_env_obs \
        else torch.empty(num_proc, 1, 1 + action.shape[1])

    if use_env_obs:
        context[:, 0, :] = torch.cat([action.float(), reward, env_obs], 1)
    else:
        context[:, 0, :] = torch.cat([action.float(), reward], 1)

    res = vi(context=context, prior=flatten_prior, use_prev_state=use_prev_state)
    res = res[1:]
    res = torch.cat([res[0].detach(), res[1].detach()], 1)
    return res
