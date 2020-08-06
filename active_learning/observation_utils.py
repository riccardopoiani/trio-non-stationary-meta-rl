import torch


def oracle_augment_obs(obs, latent, latent_dim, use_env_obs):
    """
    Augment observation with the latent space

    Input
    - obs: observation from the environment (shape: (num_proc, env_obs_shape))
    - latent: list of tuples of latent parameters (size list: num proc, size tuple: latent dim)
    - latent_dim: dimension of the latent space
    - use_env_obs: whether to use observation from the environment or not (False if the problem is MAB)

    Output:
    - tensor of size (num_processes, latent_dim+env_obs_shape) if use_env_obs True, else
    tensor of size (num_processes, latent_dim)
    """
    num_proc = obs.shape[0]

    if use_env_obs:
        new_obs = torch.empty((num_proc, latent_dim + obs.shape[1]))
        for i in range(num_proc):
            new_obs[i] = torch.cat([obs[i], torch.tensor(latent[i])])
    else:
        new_obs = torch.empty((num_proc, latent_dim))

        for i in range(num_proc):
            new_obs[i] = torch.tensor(latent[i])

    return new_obs


def augment_obs_posterior(obs, latent_dim, posterior, use_env_obs, rescale_obs=True, max_old=None, min_old=None):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    else:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    if rescale_obs:
        posterior = rescale_posterior(num_proc, posterior, latent_dim, max_old=max_old, min_old=min_old)

    if use_env_obs:
        new_obs = torch.empty((num_proc, 2 * latent_dim + obs.shape[1]))
        for i in range(num_proc):
            new_obs[i] = torch.cat([obs[i], torch.tensor(posterior[i])])
    else:
        new_obs = torch.empty((num_proc, 2 * latent_dim))
        for i in range(num_proc):
            new_obs[i] = posterior[i]

    return new_obs


def al_augment_obs(obs, latent_dim, posterior, prior, rescale_obs=True, max_old=None, min_old=None):
    num_proc = obs.shape[0]
    new_obs = torch.empty((num_proc, 4 * latent_dim))

    prior = torch.tensor([prior[i].flatten().tolist() for i in range(num_proc)])

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    else:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    if rescale_obs:
        posterior = rescale_posterior(num_proc, posterior, latent_dim, max_old=max_old, min_old=min_old)
        prior = rescale_posterior(num_proc, prior, latent_dim, max_old=max_old, min_old=min_old)

    for i in range(num_proc):
        new_obs[i] = torch.cat([prior[i], posterior[i]])

    return new_obs


def get_posterior_no_prev(vi, action, reward, prior, max_action, min_action, use_prev_state=True):
    """
    Feed the variational model with the actual reward to identify the latent space
    and get the current reward using the posterior and the true task
    """
    num_proc = action.shape[0]
    flatten_prior = torch.tensor([prior[i].flatten().tolist() for i in range(num_proc)])

    # To feed VI, i need (n_batch, 1, 2)
    context = torch.empty(num_proc, 1, 2)
    for i in range(num_proc):
        t = (max_action - (-min_action)) / (1 - (-1)) * (action[i] - 1) + max_action
        context[i] = torch.cat([t, reward[i]])

    res = vi(context=context, prior=flatten_prior, use_prev_state=use_prev_state)
    res = res[1:]
    res = torch.cat([res[0].detach(), res[1].detach()], 1)
    return res


def get_posterior(vi, action, reward, prior, prev_latent_space, max_action, min_action, use_prev_state=True):
    """
    Feed the variational model with the actual reward to identify the latent space
    and get the current reward using the posterior and the true task
    """
    num_proc = action.shape[0]
    prev_latent_space = torch.tensor(prev_latent_space)
    flatten_prior = torch.tensor([prior[i].flatten().tolist() for i in range(num_proc)])

    # To feed VI, i need (n_batch, 1, 2)
    context = torch.empty(num_proc, 1, 2)
    for i in range(num_proc):
        t = (max_action - (-min_action)) / (1 - (-1)) * (action[i] - 1) + max_action
        context[i] = torch.cat([t, reward[i]])

    res = vi(context=context, prev_z=prev_latent_space, prior=flatten_prior, use_prev_state=use_prev_state)
    res = res[1:]
    res = torch.cat([res[0].detach(), res[1].detach()], 1)
    return res


def rescale_posterior(num_proc, old_var, latent_dim, max_old, min_old, verbose=False):
    rescaled_posterior = []

    for i in range(num_proc):
        new = []
        for j in range(latent_dim * 2):
            t = (1 - (-1)) / (max_old[j] - min_old[j]) * (old_var[i][j] - max_old[j]) + 1
            if t > 1:
                if verbose:
                    print("Exceeding max in posterior dim {}, value {}, max {}".format(j, old_var[i][j], max_old[j]))
                t = 1.
            elif t < -1:
                if verbose:
                    print("Exceeding min in posterior dim {}, value {}, min {}".format(j, old_var[i][j], min_old[j]))
                t = -1
            new.append(t)
        rescaled_posterior.append(new)

    return torch.tensor(rescaled_posterior)
