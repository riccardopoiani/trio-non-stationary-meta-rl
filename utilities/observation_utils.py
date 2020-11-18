import torch


def augment_obs_oracle(obs, tasks, use_env_obs):
    if use_env_obs:
        return torch.cat([obs, tasks], 1)
    return tasks


def augment_obs_optimal(obs, latent_dim, posterior, use_env_obs, is_prior):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    elif not is_prior:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

    posterior_sample = torch.normal(posterior[:, 0:latent_dim], posterior[:, latent_dim:].sqrt())

    if use_env_obs:
        new_obs = torch.cat([obs, posterior_sample], 1)
        return new_obs

    return posterior_sample


def augment_obs_posterior(obs, latent_dim, posterior, use_env_obs, is_prior):
    num_proc = obs.shape[0]

    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_proc)])
    elif not is_prior:
        posterior[:, latent_dim:] = posterior[:, latent_dim:].exp()

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


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    # PyTorch version.
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).float()
        self.var = torch.ones(shape).float()
        self.count = epsilon

    def update(self, x):
        x = x.view((-1, x.shape[-1]))
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + torch.pow(delta, 2) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count
