import torch
import torch.nn.functional as F


def get_like(posterior, offset_star, latent_dim, num_processes):
    if type(posterior) == list:
        posterior = torch.tensor([posterior[i].flatten().tolist() for i in range(num_processes)])
        posterior[:, latent_dim:] = posterior[:, latent_dim:].log()

    likelihood = torch.sum(
        -(1 / (2 * posterior[:, latent_dim:].exp())) * (posterior[:, 0:latent_dim] - offset_star).pow(2),
        1).unsqueeze(1)
    return likelihood


def get_reward(p, t, latent_dim, num_processes):
    like = get_like(p, t, latent_dim, num_processes)
    return like


def get_mse(p, t, latent_dim, num_processes):
    if type(p) == list:
        p = torch.tensor([p[i].flatten().tolist() for i in range(num_processes)])
    return F.mse_loss(p[:, 0:latent_dim], t)
