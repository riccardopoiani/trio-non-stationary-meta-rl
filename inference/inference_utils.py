import torch
import torch.nn.functional as F


def loss_inference_closed_form(z, mu_hat, logvar_hat, mu_prior, logvar_prior,
                               n_samples, use_decay, decay_param, epoch, verbose):
    mse_direct = F.mse_loss(mu_hat, z)
    mse_var = torch.mean(torch.sum(logvar_hat.exp(), 1))
    mse = mse_direct + mse_var

    kld_1 = torch.sum(logvar_prior - logvar_hat, 1)
    kld_2 = (torch.sum(
        -1 + (mu_hat - mu_prior).pow(2) * (1 / logvar_prior.exp()) + (logvar_hat.exp() * (1 / logvar_prior.exp())), 1))

    kld = (1 / 2) * torch.mean(kld_1 + kld_2)
    if use_decay:
        kld = kld * (decay_param / n_samples)

    if verbose and epoch is not None and epoch % 100 == 0:
        print("Epoch {} MSE DIR {} MSE VAR {} KLD {} STEPS {}".format(epoch, mse_direct.item(), mse_var.item(),
                                                                      kld.item(), n_samples))
        print("\n")
    return mse + kld, kld.item(), mse.item()

