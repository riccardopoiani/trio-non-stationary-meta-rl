import torch
import torch.nn.functional as F


def loss_inference_closed_form(z, mu_hat, logvar_hat, mu_prior, logvar_prior, epoch, verbose):
    mse = torch.mean(torch.sum(logvar_hat.exp(), 1)) + F.mse_loss(mu_hat, z)
    # mse = torch.mean(torch.sum(logvar_hat.exp(), 1)) + F.l1_loss(mu_hat, z)
    # mse = F.l1_loss(mu_hat, z)

    if epoch == -1:
        print("Mu hat {}".format(mu_hat))
        print("Logvar hat {}".format(logvar_hat))
        print("True task {}".format(z))
        print(mse)
        print(F.mse_loss(mu_hat, z))
        print(torch.mean(torch.sum(logvar_hat.exp(), 1)))

    kdl_1 = (torch.log(torch.prod(logvar_prior.exp(), 1) / torch.prod(logvar_hat.exp(), 1)))
    kld_2 = (torch.sum(
        -1 + (mu_hat - mu_prior).pow(2) * (1 / logvar_prior.exp()) + (logvar_hat.exp() * (1 / logvar_prior.exp())), 1))

    kld = (1 / 2) * torch.mean(kdl_1 + kld_2)

    # if verbose and epoch % 100 == 0:
    #    print("Epoch {} L1 loss {}".format(epoch, mse.item()))
    if verbose and epoch is not None and epoch % 100 == 0:
        print("Epoch {} MSE {} KLD {}".format(epoch, mse.item(), kld.item()))
    return mse + kld, kld.item(), mse.item()
    # return mse, mse.item(), mse.item()


# Define the training procedure
def train_inference_network_family(net, optimizer, epoch, n_tasks, data_set, param, prior_dist,
                                   verbose, max_seq_len, min_seq_len, batch_per_task=1, n_batch=32):
    train_loss = 0
    mse_train_loss = 0
    kdl_train_loss = 0

    # Choose pair of init task and distribution for the next task
    task_idx = torch.randint(low=0, high=n_tasks, size=(n_batch,))
    task_loader = [data_set[i] for i in task_idx]
    target = torch.tensor([param[i] for i in task_idx])

    prev_task_param = torch.randint(low=0, high=n_tasks, size=(n_batch,))

    prior = torch.empty(n_batch, 4)
    mu_prior = torch.empty(n_batch, 2)
    logvar_prior = torch.empty(n_batch, 2)
    for t_idx in range(n_batch):
        prior[t_idx] = prior_dist[prev_task_param[t_idx]].reshape(1, 4).squeeze(0).clone().detach()
        mu_prior[t_idx] = prior_dist[prev_task_param[t_idx]][0].clone().detach()
        logvar_prior[t_idx] = prior_dist[prev_task_param[t_idx]][1].clone().detach().log()

    for k in range(batch_per_task):
        num_data_context = torch.randint(low=min_seq_len, high=max_seq_len, size=(1,)).item()
        idx = torch.randperm(max_seq_len)
        ctx_idx = idx[0:num_data_context]

        context = torch.empty(n_batch, num_data_context, 2)
        prev_task = torch.empty(n_batch, 2)

        # Retrieving data to be fed to the inference
        i = 0
        for t_idx, task in enumerate(task_loader):
            # Creating new task
            mu = prior_dist[prev_task_param[t_idx]][0].clone().detach()
            var = prior_dist[prev_task_param[t_idx]][1].clone().detach()

            offset_param = torch.normal(mu, var)
            prev_task[i] = target[i] - offset_param
            prior[i][0:2] = prev_task[i] + prior[i][0:2].clone().detach()
            mu_prior[i] = prev_task[i] + mu_prior[i].clone().detach()

            # Creating context to be fed to the inference
            batch = task[k]['train']
            batch = torch.cat([batch[0], batch[1]], dim=1)
            context[i] = batch[ctx_idx]
            i += 1

        optimizer.zero_grad()
        z_hat, mu_hat, logvar_hat = net(context, prev_task, prior)

        # Compute reconstruction
        loss, mse, kdl = loss_inference_closed_form(target, mu_hat, logvar_hat, mu_prior, logvar_prior, epoch, verbose)
        loss.backward()

        train_loss += loss.item()
        mse_train_loss += mse
        kdl_train_loss += kdl
        optimizer.step()

    return train_loss / batch_per_task, mse_train_loss / batch_per_task, kdl_train_loss / batch_per_task
