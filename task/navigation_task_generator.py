import torch

from task.task_generator import TaskGenerator


class NavigationTaskGenerator(TaskGenerator):

    def __init__(self, prior_goal_std_min,
                 prior_goal_std_max,
                 prior_signal_std_min,
                 prior_signal_std_max,
                 signals_dim):
        super(NavigationTaskGenerator, self).__init__()

        self.latent_dim = 1 + signals_dim * 2
        self.signals_dim = signals_dim

        # Env latent space
        self.latent_min_mean = -1 * torch.ones(self.latent_dim, dtype=torch.float32)
        self.latent_max_mean = 1 * torch.ones(self.latent_dim, dtype=torch.float32)

        self.latent_min_std = torch.tensor([prior_goal_std_min], dtype=torch.float32)
        self.latent_min_std = torch.cat(
            [self.latent_min_std, torch.tensor([prior_signal_std_min]).repeat(2 * signals_dim)])

        self.latent_max_std = torch.tensor([prior_goal_std_max], dtype=torch.float32)
        self.latent_max_std = torch.cat(
            [self.latent_max_std, torch.tensor([prior_signal_std_max]).repeat(2 * signals_dim)])

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplemented

    def sample_task_from_prior(self, prior):
        ok = True
        new_tasks = torch.empty(self.latent_dim)
        while ok:
            mu = prior[0].clone().detach()
            var = prior[1].clone().detach()

            new_tasks = torch.normal(mu, var.sqrt())

            if torch.sum(new_tasks > self.latent_max_mean) + torch.sum(new_tasks < self.latent_min_mean) == 0:
                ok = False

        envs_kwargs = {'goal_theta': new_tasks[0].item(),
                       'mean_x_vec': new_tasks[1:1 + self.signals_dim].numpy(),
                       'mean_y_vec': new_tasks[1 + self.signals_dim:].numpy(),
                       'signals_dim': self.signals_dim
                       }
        return envs_kwargs

    def sample_pair_tasks(self, num_p):
        mu = (self.latent_min_mean - self.latent_max_mean) * torch.rand(num_p, self.latent_dim) + self.latent_max_mean
        std = (self.latent_min_std - self.latent_max_std) * torch.rand(num_p, self.latent_dim) + self.latent_max_std
        new_tasks = torch.normal(mu, std)
        not_ok_task = torch.any(new_tasks > self.latent_max_mean, 1) | torch.any(new_tasks < self.latent_min_mean, 1)

        while torch.sum(not_ok_task) != 0:
            temp_new_tasks = torch.normal(mu, std).reshape(num_p, self.latent_dim)

            new_tasks[not_ok_task, :] = temp_new_tasks[not_ok_task]
            not_ok_task = (
                    torch.any(new_tasks > self.latent_max_mean, 1) | torch.any(new_tasks < self.latent_min_mean, 1))

        prior = [torch.tensor([mu[i].tolist(), std[i].pow(2).tolist()]) for i in range(num_p)]

        envs_kwargs = [{'goal_theta': new_tasks[i][0].item(),
                        'mean_x_vec': new_tasks[i][1:1 + self.signals_dim].numpy(),
                        'mean_y_vec': new_tasks[i][1 + self.signals_dim:].numpy(),
                        'signals_dim': self.signals_dim
                        }
                       for i in range(num_p)]

        return envs_kwargs, None, prior, new_tasks
