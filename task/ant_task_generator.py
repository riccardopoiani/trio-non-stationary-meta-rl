import torch
from task.task_generator import TaskGenerator


class AntTaskGenerator(TaskGenerator):

    def __init__(self, friction_var_min, friction_var_max, n_frictions=8):
        super(AntTaskGenerator, self).__init__()

        self.latent_min_mean = -1 * torch.ones(n_frictions, dtype=torch.float32)
        self.latent_max_mean = torch.ones(n_frictions, dtype=torch.float32)

        self.latent_min_std = (friction_var_min ** (1/2)) * torch.ones(n_frictions, dtype=torch.float32)
        self.latent_max_std = (friction_var_max ** (1/2)) * torch.ones(n_frictions, dtype=torch.float32)

        self.latent_dim = n_frictions

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplemented

    def sample_task_from_prior(self, prior):
        pass

    def sample_pair_tasks(self, num_processes):
        mu = (self.latent_min_mean - self.latent_max_mean) * torch.rand(num_processes, self.latent_dim) + self.latent_max_mean
        std = (self.latent_min_std - self.latent_max_std) * torch.rand(num_processes, self.latent_dim) + self.latent_max_std
        new_tasks = torch.normal(mu, std)
        not_ok_task = torch.any(new_tasks > self.latent_max_mean, 1) | torch.any(new_tasks < self.latent_min_mean, 1)

        while torch.sum(not_ok_task) != 0:
            temp_new_tasks = torch.normal(mu, std).reshape(num_processes, self.latent_dim)

            new_tasks[not_ok_task, :] = temp_new_tasks[not_ok_task]
            not_ok_task = (
                    torch.any(new_tasks > self.latent_max_mean, 1) | torch.any(new_tasks < self.latent_min_mean, 1))

        prior = [torch.tensor([mu[i].tolist(), std[i].pow(2).tolist()]) for i in range(num_processes)]

        envs_kwargs = [{'frictions': new_tasks[i].numpy()}
                       for i in range(num_processes)]

        return envs_kwargs, None, prior, new_tasks

