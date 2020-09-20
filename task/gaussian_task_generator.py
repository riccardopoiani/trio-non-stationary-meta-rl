import torch
import numpy as np

from task.task_generator import TaskGenerator


class GaussianTaskGenerator(TaskGenerator):

    def __init__(self, x_min, x_max, min_mean, max_mean,
                 prior_mu_min, prior_mu_max, prior_std_min, prior_std_max, std, amplitude):
        super(GaussianTaskGenerator, self).__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.x_space = torch.arange(self.x_min, self.x_max, 0.01).unsqueeze(0)
        self.n_in = 1
        self.n_out = 1

        self.min_mean = min_mean
        self.max_mean = max_mean

        self.prior_mu_min = prior_mu_min
        self.prior_mu_max = prior_mu_max
        self.prior_std_min = prior_std_min
        self.prior_std_max = prior_std_max

        self.std = std
        self.amplitude = amplitude

        self.z_dim = 1

        self.data_set = None
        self.param = None
        self.prior_dist = None
        self.n_tasks = None

    def _create_family_item(self, n_batches, test_perc=0, batch_size=128):
        m = (self.min_mean - self.max_mean) * torch.rand(1) + self.max_mean

        data = self.get_mixed_data_loader(amplitude=self.amplitude,
                                          mean=m.item(),
                                          std=self.std,
                                          num_batches=n_batches,
                                          test_perc=test_perc,
                                          batch_size=batch_size,
                                          noise_std=0.001)
        return data, m

    def _sample_prior(self):
        mu_l = []
        std_l = []

        mu = (self.prior_mu_min - self.prior_mu_max) * torch.rand(1) + self.prior_mu_max
        std = (self.prior_std_min - self.prior_std_max) * torch.rand(1) + self.prior_std_max

        mu_l.append(mu)
        std_l.append(std)

        return mu_l, std_l

    def sample_task(self, num_processes):
        curr_task_idx = np.random.randint(low=0, high=self.n_tasks, size=(num_processes,))
        curr_latent = [self.param[i] for i in curr_task_idx]

        envs_kwargs = [{'amplitude': self.amplitude,
                        'mean': self.param[curr_task_idx[i]],
                        'std': self.std,
                        'noise_std': 0.001,
                        'min_x': self.x_min,
                        'max_x': self.x_max,
                        'scale_reward': False} for i in range(num_processes)]

        return envs_kwargs, curr_latent

    def sample_task_from_prior(self, prior):
        mu = prior[0].clone().detach()
        var = prior[1].clone().detach()

        task_param = torch.normal(mu, var)

        envs_kwargs = {'amplitude': self.amplitude,
                       'mean': task_param.item(),
                       'std': self.std,
                       'noise_std': 0.001,
                       'min_x': self.x_min,
                       'max_x': self.x_max,
                       'scale_reward': False}

        return envs_kwargs

    def sample_pair_tasks(self, num_processes):
        # Choose pair of init task and distribution for the next task
        task_idx = torch.randint(low=0, high=self.n_tasks, size=(num_processes,))
        new_tasks = torch.tensor([self.param[i] for i in task_idx]).reshape(num_processes, 1)

        prev_task_param = torch.randint(low=0, high=self.n_tasks, size=(num_processes,))

        prior = [self.prior_dist[prev_task_param[i]].clone().detach() for i in range(num_processes)]

        mu = [prior[i][0].clone().detach() for i in range(num_processes)]
        var = [prior[i][1].clone().detach() for i in range(num_processes)]

        offset_param = [torch.normal(mu[i], var[i]).tolist() for i in range(num_processes)]
        offset_param = torch.tensor(offset_param)

        prev_task = new_tasks - offset_param

        for i in range(num_processes):
            prior[i][0, :] = prev_task[i] + prior[i][0, :].clone().detach()

        prev_task = prev_task.tolist()

        # Sample new task
        envs_kwargs = [{'amplitude': self.amplitude,
                        'mean': new_tasks[i].item(),
                        'std': self.std,
                        'noise_std': 0.001,
                        'min_x': self.x_min,
                        'max_x': self.x_max,
                        'scale_reward': False} for i in range(num_processes)]

        return envs_kwargs, prev_task, prior, new_tasks

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        data_set = []
        param = []

        for _ in range(n_tasks):
            data, mean = self._create_family_item(n_batches=n_batches, test_perc=test_perc, batch_size=batch_size)
            data_set.append(data)
            param.append((mean.item()))

        prior_dist = []
        for _ in range(n_tasks):
            prior_dist.append(torch.Tensor(self._sample_prior()))

        self.data_set = data_set
        self.param = param
        self.prior_dist = prior_dist
        self.n_tasks = n_tasks

        return data_set, param, prior_dist
