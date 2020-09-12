import torch

from task.task_generator import TaskGenerator


class ExploreTaskGenerator(TaskGenerator):

    def __init__(self, x_min, x_max, noise_std, std, mean_max, mean_min, amplitude):
        super(ExploreTaskGenerator, self).__init__()

        self.x_min = x_min
        self.x_max = x_max
        self.noise_std = noise_std
        self.std = std
        self.mean_max = mean_max
        self.mean_min = mean_min
        self.amplitude = amplitude

        self.data_set = None
        self.param = None
        self.prior_dist = None
        self.n_tasks = None

    def sample_task_from_prior(self, prior):
        mu = prior[0].clone().detach()
        var = prior[1].clone().detach()

        task_param = torch.normal(mu, var)

        envs_kwargs = {'min_x': self.x_min,
                       'max_x': self.x_max,
                       'noise_std': self.noise_std,
                       'std': self.std,
                       'mean': task_param.item(),
                       'amplitude': self.amplitude
                       }

        return envs_kwargs

    def sample_pair_tasks(self, num_processes):
        task_idx = torch.randint(low=0, high=self.n_tasks, size=(num_processes,))
        new_tasks = torch.tensor([self.param[i] for i in task_idx]).reshape(num_processes, 1)

        prior_idx = torch.randint(low=0, high=self.n_tasks, size=(num_processes,))
        prior = [self.prior_dist[prior_idx[i]].clone().detach() for i in range(num_processes)]

        envs_kwargs = [{'min_x': self.x_min,
                        'max_x': self.x_max,
                        'noise_std': self.noise_std,
                        'std': self.std,
                        'mean': new_tasks[i].item(),
                        'amplitude': self.amplitude
                        } for i in range(num_processes)]

        return envs_kwargs, None, prior, new_tasks

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

    def _create_family_item(self, n_batches, test_perc=0, batch_size=128):
        m = (1 - self.x_max) * torch.rand(1) + self.x_max

        return None, torch.tensor([m], dtype=torch.float32)

    def _sample_prior(self):
        mu_l = []
        std_l = []

        mu = torch.tensor([(self.mean_max + self.mean_min) / 2], dtype=torch.float32)
        std = torch.tensor([self.mean_max - self.mean_min], dtype=torch.float32)

        mu_l.append(mu)
        std_l.append(std)

        return mu_l, std_l
