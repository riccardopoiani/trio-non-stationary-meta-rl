import torch
import numpy as np

from task.TaskGenerator import TaskGenerator


class GaussianTaskGenerator(TaskGenerator):

    def __init__(self, x_min=-100, x_max=100, min_mean=-40, max_mean=40, min_std=15, max_std=35,
                 prior_mu_min=-10, prior_mu_max=10, prior_std_min=0.1, prior_std_max=5):
        super(GaussianTaskGenerator, self).__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.x_space = torch.arange(self.x_min, self.x_max, 0.01).unsqueeze(0)
        self.n_in = 1
        self.n_out = 1

        self.min_mean = min_mean
        self.max_mean = max_mean
        self.min_std = min_std
        self.max_std = max_std
        self.prior_mu_min = prior_mu_min
        self.prior_mu_max = prior_mu_max
        self.prior_std_min = prior_std_min
        self.prior_std_max = prior_std_max

        self.z_dim = 2

        self.data_set = None
        self.param = None
        self.prior_dist = None
        self.n_tasks = None

    def _sample_task_family(self, n_batches, test_perc=0, batch_size=128):
        a = 1
        m = (self.min_mean - self.max_mean) * torch.rand(1) + self.max_mean
        s = (self.min_std - self.max_std) * torch.rand(1) + self.max_std

        data = self.get_mixed_data_loader(amplitude=a,
                                          mean=m.item(),
                                          std=s.item(),
                                          num_batches=n_batches,
                                          test_perc=test_perc,
                                          batch_size=batch_size)
        return data, m, s

    def _sample_prior_dist_family(self, dim):
        mu_l = []
        std_l = []
        for i in range(dim):
            mu = (self.prior_mu_min - self.prior_mu_max) * torch.rand(1) + self.prior_mu_max
            var = (self.prior_std_min - self.prior_std_max) * torch.rand(1) + self.prior_std_max

            mu_l.append(mu)
            std_l.append(var)

        return mu_l, std_l

    def sample_single_task(self, num_processes):
        curr_task_idx = np.random.randint(low=0, high=self.n_tasks, size=(num_processes,))
        curr_latent = [self.param[i] for i in curr_task_idx]

        envs_kwargs = [{'mean': self.param[curr_task_idx[i]][0],
                        'std': self.param[curr_task_idx[i]][1],
                        'scale_reward': False} for i in range(num_processes)]

        return envs_kwargs, curr_latent

    def sample_pair_tasks(self, num_processes):
        # Choose pair of init task and distribution for the next task
        task_idx = torch.randint(low=0, high=self.n_tasks, size=(num_processes,))
        new_tasks = torch.tensor([self.param[i] for i in task_idx])

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
        mu = [mu[i] + torch.tensor(prev_task[i]) for i in range(num_processes)]

        # Sample new task
        envs_kwargs = [{'amplitude': 1,
                        'mean': new_tasks[i][0].item(),
                        'std': new_tasks[i][1].item(),
                        'noise_std': 0.001,
                        'scale_reward': False} for i in range(num_processes)]

        return envs_kwargs, prev_task, prior, new_tasks

    def get_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        data_set = []
        mean_set = []
        std_set = []
        param = []

        for _ in range(n_tasks):
            data, mean, std = self._sample_task_family(n_batches=n_batches, test_perc=test_perc, batch_size=batch_size)
            data_set.append(data)
            mean_set.append(mean)
            std_set.append(std)
            param.append((mean.item(), std.item()))

        prior_dist = []
        for _ in range(n_tasks):
            prior_dist.append(torch.Tensor(self._sample_prior_dist_family(self.z_dim)))

        self.data_set = self.data_set
        self.param = param
        self.prior_dist = prior_dist
        self.n_tasks = n_tasks

        return data_set, param, prior_dist

    def get_mixed_data_loader(self, amplitude=1, mean=0, std=1, noise_mean=0, noise_std=0.1,
                              num_batches=100, test_perc=1, batch_size=128, info_split=0.5):
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)
        f_space = torch.exp(-((self.x_space - mean) ** 2) / (std ** 2))
        w = f_space / f_space.sum()

        for i in range(self.n_in):
            temp = torch.multinomial(w, num_batches * batch_size, replacement=True)
            idx[i] = temp.reshape(num_batches, batch_size)

        num_random_elem_per_batch = int(batch_size * info_split)
        for i in range(self.n_in):
            for b in range(num_batches):
                new_idx = torch.randint(low=0, high=self.x_space[i].shape[0], size=(num_random_elem_per_batch,))
                where = torch.randint(low=0, high=batch_size, size=(num_random_elem_per_batch,))
                idx[i][b][where] = new_idx

        x_points = torch.empty(num_batches, batch_size, self.n_in)
        y_points = torch.empty(num_batches, batch_size, self.n_out)
        for b in range(num_batches):
            for elem in range(batch_size):
                for dim in range(self.n_in):
                    x_points[b, elem, dim] = self.x_space[dim, idx[dim, b, elem]]

        for b in range(num_batches):
            y_clean = amplitude * torch.exp(-((x_points[b] - mean) ** 2) / (std ** 2))
            noise = torch.from_numpy(np.random.normal(loc=noise_mean, scale=noise_std, size=y_clean.shape))
            y_points[b] = y_clean + noise

        # Prepare data in data_loader format (i.e. data_loader is a list of dict, each dict is a batch (key test train))
        data_loader = []
        for b in range(num_batches):
            new_dict = {}
            x_train = x_points[b][0:int(x_points[b].shape[0] * (1 - test_perc))]
            x_test = x_points[b][int(x_points[b].shape[0] * (1 - test_perc)):]

            y_train = y_points[b][0:int(y_points[b].shape[0] * (1 - test_perc))]
            y_test = y_points[b][int(y_points[b].shape[0] * (1 - test_perc)):]

            new_dict['train'] = [x_train, y_train]
            new_dict['test'] = [x_test, y_test]

            data_loader.append(new_dict)

        return data_loader

    def get_informative_data_loader(self, amplitude=1, mean=0, std=1, noise_mean=0, noise_std=0.1,
                                    num_batches=100, test_perc=1, batch_size=32):
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)
        f_space = torch.exp(-((self.x_space - mean) ** 2) / (std ** 2))
        w = f_space / f_space.sum()

        for i in range(self.n_in):
            temp = torch.multinomial(w, num_batches * batch_size, replacement=True)
            idx[i] = temp.reshape(num_batches, batch_size)

        x_points = torch.empty(num_batches, batch_size, self.n_in)
        y_points = torch.empty(num_batches, batch_size, self.n_out)
        for b in range(num_batches):
            for elem in range(batch_size):
                for dim in range(self.n_in):
                    x_points[b, elem, dim] = self.x_space[dim, idx[dim, b, elem]]

        for b in range(num_batches):
            y_clean = amplitude * torch.exp(-((x_points[b] - mean) ** 2) / (std ** 2))
            noise = torch.from_numpy(np.random.normal(loc=noise_mean, scale=noise_std, size=y_clean.shape))
            y_points[b] = y_clean + noise

        # Prepare data in data_loader format (i.e. data_loader is a list of dict, each dict is a batch (key test train))
        data_loader = []
        for b in range(num_batches):
            new_dict = {}
            x_train = x_points[b][0:int(x_points[b].shape[0] * (1 - test_perc))]
            x_test = x_points[b][int(x_points[b].shape[0] * (1 - test_perc)):]

            y_train = y_points[b][0:int(y_points[b].shape[0] * (1 - test_perc))]
            y_test = y_points[b][int(y_points[b].shape[0] * (1 - test_perc)):]

            new_dict['train'] = [x_train, y_train]
            new_dict['test'] = [x_test, y_test]

            data_loader.append(new_dict)

        return data_loader

    def get_data_loader(self, amplitude=1, mean=0, std=1, noise_mean=0, noise_std=0.1,
                        num_batches=100, test_perc=1, batch_size=32):
        # Task generation
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)
        for i in range(self.n_in):
            idx[i] = torch.randint(low=0, high=self.x_space[i].shape[0], size=(num_batches, batch_size))

        x_points = torch.empty(num_batches, batch_size, self.n_in)
        y_points = torch.empty(num_batches, batch_size, self.n_out)
        for b in range(num_batches):
            for elem in range(batch_size):
                for dim in range(self.n_in):
                    x_points[b, elem, dim] = self.x_space[dim, idx[dim, b, elem]]

        for b in range(num_batches):
            y_clean = amplitude * torch.exp(-((x_points[b] - mean) ** 2) / (std ** 2))
            noise = torch.from_numpy(np.random.normal(loc=noise_mean, scale=noise_std, size=y_clean.shape))
            y_points[b] = y_clean + noise

        # Prepare data in data_loader format (i.e. data_loader is a list of dict, each dict is a batch (key test train))
        data_loader = []
        for b in range(num_batches):
            new_dict = {}
            x_train = x_points[b][0:int(x_points[b].shape[0] * (1 - test_perc))]
            x_test = x_points[b][int(x_points[b].shape[0] * (1 - test_perc)):]

            y_train = y_points[b][0:int(y_points[b].shape[0] * (1 - test_perc))]
            y_test = y_points[b][int(y_points[b].shape[0] * (1 - test_perc)):]

            new_dict['train'] = [x_train, y_train]
            new_dict['test'] = [x_test, y_test]

            data_loader.append(new_dict)

        return data_loader
