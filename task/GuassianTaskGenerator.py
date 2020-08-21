import torch
import numpy as np

from task.TaskGenerator import TaskGenerator


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
        var = (self.prior_std_min - self.prior_std_max) * torch.rand(1) + self.prior_std_max

        mu_l.append(mu)
        std_l.append(var)

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

    def sample_pair_tasks_data_loader(self, num_processes):
        # Choose pair of init task and distribution for the next task
        task_idx = torch.randint(low=0, high=self.n_tasks, size=(num_processes,))
        new_tasks = torch.tensor([self.param[i] for i in task_idx]).reshape(num_processes, 1)
        data = [self.data_set[i] for i in task_idx]

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

        return data, prev_task, prior, new_tasks

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

    def get_mixed_data_loader(self, amplitude=1, mean=0, std=1, noise_mean=0, noise_std=0.001,
                              num_batches=100, test_perc=1, batch_size=128,
                              info_split=0.5, high_split=0.05):
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)

        if np.random.binomial(n=1, p=0.5):
            for i in range(self.n_in):
                for b in range(num_batches):
                    idx[i][b] = torch.randint(low=0, high=self.x_space[i].shape[0], size=(batch_size,))
        else:
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
            num_high = int(batch_size * high_split)
            for i in range(self.n_in):
                for b in range(num_batches):
                    new_idx = torch.where(f_space > 0.97)[1]
                    select = torch.randint(low=0, high=new_idx.shape[0], size=(num_high,))

                    where = torch.randint(low=0, high=batch_size, size=(num_high,))
                    idx[i][b][where] = new_idx[select]

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
