import torch
from task.task_generator import TaskGenerator


class AntTaskGeneratorEasy(TaskGenerator):

    def __init__(self, friction_var_min, friction_var_max, n_frictions=8):
        super(AntTaskGeneratorEasy, self).__init__()

        self.max_mean_ok = 1
        self.min_mean_ok = 0.6

        self.max_std = friction_var_max ** (1/2)
        self.min_std = friction_var_min ** (1/2)

        self.latent_dim = n_frictions

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplemented

    def sample_task_from_prior(self, prior):
        ok = True

        while ok:
            mu = prior[0].clone().detach()
            var = prior[1].clone().detach()

            task_param = torch.normal(mu, var.sqrt())

            if torch.any(task_param > 1) | torch.any(task_param < -1):
                ok = True
            else:
                ok = False

        envs_kwargs = {'frictions': task_param.numpy()}

        return envs_kwargs

    def sample_pair_tasks(self, num_processes):
        # Sample tasks
        nt, mu, std = self.sample_ok_tasks(num_processes)

        # Merge tasks
        prior = [torch.tensor([mu[i].tolist(), std[i].pow(2).tolist()]) for i in range(num_processes)]

        envs_kwargs = [{'frictions': nt[i].numpy()}
                       for i in range(num_processes)]

        return envs_kwargs, None, prior, nt

    def sample_ok_tasks(self, num_p):
        mu = (self.min_mean_ok - self.max_mean_ok) * torch.rand(num_p, self.latent_dim) + self.max_mean_ok
        std = (self.min_std - self.max_std) * torch.rand(num_p, self.latent_dim) + self.max_std

        new_t = self._sample(mu=mu,
                             std=std,
                             max_m=torch.ones(self.latent_dim, dtype=torch.float32) * self.max_mean_ok,
                             min_m=torch.ones(self.latent_dim, dtype=torch.float32) * self.min_mean_ok)

        return new_t, mu, std

    def _sample(self, mu, std, max_m, min_m):
        new_tasks = torch.normal(mu, std)
        not_ok_task = torch.any(new_tasks > max_m, 1) | torch.any(new_tasks < min_m, 1)

        while torch.sum(not_ok_task) != 0:
            temp_new_tasks = torch.normal(mu, std)

            new_tasks[not_ok_task, :] = temp_new_tasks[not_ok_task]
            not_ok_task = (
                    torch.any(new_tasks > max_m, 1) | torch.any(new_tasks < min_m, 1))
        return new_tasks
