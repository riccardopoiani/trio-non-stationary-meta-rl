import torch
from task.task_generator import TaskGenerator


class AntTaskGeneratorAdvanced(TaskGenerator):

    def __init__(self,
                 friction_var_min_rnd,
                 friction_var_max_rnd,
                 friction_var_max_ok,
                 friction_var_min_ok,
                 n_frictions=8):
        super(AntTaskGeneratorAdvanced, self).__init__()

        self.max_mean_ok = 1
        self.min_mean_ok = 0.6

        self.max_mean_rnd = 1
        self.min_mean_rnd = -1

        self.max_mean_broken = -0.6
        self.min_mean_broken = -1

        self.max_std_rnd = friction_var_max_rnd ** (1 / 2)
        self.min_std_rnd = friction_var_min_rnd ** (1 / 2)

        self.max_std_ok = friction_var_max_ok ** (1 / 2)
        self.min_std_ok = friction_var_min_ok ** (1 / 2)

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
        # Compute how much of each task we want to sample
        v = torch.multinomial(torch.tensor([0.6, 0.4]), num_processes, replacement=True)
        num_rnd = (v == 0).sum().item()
        num_b = (v == 1).sum().item()

        # Sample tasks
        nt_rnd, mu_rnd, std_rnd = self.sample_rnd_leg_tasks(num_rnd)
        nt_b, mu_b, std_b = self.sample_broke_leg_tasks(num_b)

        # Merge tasks
        nt = torch.cat([nt_rnd, nt_b], 0)
        mu = torch.cat([mu_rnd, mu_b], 0)
        std = torch.cat([std_rnd, std_b], 0)

        perm = torch.randperm(num_processes)

        nt = nt[perm]
        mu = mu[perm]
        std = std[perm]

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

    def sample_rnd_leg_tasks(self, num_p):
        rnd_leg_idx = torch.randint(low=0, high=4, size=(num_p,))
        rnd_leg_idx_2 = torch.randint(low=0, high=4, size=(num_p,))

        max_m = torch.ones(num_p, self.latent_dim) * self.max_mean_ok
        min_m = torch.ones(num_p, self.latent_dim) * self.min_mean_ok
        max_s = torch.ones(num_p, self.latent_dim) * self.max_std_ok
        min_s = torch.ones(num_p, self.latent_dim) * self.min_std_ok

        for p in range(num_p):
            # Rnd leg 1 - Mean
            max_m[p, rnd_leg_idx[p] * 2:  rnd_leg_idx[p] * 2 + 2] = self.max_mean_rnd
            min_m[p, rnd_leg_idx[p] * 2:  rnd_leg_idx[p] * 2 + 2] = self.min_mean_rnd

            # Rnd leg 1 - Std
            max_s[p, rnd_leg_idx[p] * 2:  rnd_leg_idx[p] * 2 + 2] = self.max_std_rnd
            min_s[p, rnd_leg_idx[p] * 2:  rnd_leg_idx[p] * 2 + 2] = self.min_std_rnd

            # Rnd leg 2 - Mean
            max_m[p, rnd_leg_idx_2[p] * 2:  rnd_leg_idx_2[p] * 2 + 2] = self.max_mean_rnd
            min_m[p, rnd_leg_idx_2[p] * 2:  rnd_leg_idx_2[p] * 2 + 2] = self.min_mean_rnd

            # Rnd leg 2 -Std
            max_s[p, rnd_leg_idx_2[p] * 2:  rnd_leg_idx_2[p] * 2 + 2] = self.max_std_rnd
            min_s[p, rnd_leg_idx_2[p] * 2:  rnd_leg_idx_2[p] * 2 + 2] = self.min_std_rnd

        mu = (min_m - max_m) * torch.rand(num_p, self.latent_dim) + max_m
        std = (min_s - max_s) * torch.rand(num_p, self.latent_dim) + max_s
        new_t = self._sample(mu=mu, std=std, max_m=max_m, min_m=min_m)

        for p in range(num_p):
            for leg in range(4):
                if leg != rnd_leg_idx[p] and leg != rnd_leg_idx_2[p]:
                    std[p, leg * 2: leg * 2 + 2] = (self.max_std_rnd - self.min_std_rnd) / (self.max_std_ok - self.min_std_ok) * (std[p, leg * 2: leg * 2 + 2] - self.max_std_ok) + self.max_std_rnd

        return new_t, mu, std

    def sample_broke_leg_tasks(self, num_p):
        broken_leg_idx = torch.randint(low=0, high=4, size=(num_p,))
        broken_leg_idx_2 = torch.randint(low=0, high=4, size=(num_p,))

        max_m = torch.ones(num_p, self.latent_dim) * self.max_mean_ok
        min_m = torch.ones(num_p, self.latent_dim) * self.min_mean_ok
        max_s = torch.ones(num_p, self.latent_dim) * self.max_std_ok
        min_s = torch.ones(num_p, self.latent_dim) * self.min_std_ok

        for p in range(num_p):
            # Rnd leg 1 - Mean
            max_m[p, broken_leg_idx[p] * 2:  broken_leg_idx[p] * 2 + 2] = self.max_mean_broken
            min_m[p, broken_leg_idx[p] * 2:  broken_leg_idx[p] * 2 + 2] = self.min_mean_broken

            # Rnd leg 2 - Mean
            max_m[p, broken_leg_idx_2[p] * 2:  broken_leg_idx_2[p] * 2 + 2] = self.max_mean_broken
            min_m[p, broken_leg_idx_2[p] * 2:  broken_leg_idx_2[p] * 2 + 2] = self.min_mean_broken

        mu = (min_m - max_m) * torch.rand(num_p, self.latent_dim) + max_m
        std = (min_s - max_s) * torch.rand(num_p, self.latent_dim) + max_s
        new_t = self._sample(mu=mu, std=std, max_m=max_m, min_m=min_m)

        std = (self.max_std_rnd - self.min_std_rnd) / (self.max_std_ok - self.min_std_ok) * (std - self.max_std_ok) + self.max_std_rnd

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
