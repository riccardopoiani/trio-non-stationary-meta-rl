import torch

from task.task_generator import TaskGenerator


class MiniGolfSignalsTaskGenerator(TaskGenerator):

    def __init__(self, num_signals, prior_var_min, prior_var_max):
        super(MiniGolfSignalsTaskGenerator, self).__init__()
        self.latent_dim = 1 + num_signals
        self.num_signals = num_signals
        self.min_value = -1 * torch.ones(self.latent_dim)
        self.max_value = 1 * torch.ones(self.latent_dim)
        self.prior_std_min = prior_var_min ** (1 / 2) * torch.ones(self.latent_dim)
        self.prior_std_max = prior_var_max ** (1 / 2) * torch.ones(self.latent_dim)

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplementedError

    def sample_task_from_prior(self, prior):
        ok = True

        # while ok:
        mu = prior[0].clone().detach()
        var = prior[1].clone().detach()

        task_param = torch.normal(mu, var.sqrt())

        # if self.max_value > task_param.item() > self.min_value:
        #    ok = False
        if self.num_signals == 0:
            envs_kwargs = {'friction': task_param.item(),
                           'signals': None}
        else:
            envs_kwargs = {'friction': task_param[0].item(),
                           'signals': task_param[1:].numpy()}

        return envs_kwargs

    def sample_pair_tasks(self, num_processes):
        mu = (self.min_value - self.max_value) * torch.rand(num_processes, self.latent_dim) + self.max_value
        std = (self.prior_std_min - self.prior_std_max) * torch.rand(num_processes,
                                                                     self.latent_dim) + self.prior_std_max
        new_tasks = torch.normal(mu, std)
        not_ok_task = torch.any(new_tasks > self.max_value, 1) | torch.any(new_tasks < self.min_value, 1)

        while torch.sum(not_ok_task) != 0:
            temp_new_tasks = torch.normal(mu, std).reshape(num_processes, self.latent_dim)

            new_tasks[not_ok_task, :] = temp_new_tasks[not_ok_task]
            not_ok_task = (
                    torch.any(new_tasks > self.max_value, 1) | torch.any(new_tasks < self.min_value, 1))

        prior = [torch.tensor([mu[i].tolist(), std[i].pow(2).tolist()]) for i in range(num_processes)]

        if self.num_signals > 0:
            envs_kwargs = [{'friction': new_tasks[i][0].item(),
                            'signals': new_tasks[i][1:].numpy()} for i in range(num_processes)]
        else:
            envs_kwargs = [{'friction': new_tasks[i].item(),
                            'signals': None} for i in range(num_processes)]

        return envs_kwargs, None, prior, new_tasks
