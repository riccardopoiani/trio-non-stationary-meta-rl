import torch

from task.task_generator import TaskGenerator


class MiniGolfTaskGenerator(TaskGenerator):

    def __init__(self, prior_std_min, prior_std_max):
        super(MiniGolfTaskGenerator, self).__init__()
        self.min_friction = -1
        self.max_friction = 1
        self.prior_std_min = prior_std_min
        self.prior_std_max = prior_std_max

        self.data_set = None
        self.param = None
        self.prior_dist = None
        self.n_tasks = None

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplementedError

    def sample_task_from_prior(self, prior):
        ok = True

        while ok:
            mu = prior[0].clone().detach()
            var = prior[1].clone().detach()

            task_param = torch.normal(mu, var.sqrt())

            if self.max_friction > task_param.item() > self.min_friction:
                ok = False

        envs_kwargs = {'friction': task_param.item()}

        return envs_kwargs

    def sample_pair_tasks(self, num_processes):
        ok = True
        while ok:
            mu = (self.min_friction - self.max_friction) * torch.rand(num_processes) + self.max_friction
            std = (self.prior_std_min - self.prior_std_max) * torch.rand(num_processes) + self.prior_std_max

            new_tasks = torch.normal(mu, std).reshape(num_processes, 1)

            if torch.sum(new_tasks > self.max_friction) + torch.sum(new_tasks < self.min_friction) == 0:
                ok = False

        prior = [torch.tensor([[mu[i]], [std[i].pow(2)]]) for i in range(num_processes)]

        envs_kwargs = [{'friction': new_tasks[i].item()} for i in range(num_processes)]

        return envs_kwargs, None, prior, new_tasks
