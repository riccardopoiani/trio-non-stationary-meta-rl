import torch

from task.task_generator import TaskGenerator


class AntGoalWithSignalTaskGenerator(TaskGenerator):

    def __init__(self, prior_var_min, prior_var_max):
        super(AntGoalWithSignalTaskGenerator, self).__init__()

        self.prior_std_min = prior_var_min ** (1 / 2)
        self.prior_std_max = prior_var_max ** (1 / 2)

        self.latent_dim = 4

        self.max = 1
        self.min = -1

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplemented()

    def sample_task_from_prior(self, prior):
        ok = True
        task_param = None

        while ok:
            mu = prior[0].clone().detach()
            var = prior[1].clone().detach()

            task_param = torch.normal(mu, var.sqrt())

            if torch.any(task_param > self.max) | torch.any(task_param < self.min):
                ok = True
            else:
                ok = False

        envs_kwargs = {'goal_x': task_param[0].item(),
                       'goal_y': task_param[1].item(),
                       'signal_x': task_param[2].item(),
                       'signal_y': task_param[3].item()}

        return envs_kwargs

    def sample_pair_tasks(self, num_processes):
        mu = (self.min - self.max) * torch.rand(num_processes, self.latent_dim) + self.max
        std = (self.prior_std_min - self.prior_std_max) * torch.rand(num_processes,
                                                                     self.latent_dim) + self.prior_std_max
        new_tasks = torch.normal(mu, std)
        not_ok_task = torch.any(new_tasks > self.max, 1) | torch.any(new_tasks < self.min, 1)

        while torch.sum(not_ok_task) != 0:
            temp_new_tasks = torch.normal(mu, std)

            new_tasks[not_ok_task, :] = temp_new_tasks[not_ok_task]
            not_ok_task = (
                    torch.any(new_tasks > self.max, 1) | torch.any(new_tasks < self.min, 1))

        prior = [torch.tensor([mu[i].tolist(), std[i].pow(2).tolist()]) for i in range(num_processes)]

        envs_kwargs = [{'goal_x': new_tasks[i][0].item(),
                        'goal_y': new_tasks[i][1].item(),
                        'signal_x': new_tasks[i][2].item(),
                        'signal_y': new_tasks[i][3].item()}
                       for i in range(num_processes)]

        return envs_kwargs, None, prior, new_tasks

