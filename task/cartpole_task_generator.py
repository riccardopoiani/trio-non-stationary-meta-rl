import torch

from task.task_generator import TaskGenerator


class CartPoleTaskGenerator(TaskGenerator):

    def __init__(self, prior_var_min, prior_var_max):
        super(CartPoleTaskGenerator, self).__init__()

        self.prior_std_min = prior_var_min ** (1 / 2)
        self.prior_std_max = prior_var_max ** (1 / 2)

        self.latent_dim = 1

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

        # envs_kwargs = {'gravity': task_param[0].item(),
        #               'masscart': task_param[1].item(),
        #               'masspole': task_param[2].item(),
        #               'lenght': task_param[3].item(),
        #               'force_mag': task_param[4].item(),
        #               'goal_pos':task_param[5].item()}

        envs_kwargs = {'gravity': None,
                       'masscart': None,
                       'masspole': None,
                       'lenght': None,
                       'force_mag': None,
                       'goal_pos': task_param[0].item()}

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

        # envs_kwargs = [{'gravity': new_tasks[i][0].item(),
        #                'masscart': new_tasks[i][1].item(),
        #                'masspole': new_tasks[i][2].item(),
        #                'lenght': new_tasks[i][3].item(),
        #                'force_mag': new_tasks[i][4].item(),
        #                'goal_pos': new_tasks[i][5].item()}
        #               for i in range(num_processes)]
        envs_kwargs = [{'gravity': None,
                        'masscart': None,
                        'masspole': None,
                        'lenght': None,
                        'force_mag': None,
                        'goal_pos': new_tasks[i][0].item()}
                       for i in range(num_processes)]

        return envs_kwargs, None, prior, new_tasks
