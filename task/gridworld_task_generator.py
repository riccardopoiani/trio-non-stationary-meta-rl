import torch

from task.task_generator import TaskGenerator


class GridWorldTaskGenerator(TaskGenerator):

    def __init__(self, size, goal_radius, charge_forward_min, charge_forward_max,
                 charge_rotation_min, charge_rotation_max, prior_goal_std_min,
                 prior_goal_std_max, prior_charge_std_min, prior_charge_std_max):
        super(GridWorldTaskGenerator, self).__init__()

        self.latent_dim = 5

        # Fixed env setting
        self.size = size
        self.goal_radius = goal_radius

        # Env latent space
        goal_max = size / 2
        goal_min = -size / 2

        self.latent_min_mean = torch.tensor([goal_min, goal_min, charge_forward_min,
                                             charge_rotation_min, charge_rotation_min], dtype=torch.float32)
        self.latent_max_mean = torch.tensor([goal_max, goal_max, charge_forward_max,
                                             charge_rotation_max, charge_rotation_max], dtype=torch.float32)
        self.latent_min_std = torch.tensor([prior_goal_std_min, prior_goal_std_min, prior_charge_std_min,
                                            prior_charge_std_min, prior_charge_std_max], dtype=torch.float32)
        self.latent_max_std = torch.tensor([prior_goal_std_max, prior_goal_std_max, prior_charge_std_max,
                                            prior_charge_std_max, prior_charge_std_max], dtype=torch.float32)

    def create_task_family(self, n_tasks, n_batches=1, test_perc=0, batch_size=160):
        raise NotImplemented

    def sample_task_from_prior(self, prior):
        ok = True
        while ok:
            mu = prior[0].clone().detach()
            var = prior[1].clone().detach()

            task_param = torch.normal(mu, var)

            if torch.sum(task_param > self.latent_max_mean) + torch.sum(task_param < self.latent_min_mean) == 0:
                ok = False

        envs_kwargs = {'size': self.size,
                       'goal_x': task_param[0].item(),
                       'goal_y': task_param[1].item(),
                       'goal_radius': self.goal_radius,
                       'charge_forward': task_param[2].item(),
                       'charge_left': task_param[3].item(),
                       'charge_right': task_param[4].item()}

        return envs_kwargs

    def sample_pair_tasks(self, num_p):
        mu = (self.latent_min_mean - self.latent_max_mean) * torch.rand(num_p, self.latent_dim) + self.latent_max_mean
        std = (self.latent_min_std - self.latent_max_std) * torch.rand(num_p, self.latent_dim) + self.latent_max_std
        new_tasks = torch.normal(mu, std).reshape(num_p, self.latent_dim)
        not_ok_task = torch.any(new_tasks > self.latent_max_mean, 1) | torch.any(new_tasks < self.latent_min_mean, 1)

        while torch.sum(not_ok_task) != 0:
            temp_new_tasks = torch.normal(mu, std).reshape(num_p, self.latent_dim)

            new_tasks[not_ok_task, :] = temp_new_tasks[not_ok_task]
            not_ok_task = (
                        torch.any(new_tasks > self.latent_max_mean, 1) | torch.any(new_tasks < self.latent_min_mean, 1))

        prior = [torch.tensor([mu[i].tolist(), std[i].tolist()]) for i in range(num_p)]

        envs_kwargs = [{'size': self.size,
                        'goal_x': new_tasks[i][0].item(),
                        'goal_y': new_tasks[i][1].item(),
                        'goal_radius': self.goal_radius,
                        'charge_forward': new_tasks[i][2].item(),
                        'charge_left': new_tasks[i][3].item(),
                        'charge_right': new_tasks[i][4].item()}
                       for i in range(num_p)]

        return envs_kwargs, None, prior, new_tasks