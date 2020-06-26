import torch
import torch.nn.functional as F

from learner.MetaLearner import RegressorMAML
from utilities.meta_utils import tensors_to_device


class WrapperLearner(RegressorMAML):

    def __init__(self, network, n_in, n_out, x_space, device="cpu", step_size=0.5, first_order=False,
                 optimizer=None, learn_step_size=False, per_param_step_size=False,
                 num_adaptation_step=1, scheduler=None, loss_function=F.mse_loss):

        super(WrapperLearner, self).__init__(network=network, device=device, step_size=step_size,
                                             first_order=first_order, optimizer=optimizer,
                                             learn_step_size=learn_step_size, per_param_step_size=per_param_step_size,
                                             num_adaptation_step=num_adaptation_step, scheduler=scheduler,
                                             loss_function=loss_function)
        self.n_in = n_in
        self.n_out = n_out
        self.x_space = x_space

    def reset_optimizer(self, optimizer):
        self.optimizer = optimizer

    def simulate_meta_training(self, num_batches=100, batch_size=32, test_split=0.1,
                               max_batches=100, first_task=True, verbose=True):
        self.network.set_meta(not first_task)

        # Task generation
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)
        for i in range(self.n_in):
            idx[i] = torch.randint(low=0, high=self.x_space[i].shape[0], size=(num_batches, batch_size))

        x_points = torch.empty(num_batches, batch_size, self.n_in)
        y_points = torch.empty(num_batches, batch_size, self.n_out, device=self.device)
        for b in range(num_batches):
            for elem in range(batch_size):
                for dim in range(self.n_in):
                    x_points[b, elem, dim] = self.x_space[dim, idx[dim, b, elem]]

        x_points = x_points.to(device=self.device)

        # Targets are the residual between perturbations and non-perturbations
        for b in range(num_batches):
            y_points[b] = self.network(x_points[b], perturbation=True).detach()

        # Prepare data in data_loader format (i.e. data_loader is a list of dict, each dict is a batch (key test train))
        data_loader = []
        for b in range(num_batches):
            new_dict = {}
            x_train = x_points[b][0:int(x_points[b].shape[0] * (1 - test_split))]
            x_test = x_points[b][int(x_points[b].shape[0] * (1 - test_split)):]

            y_train = y_points[b][0:int(y_points[b].shape[0] * (1 - test_split))]
            y_test = y_points[b][int(y_points[b].shape[0] * (1 - test_split)):]

            new_dict['train'] = [x_train, y_train]
            new_dict['test'] = [x_test, y_test]

            data_loader.append(new_dict)

        # Meta-train: meta-train the MAML residual parameters
        self.network.set_learn_net(False)
        self.network.set_learn_noise(False)
        self.network.set_learn_meta(True)
        self.network.set_meta(True)
        self.meta_train(data_loader=data_loader, max_batches=max_batches, verbose=verbose)

    def task_adaptation(self, data_loader, max_batches):
        """
        Once residuals has been trained, you receive the data, and you need to adapt fast to them...
        in particular the standard network will remain the same, while the residual one, will adapt to the task.
        Furthermore, at the end of this procedure, the two networks will be mixed together

        :param data_loader: observed data from the current task
        :param max_batches: number of batches for task adaptation
        :return: None
        """
        num_batches = 0
        optim = torch.optim.SGD(self.network.parameters(), lr=self.step_size, momentum=0.)

        self.network.set_learn_net(False)
        self.network.set_learn_noise(False)
        self.network.set_learn_meta(True)

        # Adaptation to the new task
        while num_batches < max_batches:
            for batch in data_loader:
                if num_batches >= max_batches:
                    break
                batch = tensors_to_device(batch['train'], device=self.device)
                for i in range(self.num_adaptation_steps):
                    inputs = batch[0]
                    target = batch[1]
                    optim.zero_grad()
                    pred = self.network(inputs)
                    loss = self.loss_function(pred, target)
                    loss.backward()
                    optim.step()
                num_batches += 1
