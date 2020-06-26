import copy
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tqdm import tqdm

from learner.PerturbatedMAML import PerturbatedMAML
from network.meta_network import NoisyMetaNetwork


class LearnPerturbationMAML(PerturbatedMAML):

    def __init__(self, network: NoisyMetaNetwork, x_space,
                 n_in, n_out, device="cpu", step_size=0.5, first_order=False,
                 optimizer=None, learn_step_size=False, per_param_step_size=False,
                 num_adaptation_step=1, scheduler=None, loss_function=F.mse_loss):
        super(LearnPerturbationMAML, self).__init__(network=network,
                                                    device=device,
                                                    step_size=step_size,
                                                    first_order=first_order,
                                                    optimizer=optimizer,
                                                    learn_step_size=learn_step_size,
                                                    per_param_step_size=per_param_step_size,
                                                    num_adaptation_step=num_adaptation_step,
                                                    scheduler=scheduler,
                                                    loss_function=loss_function,
                                                    n_in=n_in,
                                                    n_out=n_out,
                                                    x_space=x_space)
        self.previous_network: NoisyMetaNetwork = copy.deepcopy(network)
        self.previous_network.to(device=self.device)

    def simulate_meta_training(self, num_batches=100, batch_size=32, test_split=0.1,
                               max_batches=100, verbose=True):
        super().simulate_meta_training(num_batches=num_batches, batch_size=batch_size, test_split=test_split,
                                       max_batches=max_batches, verbose=verbose)
        self.previous_network = copy.deepcopy(self.network)

    def learn_perturbation_model(self, curr_data_loader, stochastic_iteration=10, lr=0.1,
                                 verbose=True, **kwargs):
        """
        Here the goal is to adapt the weights and standard deviation of the network in order to achieve
        a better perturbation for the next task


        With the current data loader and the perturbation network, we need to adjust the perturbation model
        Moreover, here we assume that we already trained the model on the previous dataset, so that network
        is one step ahead w.r.t. previous network

        :param curr_data_loader: data of the current task
        :param stochastic_iteration: number of iterations to update mean/std vectors of perturbation model
        :param lr: learning rate for the distributions weights
        :param verbose: whether to have visual progress or not
        :return: None
        """
        noise_optimizer = torch.optim.Adam(self.previous_network.parameters(), lr=lr)
        self.previous_network.set_learn_net(state=False)
        self.previous_network.set_learn_noise(state=True)

        n_iter = len(curr_data_loader)
        with tqdm(total=n_iter, disable=not verbose, **kwargs) as pbar:
            for batch in curr_data_loader:
                input_data = [batch['train'][0], batch['test'][0]]
                target_data = [batch['train'][1], batch['test'][1]]

                for i in range(stochastic_iteration):
                    for x, y in zip(input_data, target_data):
                        self.optimizer.zero_grad()
                        y_pred = self.previous_network(x, perturbation=True)
                        loss = self.loss_function(y_pred, y)
                        loss.backward()
                        noise_optimizer.step()
                pbar.update(1)

        # Set to the current network the new noise parameters
        params = OrderedDict(self.previous_network.named_meta_parameters())
        self.network.set_noise_layer(new_params=params)
