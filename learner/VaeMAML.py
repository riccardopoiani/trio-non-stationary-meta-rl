from collections import OrderedDict

import torch
import torch.nn.functional as F

from learner.MetaLearner import RegressorMAML
from utilities.meta_utils import tensors_to_device


class VaeMAML(RegressorMAML):

    def __init__(self, network, vae_model, x_space, n_hallucination=15,
                 n_batch_hallucination=5,
                 device="cpu", step_size=0.5, first_order=False,
                 optimizer=None, learn_step_size=False, per_param_step_size=False,
                 num_adaptation_step=1, scheduler=None, loss_function=F.mse_loss):
        super(VaeMAML, self).__init__(network,
                                      device=device,
                                      step_size=step_size,
                                      first_order=first_order,
                                      optimizer=optimizer,
                                      learn_step_size=learn_step_size,
                                      per_param_step_size=per_param_step_size,
                                      num_adaptation_step=num_adaptation_step,
                                      scheduler=scheduler,
                                      loss_function=loss_function)
        self.vae_model = vae_model
        self.x_space = x_space
        self.n_hallucination = n_hallucination
        self.n_batch_hallucination = n_batch_hallucination

    def meta_testing(self, data):
        """
        During meta testing:
        - Analyze the current data used with VAE in order to infer the latent parameters
        - Use the current network and the inferred parameter in order to generate data for the adaptation
        - Adapt the model
        """
        # Analyze the current data with VAE to infer the latent parameters
        z = self.vae_model(self.network, data)

        # Use the current network and the inferred parameters in order to generate data for adaptation
        hallucinated_dataset = []
        for b in range(self.n_batch_hallucination):
            new_idx = torch.randint(low=0, high=self.x_space[-1], size=(self.n_hallucination,))
            new_x = self.x_space[new_idx]
            new_t = z[0] * self.network(z[1] * new_x - z[2])
            hallucinated_dataset.append([new_x, new_t])

        # Adapt the model using the new dataset
        optim = torch.optim.SGD(self.network.parameters(), lr=self.step_size, momentum=0.)

        for batch in hallucinated_dataset:
            batch = tensors_to_device(batch, device=self.device)

            for i in range(self.num_adaptation_steps):
                inputs = batch[0]
                target = batch[1]
                optim.zero_grad()
                pred = self.network(inputs)
                loss = self.loss_function(pred, target)
                loss.backward()
                optim.step()



