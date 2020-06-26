import asyncio
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from learner.BasicLearner import BasicLearner
from utilities.meta_utils import tensors_to_device


class GradientBasedMetaLearner(BasicLearner):

    def __init__(self, network: torch.nn.Module, optimizer, loss_function, scheduler=None, device="cpu"):
        super(GradientBasedMetaLearner, self).__init__(network=network, device=device, loss_function=loss_function,
                                                       scheduler=scheduler, optimizer=optimizer)
        self._event_loop = asyncio.get_event_loop()

    def adapt(self, input, targets, *args, **kwargs):
        raise NotImplemented()

    def step(self, train_episodes, valid_episodes, *args, **kwargs):
        raise NotImplemented

    def _async_gather(self, coroutines):
        coroutine = asyncio.gather(*coroutines)
        return zip(*self._event_loop.run_until_complete(coroutine))


class RegressorMAML(GradientBasedMetaLearner):

    def __init__(self, network, device="cpu", step_size=0.5, first_order=False,
                 optimizer=None, learn_step_size=False, per_param_step_size=False,
                 num_adaptation_step=1, scheduler=None, loss_function=F.mse_loss):
        super(RegressorMAML, self).__init__(network=network, device=device, optimizer=optimizer,
                                            scheduler=scheduler, loss_function=loss_function)
        self.step_size = step_size
        self.first_order = first_order
        self.num_adaptation_steps = num_adaptation_step

        if per_param_step_size:
            # In this case, we are applying different steps size for each weight, and we create
            # a structure ad-hoc
            self.step_size = OrderedDict((name, torch.tensor(step_size,
                                                             dtype=param.dtype, device=self.device,
                                                             requires_grad=learn_step_size)) for (name, param)
                                         in network.named_meta_parameters())
        else:
            # In this case we are not applying a different gradient to different weights
            # we just create the step size as a tensor, send it to the device, and we require to learn
            # only if learn_step_size = True
            self.step_size = torch.tensor(step_size, dtype=torch.float32,
                                          device=self.device, requires_grad=learn_step_size)

        # In this case we want to to add to the optimizer the fact that he has to learn the step size as well
        if (self.optimizer is not None) and learn_step_size:
            self.optimizer.add_param_group(
                {'params': self.step_size.values() if per_param_step_size else [self.step_size]})
            if scheduler is not None:
                for group in self.optimizer.param_groups:
                    group.setdefault('initial_lr', group['lr'])
                self.scheduler.base_lrs([group['initial_lr']
                                         for group in self.optimizer.param_groups])

    def meta_train(self, data_loader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.meta_train_iter(data_loader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_outer_loss'])}
                if 'accuracies_after' in results:
                    postfix['accuracy'] = '{0:.4f}'.format(
                        np.mean(results['accuracies_after']))
                pbar.set_postfix(**postfix)

    def meta_train_iter(self, data_loader, max_batches):
        num_batches = 0
        self.network.train()

        while num_batches < max_batches:
            for batch in data_loader:
                if num_batches > max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch, device=self.device)
                outer_loss, results = self.get_outer_loss(batch)
                yield results

                outer_loss.backward()
                self.optimizer.step()

                num_batches += 1

    def get_outer_loss(self, batch):
        _, test_targets = batch['test']
        num_tasks = test_targets.size(0)
        results = {
            'num_tasks': num_tasks,
            'inner_losses': np.zeros((self.num_adaptation_steps,
                                      num_tasks), dtype=np.float32),
            'outer_losses': np.zeros((num_tasks,), dtype=np.float32),
            'mean_outer_loss': 0.
        }

        mean_outer_loss = torch.tensor(0., device=self.device)

        for task_id, (train_inputs, train_targets, test_inputs, test_targets) \
                in enumerate(zip(*batch['train'], *batch['test'])):
            params, adaptation_results = self.adapt(train_inputs, train_targets,
                                                    num_adaptation_steps=self.num_adaptation_steps,
                                                    step_size=self.step_size, first_order=self.first_order)
            results['inner_losses'][:, task_id] = adaptation_results['inner_losses']

            with torch.set_grad_enabled(self.network.training):
                test_logits = self.network(test_inputs, params=params)
                outer_loss = self.loss_function(test_logits, test_targets)
                results['outer_losses'][task_id] = outer_loss.item()
                mean_outer_loss += outer_loss

        mean_outer_loss.div_(num_tasks)
        results['mean_outer_loss'] = mean_outer_loss.item()

        return mean_outer_loss, results

    def adapt(self, inputs, targets, num_adaptation_steps=1, step_size=0.1, first_order=False):
        params = None
        results = {'inner_losses': np.zeros((num_adaptation_steps,), dtype=np.float32)}
        for step in range(num_adaptation_steps):
            logits = self.network(inputs, params=params)
            inner_loss = self.loss_function(logits, targets)
            results['inner_losses'][step] = inner_loss.item()

            self.network.zero_grad()
            params = self.network.update_parameters(inner_loss,
                                                    step_size=step_size, params=params,
                                                    first_order=(not self.network.training) or first_order)

        return params, results

    @DeprecationWarning
    def meta_evaluate(self, dataloader, max_batches=500, verbose=True, **kwargs):
        mean_outer_loss, mean_accuracy, count = 0., 0., 0
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.meta_evaluate_iter(dataloader, max_batches=max_batches):
                pbar.update(1)
                count += 1
                mean_outer_loss += (results['mean_outer_loss']
                                    - mean_outer_loss) / count
                postfix = {'loss': '{0:.4f}'.format(mean_outer_loss)}
                if 'accuracies_after' in results:
                    mean_accuracy += (np.mean(results['accuracies_after'])
                                      - mean_accuracy) / count
                    postfix['accuracy'] = '{0:.4f}'.format(mean_accuracy)
                pbar.set_postfix(**postfix)

        mean_results = {'mean_outer_loss': mean_outer_loss}
        if 'accuracies_after' in results:
            mean_results['accuracies_after'] = mean_accuracy

        return mean_results

    @DeprecationWarning
    def meta_evaluate_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.network.eval()
        while num_batches < max_batches:
            for batch in dataloader:
                if num_batches >= max_batches:
                    break

                batch = tensors_to_device(batch, device=self.device)
                _, results = self.get_outer_loss(batch)
                yield results

                num_batches += 1
