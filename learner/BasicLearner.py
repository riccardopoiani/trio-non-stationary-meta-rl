import torch
from tqdm import tqdm

from utilities.meta_utils import tensors_to_device


class BasicLearner(object):
    """
    Standard regressor that performs training and evaluation using data_loaders.
    Data loaders are nothing but a list of dictionaries with two keys: 'train' and 'test'.
    In both 'train' and 'test' we find a list of two elements: one containing the inputs
    and one containing the outputs
    """

    def __init__(self, network: torch.nn.Module, optimizer, loss_function, scheduler=None, device="cpu"):
        self.device = device
        self.network: torch.nn.Module = network
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.scheduler = scheduler
        self.network.to(self.device)

    def evaluate(self, data_loader, verbose=True, **kwargs):
        n_batches = len(data_loader)
        with tqdm(total=n_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.evaluate_iter(data_loader):
                pbar.update(1)
                eval_loss = results['eval_loss']
                postfix = {'eval_loss': '{0:.4f}'.format(results['eval_loss'])}
                pbar.set_postfix(**postfix)
        return eval_loss

    def evaluate_iter(self, data_loader):
        results = {'eval_loss': 0.}

        mean_loss = torch.tensor(0., device=self.device)

        for i, batch in enumerate(data_loader):
            batch = batch['test']
            batch = tensors_to_device(batch, self.device)
            test_input = batch[0]
            test_targets = batch[1]
            pred = self.network(test_input).detach()
            loss = self.loss_function(pred, test_targets)

            mean_loss += loss.item()
            results['eval_loss'] = mean_loss.div(i)
            yield results

    def train(self, data_loader, max_batches=500, verbose=True, **kwargs):
        with tqdm(total=max_batches, disable=not verbose, **kwargs) as pbar:
            for results in self.train_iter(data_loader, max_batches=max_batches):
                pbar.update(1)
                postfix = {'loss': '{0:.4f}'.format(results['mean_loss'])}
                pbar.set_postfix(**postfix)

    def train_iter(self, dataloader, max_batches=500):
        num_batches = 0
        self.network.train()

        results = {
            'num_tasks': max_batches,
            'mean_loss': 0.
        }

        mean_loss = torch.tensor(0., device=self.device)

        while num_batches < max_batches:
            for batch in dataloader:
                batch_train = batch['train']

                if num_batches > max_batches:
                    break

                if self.scheduler is not None:
                    self.scheduler.step(epoch=num_batches)

                self.optimizer.zero_grad()

                batch = tensors_to_device(batch_train, device=self.device)
                train_inputs = batch[0]
                train_targets = batch[1]
                pred = self.network(train_inputs)
                loss = self.loss_function(pred, train_targets)
                loss.backward()
                self.optimizer.step()
                mean_loss += loss.item()
                results['mean_loss'] = mean_loss.div(num_batches)
                yield results

                num_batches += 1