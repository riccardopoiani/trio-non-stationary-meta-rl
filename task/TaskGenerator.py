from abc import ABC, abstractmethod

import torch
import numpy as np
import matplotlib.pyplot as plt


class TaskGenerator(ABC):

    def __init__(self):
        super(TaskGenerator, self).__init__()


class SinTaskGenerator(TaskGenerator):

    def __init__(self, x_min, x_max):
        super(SinTaskGenerator, self).__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.x_space = torch.arange(self.x_min, self.x_max, 0.01).unsqueeze(0)
        self.n_in = 1
        self.n_out = 1

    def get_data_loader_eval(self, amplitude, phase=0, n_batch=32):
        x_points = self.x_space
        y_points = amplitude * torch.sin(x_points - phase)

        x_chunk = x_points.chunk(n_batch, 1)
        y_chunk = y_points.chunk(n_batch, 1)
        real_n_batch = len(x_chunk)

        data_loader = []
        for b in range(real_n_batch):
            new_dict = {'train': [], 'test': [x_chunk[b][0].unsqueeze(1), y_chunk[b][0].unsqueeze(1)]}
            data_loader.append(new_dict)

        return data_loader

    def get_pair_tasks(self, amplitude=(1, 1), phase=(0, 0), frequency=(1, 1), noise_mean=0,
                       noise_std=0.1, num_batches=100, test_perc=0, batch_size=32):
        # Task generation
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)
        for i in range(self.n_in):
            idx[i] = torch.randint(low=0, high=self.x_space[i].shape[0], size=(num_batches, batch_size))

        x_points = torch.empty(num_batches, batch_size, self.n_in)
        y_points_1 = torch.empty(num_batches, batch_size, self.n_out)
        y_points_2 = torch.empty(num_batches, batch_size, self.n_out)
        for b in range(num_batches):
            for elem in range(batch_size):
                for dim in range(self.n_in):
                    x_points[b, elem, dim] = self.x_space[dim, idx[dim, b, elem]]

        for b in range(num_batches):
            y_clean_1 = amplitude[0] * torch.sin(frequency[0] * x_points[b] - phase[0])
            y_clean_2 = amplitude[1] * torch.sin(frequency[1] * x_points[b] - phase[1])

            noise_1 = torch.from_numpy(np.random.normal(loc=noise_mean, scale=noise_std, size=y_clean_1.shape))
            noise_2 = torch.from_numpy(np.random.normal(loc=noise_mean, scale=noise_std, size=y_clean_2.shape))

            y_points_1[b] = y_clean_1 + noise_1
            y_points_2[b] = y_clean_2 + noise_2

        # Prepare data in data_loader format (i.e. data_loader is a list of dict, each dict is a batch (key test train))
        data_loader = []
        for b in range(num_batches):
            new_dict = {}
            x_train = x_points[b][0:int(x_points[b].shape[0] * (1 - test_perc))]
            x_test = x_points[b][int(x_points[b].shape[0] * (1 - test_perc)):]

            y_train_1 = y_points_1[b][0:int(y_points_1[b].shape[0] * (1 - test_perc))]
            y_test_1 = y_points_1[b][int(y_points_1[b].shape[0] * (1 - test_perc)):]

            y_train_2 = y_points_2[b][0:int(y_points_2[b].shape[0] * (1 - test_perc))]
            y_test_2 = y_points_2[b][int(y_points_2[b].shape[0] * (1 - test_perc)):]

            new_dict['train'] = [x_train, y_train_1, y_train_2]
            new_dict['test'] = [x_test, y_test_1, y_test_2]

            data_loader.append(new_dict)

        return data_loader

    def get_data_loader(self, amplitude=1, phase=0, frequency=1, noise_mean=0, noise_std=0.1,
                        num_batches=100, test_perc=1, batch_size=32):
        # Task generation
        idx = torch.empty((self.n_in, num_batches, batch_size), dtype=torch.long)
        for i in range(self.n_in):
            idx[i] = torch.randint(low=0, high=self.x_space[i].shape[0], size=(num_batches, batch_size))

        x_points = torch.empty(num_batches, batch_size, self.n_in)
        y_points = torch.empty(num_batches, batch_size, self.n_out)
        for b in range(num_batches):
            for elem in range(batch_size):
                for dim in range(self.n_in):
                    x_points[b, elem, dim] = self.x_space[dim, idx[dim, b, elem]]

        for b in range(num_batches):
            y_clean = amplitude * torch.sin(frequency * x_points[b] - phase)
            noise = torch.from_numpy(np.random.normal(loc=noise_mean, scale=noise_std, size=y_clean.shape))
            y_points[b] = y_clean + noise

        # Prepare data in data_loader format (i.e. data_loader is a list of dict, each dict is a batch (key test train))
        data_loader = []
        for b in range(num_batches):
            new_dict = {}
            x_train = x_points[b][0:int(x_points[b].shape[0] * (1 - test_perc))]
            x_test = x_points[b][int(x_points[b].shape[0] * (1 - test_perc)):]

            y_train = y_points[b][0:int(y_points[b].shape[0] * (1 - test_perc))]
            y_test = y_points[b][int(y_points[b].shape[0] * (1 - test_perc)):]

            new_dict['train'] = [x_train, y_train]
            new_dict['test'] = [x_test, y_test]

            data_loader.append(new_dict)

        return data_loader

    def plot_prediction_vs_task(self, net: torch.nn.Module, amplitude_list, phase_list):
        n_task = len(amplitude_list)
        y_clean = [amplitude_list[i] * torch.sin(self.x_space - phase_list[i]) for i in range(n_task)]
        y_pred = net(self.x_space.unsqueeze(1)).detach()

        for task in range(n_task):
            plt.plot(self.x_space, y_clean[task], label="True function {}".format(task))
        plt.plot(self.x_space, y_pred, label="Prediction")
        plt.legend()
        plt.show()
