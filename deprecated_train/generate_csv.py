import pickle
import os

import numpy as np
import torch
import envs
from gym import spaces
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

from utilities.plots.plots import create_csv_rewards, create_csv_tracking


def f_double_step(x, y_min=-0.2, y_max=0.5, first_peak=10, second_peak=20):
    if x < first_peak or x > second_peak:
        return y_min
    return y_max


def f_linear(x, m=0.08, q=-0.8):
    return x * m + q


def f_const(x, const=0):
    return const


def f_mixture_changes(x):
    if x < 10:
        return 1
    elif 20 > x >= 10:
        return 0 - (x / 20)
    elif x >= 20:
        return -1 + np.power((x - 20), 4) / (130321 / 2)


def get_const_task_sequence(n_restarts, num_test_processes, std):
    kernel = C(1) * RBF(1) + WhiteKernel(0.05, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[f_const(0)], [0.2 ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 20):
        friction = f_const(idx)
        prior_seq.append(torch.tensor([[friction], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_linear_task_sequence(n_restarts, num_test_processes, std):
    kernel = C(1) * RBF(1) + WhiteKernel(0.05, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[f_linear(0)], [0.2 ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 15):
        friction = f_linear(idx)
        prior_seq.append(torch.tensor([[friction], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_double_step_sequences(n_restarts, num_test_processes, std):
    kernel = C(1) * RBF(1) + WhiteKernel(0.05, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[f_double_step(0)], [0.2 ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 30):
        friction = f_double_step(idx)
        prior_seq.append(torch.tensor([[friction], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_strange_sequences(n_restarts, num_test_processes, std):
    kernel = C(1) * RBF(1) + WhiteKernel(0.05, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[f_mixture_changes(0)], [0.2 ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 40):
        friction = f_mixture_changes(idx)
        prior_seq.append(torch.tensor([[friction], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_const, prior_seq_const, init_prior_const = get_const_task_sequence(n_restarts, num_test_processes, std)
    gp_list_lin, prior_seq_lin, init_prior_lin = get_linear_task_sequence(n_restarts, num_test_processes, std)
    gp_list_step, prior_seq_step, init_prior_step = get_double_step_sequences(n_restarts, num_test_processes, std)
    gp_list_mix, prior_seq_mix, init_prior_mix = get_strange_sequences(n_restarts, num_test_processes, std)

    # Fill lists
    p = [prior_seq_lin, prior_seq_const, prior_seq_step, prior_seq_mix]
    gp = [gp_list_lin, gp_list_const, gp_list_step, gp_list_mix]
    ip = [init_prior_lin, init_prior_const, init_prior_step, init_prior_mix]
    return p, gp, ip


def main():
    folder = "result/metatest/scalegauss/data_results.pkl"
    num_seq = 4
    seq_len_list = [15, 20, 30, 40]
    sequence_name_list = ['linear', 'const', "doublestep", 'mix']

    # Reading meta-test results
    objects = []
    with (open(folder, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    meta_test_res = objects[0]

    # Retrieving test sequences
    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=1,
                                                                   num_test_processes=20,
                                                                   std=0.001 ** (1 / 2))

    # Dumping results on file
    create_csv_rewards(meta_test_res, ['Ours', 'TS', 'RL'], [True, True, False], num_seq,
                       prior_sequences, seq_len_list, sequence_name_list,
                       "result/metatest/scalegauss/")

    create_csv_tracking(meta_test_res, ['Ours', 'TS', 'RL2'], [True, True, False], num_seq,
                        prior_sequences, seq_len_list, sequence_name_list,
                        "result/metatest/scalegauss/", init_prior, [-0.8, 0.8], 1)


if __name__ == "__main__":
    main()
