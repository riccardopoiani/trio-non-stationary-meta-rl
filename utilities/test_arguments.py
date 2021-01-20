import argparse

import torch


def get_test_args():
    parser = argparse.ArgumentParser(description="AL")

    # GP parameters
    parser.add_argument('--n-restarts-gp', default=10, type=int, help="number of restarts for GP at meta-test time")
    parser.add_argument('--sw-size', default=1000, type=int, help="GP will use only the last sw number of samples")

    # General settings
    parser.add_argument('--training-iter', default=10000, type=int, help="number of training iterations")
    parser.add_argument('--log-dir', type=str, default=".")
    parser.add_argument('--num-test-processes', type=int, default=1,
                        help="number of processes to be used at meta-test time")
    parser.add_argument('--num-processes', type=int, default=1, help="number of envs that will be run in parallalel")
    parser.add_argument('--verbose', type=lambda x: int(x) != 0, default=True)
    parser.add_argument('--save-fig', type=lambda x: int(x) != 0, default=True)
    parser.add_argument('--dump-data', type=lambda x: int(x) != 0, default=True)
    parser.add_argument('--task-len', type=int, default=1)
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--n-runs', type=int, default=1)

    # Cuda parameters
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    # Folder parameters
    parser.add_argument('--rl2-folder', default=None,
                        type=str, help="folder that contains folders of rl2 training results")
    parser.add_argument('--ts-folder', default=None,
                        type=str, help="folder that contains folders of ts training results")
    parser.add_argument('--bayes-folder', default=None,
                        type=str, help="folder that contains folders of bayes training results")

    parser.add_argument('--bayes-sig1-folder', default=None,
                        type=str, help="folder that contains folders of bayes training results for "
                                       "minigolf with 1 additional signal")
    parser.add_argument('--bayes-sig2-folder', default=None,
                        type=str, help="folder that contains folders of bayes training results for "
                                       "minigolf with 2 additional signals")
    parser.add_argument('--bayes-sig3-folder', default=None,
                        type=str, help="folder that contains folders of bayes training results "
                                       "for minigolf with 3 additional signals")

    parser.add_argument('--output-folder', default=None, required=True,
                        type=str, help="Folder where to store results")

    args, rest_args = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args, rest_args
