import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="AL")

    parser.add_argument("--algo", default="oracle",
                        help="algorithm to use: oracle | fixed_id | posterior_multi_task | ours")

    # RL parameters
    parser.add_argument('--gamma', default=0.99)

    # PPO-parameters
    parser.add_argument('--num-steps', type=int, default=150, help="number of steps before updating")
    parser.add_argument('--ppo-epoch', type=int, default=4, help="number of ppo epochs during update")
    parser.add_argument('--clip-param', default=0.2)
    parser.add_argument('--num-mini-batch', type=int, default=8)
    parser.add_argument('--value-loss-coef', default=0.5, type=float)
    parser.add_argument('--entropy-coef', default=0., type=float)
    parser.add_argument('--max-grad-norm', default=0.5, type=float)
    parser.add_argument('--ppo-lr', default=0.0001, type=float)
    parser.add_argument('--ppo-eps', default=1e-6, type=float)
    parser.add_argument('--recurrent', default=False, type=bool)
    parser.add_argument('--hidden-size', default=64, type=int)
    parser.add_argument('--use-elu', default=True, type=bool)
    parser.add_argument('--use-linear-lr-decay', default=False, type=bool)
    parser.add_argument('--use-gae', default=False, type=bool)
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--use-proper-time-limits', default=False, type=bool)

    # Variational inference
    parser.add_argument('--num-vae-steps', default=1, type=int)
    parser.add_argument('--init-vae-steps', default=1000, type=int)

    # General settings
    parser.add_argument('--training-iter', default=10000, type=int)
    parser.add_argument('--rescale-obs', default=True, type=bool)
    parser.add_argument('--num-update-per-meta-training-iter', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=20)
    parser.add_argument('--log-dir', type=str, default=".")
    parser.add_argument('--num-task-to-eval', type=int, default=128)
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1, help="number of envs that will be run in parallalel")
    parser.add_argument('--run_windows', type=bool, default=True)

    # Cuda parameters
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.algo in ['oracle', 'fixed_id', 'posterior_multi_task', 'ours']

    return args
