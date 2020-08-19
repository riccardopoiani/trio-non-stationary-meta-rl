import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="AL")

    parser.add_argument("--algo", default="oracle",
                        help="algorithm to use: oracle | fixed_id | posterior_multi_task | ours"
                             "new_posterior_multi_task")

    # RL parameters
    parser.add_argument('--gamma', default=0.99)

    # PPO-parameters
    parser.add_argument('--num-steps', type=int, default=150, help="number of steps before updating")
    parser.add_argument('--ppo-epoch', type=int, default=4, help="number of ppo epochs during update")
    parser.add_argument('--clip-param', default=0.2, help="clip parameters for ppo")
    parser.add_argument('--num-mini-batch', type=int, default=8, help="number of mini batches for ppo")
    parser.add_argument('--value-loss-coef', default=0.5, type=float, help="value loss coefficient for ppo")
    parser.add_argument('--entropy-coef', default=0., type=float, help="entropy coefficient for ppo")
    parser.add_argument('--max-grad-norm', default=0.5, type=float, help="maximum gradient norm in ppo updates")
    parser.add_argument('--ppo-lr', default=0.0001, type=float, help="ppo learning rate")
    parser.add_argument('--ppo-eps', default=1e-6, type=float, help="epsilon param for adam optimizer in ppo")
    parser.add_argument('--recurrent', default=False, type=bool, help="if the policy in ppo is recurrent")
    parser.add_argument('--hidden-size', default=64, type=int, help="number of hidden neurons in agent policy")
    parser.add_argument('--use-elu', default=True, type=bool, help="true if hidden neurons use elu, false for tanh")
    parser.add_argument('--use-linear-lr-decay', default=False, type=bool, help="whether to use or not linear lr decay")
    parser.add_argument('--use-gae', default=False, type=bool)
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--use-proper-time-limits', default=False, type=bool)

    # Identification parameters
    parser.add_argument('--hidden-size-id', default=32, type=int)
    parser.add_argument('--use-elu-id', default=True, type=bool)
    parser.add_argument('--recurrent-id', default=False, type=bool)

    parser.add_argument('--clip-param-id', default=0.2, type=float)
    parser.add_argument('--ppo-epoch-id', default=4, type=int)
    parser.add_argument('--value-loss-coef-id', default=0.5, type=float)
    parser.add_argument('--lr-id', default=0.00005, type=float)
    parser.add_argument('--eps-id', default=1e-6, type=float)
    parser.add_argument('--max-grad-norm-id', default=0.5, type=float)
    parser.add_argument('--num-step-id', default=30, type=float)
    parser.add_argument('--gamma-id', default=0.9, type=float)
    parser.add_argument('--num-mini-batch-id', default=8, type=int)
    parser.add_argument('--entropy-coef-id', default=0., type=float)

    parser.add_argument('--training-iter-opt', default=2000, type=int)
    parser.add_argument('--training-iter-id', default=2000, type=int)
    parser.add_argument('--max-id-iteration', default=10, type=int)

    # GP parameters
    parser.add_argument('--n-restarts-gp', default=1, type=int, help="number of restarts for GP at meta-test time")
    parser.add_argument('--alpha-gp', default=0.25, type=float, help="alpha parameter for GP at meta-test time")
    parser.add_argument('--sw-gp', default=10, type=int, help="GP will use only the last sw number of samples")

    # Variational inference
    parser.add_argument('--num-vae-steps', default=1, type=int, help="number of variational steps for each ppo update")
    parser.add_argument('--init-vae-steps', default=1000, type=int, help="initial number of inference training step")

    # General settings
    parser.add_argument('--training-iter', default=10000, type=int, help="number of training iterations")
    parser.add_argument('--rescale-obs', default=True, type=bool, help="if observations should be rescaled or not")
    parser.add_argument('--num-update-per-meta-training-iter', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=20, help="evaluate agent every x iteration")
    parser.add_argument('--log-dir', type=str, default=".")
    parser.add_argument('--num_random_task_to_eval', type=int, default=128, help="number of random task to evalute")
    parser.add_argument('--num_test_processes', type=int, default=2, help="number of processes to be used at meta-test time")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1, help="number of envs that will be run in parallalel")
    parser.add_argument('--run_windows', type=bool, default=True)

    # Cuda parameters
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.algo in ['oracle', 'fixed_id', 'posterior_multi_task', 'ours', 'new_posterior_multi_task']

    return args
