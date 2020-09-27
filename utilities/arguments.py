import argparse

import torch


def get_args():
    parser = argparse.ArgumentParser(description="AL")

    parser.add_argument("--algo", default="oracle",
                        help="algorithm to use: gp_ts | ours| ts_posterior | rl2 "
                             "new_posterior_multi_task | ts_opt")

    # RL parameters
    parser.add_argument('--gamma', default=0.99, type=float)

    # PPO-parameters
    parser.add_argument('--num-steps', type=int, default=150, help="number of steps before updating")
    parser.add_argument('--ppo-epoch', type=int, default=4, help="number of ppo epochs during update")
    parser.add_argument('--clip-param', type=float, default=0.2, help="clip parameters for ppo")
    parser.add_argument('--num-mini-batch', type=int, default=8, help="number of mini batches for ppo")
    parser.add_argument('--value-loss-coef', default=0.5, type=float, help="value loss coefficient for ppo")
    parser.add_argument('--entropy-coef', default=0., type=float, help="entropy coefficient for ppo")
    parser.add_argument('--max-grad-norm', default=0.5, type=float, help="maximum gradient norm in ppo updates")
    parser.add_argument('--ppo-lr', default=0.0001, type=float, help="ppo learning rate")
    parser.add_argument('--ppo-eps', default=1e-6, type=float, help="epsilon param for adam optimizer in ppo")
    parser.add_argument('--recurrent', default=False, type=lambda x: int(x) != 0, help="if the policy in ppo is recurrent")
    parser.add_argument('--hidden-size', default=64, type=int, help="number of hidden neurons in agent policy")
    parser.add_argument('--use-elu', default=True, type=lambda x: int(x) != 0, help="true if hidden neurons use elu, false for tanh")
    parser.add_argument('--use-linear-lr-decay', default=False, type=lambda x: int(x) != 0, help="whether to use or not linear lr decay")
    parser.add_argument('--use-gae', default=False, type=lambda x: int(x) != 0)
    parser.add_argument('--gae_lambda', default=0.95, type=float)
    parser.add_argument('--use-proper-time-limits', default=False, type=lambda x: int(x) != 0)
    parser.add_argument('--use-xavier', default=False, type=lambda x: int(x) != 0)

    # GP parameters
    parser.add_argument('--n-restarts-gp', default=10, type=int, help="number of restarts for GP at meta-test time")
    parser.add_argument('--sw-size', default=10000, type=int, help="GP will use only the last sw number of samples")

    # Variational inference
    parser.add_argument('--init-vae-steps', default=1000, type=int, help="initial number of inference training step")
    parser.add_argument('--vae-smart', type=lambda x: int(x) != 0, default=False)
    parser.add_argument('--vae-lr', type=float, default=1e-3)
    parser.add_argument('--use-decay-kld', type=lambda x: int(x) != 0, default=True)
    parser.add_argument('--decay-kld-rate', type=float, default=None)
    parser.add_argument('--vae-max-steps', type=int, default=None)

    # General settings
    parser.add_argument('--training-iter', default=10000, type=int, help="number of training iterations")
    parser.add_argument('--num-update-per-meta-training-iter', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=20, help="evaluate agent every x iteration")
    parser.add_argument('--log-dir', type=str, default=".")
    parser.add_argument('--num-random-task-to-eval', type=int, default=128, help="number of random task to evalute")
    parser.add_argument('--num-test-processes', type=int, default=1, help="number of processes to be used at meta-test time")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=1, help="number of envs that will be run in parallalel")
    parser.add_argument('--verbose', type=lambda x: int(x) != 0, default=True)
    parser.add_argument('--folder', type=str, default="")
    parser.add_argument('--task-len', type=int, default=1)

    # Cuda parameters
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    # Environment specific settings
    parser.add_argument('--num-signals', type=int, default=None)

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert args.algo in ['gp_ts', 'ours', 'ts_posterior', 'rl2', 'ts_opt']

    if args.vae_max_steps is None:
        args.vae_max_steps = args.num_steps

    return args
