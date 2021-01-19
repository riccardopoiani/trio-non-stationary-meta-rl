import argparse

import torch


def get_args(rest_args):
    parser = argparse.ArgumentParser(description="AL")

    # RL parameters
    parser.add_argument('--gamma', default=0.995, type=float, help="RL discount factor")

    # PPO-parameters
    parser.add_argument('--num-steps', type=int, default=200, help="number of steps before updating")
    parser.add_argument('--ppo-epoch', type=int, default=2, help="number of ppo epochs during update")
    parser.add_argument('--clip-param', type=float, default=0.1, help="clip parameters for ppo")
    parser.add_argument('--num-mini-batch', type=int, default=1, help="number of mini batches for ppo")
    parser.add_argument('--value-loss-coef', default=0.5, type=float, help="value loss coefficient for ppo")
    parser.add_argument('--entropy-coef', default=0.01, type=float, help="entropy coefficient for ppo")
    parser.add_argument('--max-grad-norm', default=0.5, type=float, help="maximum gradient norm in ppo updates")
    parser.add_argument('--ppo-lr', default=0.0005, type=float, help="ppo learning rate")
    parser.add_argument('--ppo-eps', default=1e-8, type=float, help="epsilon param for adam optimizer in ppo")
    parser.add_argument('--recurrent', default=False, type=lambda x: int(x) != 0,
                        help="if the policy in ppo is recurrent")
    parser.add_argument('--hidden-size', default=128, type=int, help="number of hidden neurons in agent policy")
    parser.add_argument('--use-elu', default=False, type=lambda x: int(x) != 0,
                        help="true if hidden neurons use elu, false for tanh")
    parser.add_argument('--use-linear-lr-decay', default=False, type=lambda x: int(x) != 0,
                        help="whether to use or not linear lr decay")
    parser.add_argument('--use-gae', default=True, type=lambda x: int(x) != 0,
                        help="True if Generalized Advantage Estimation should be used")
    parser.add_argument('--gae_lambda', default=0.95, type=float,
                        help="Generalized Advantage Estimation lambda parameter")
    parser.add_argument('--use-proper-time-limits', default=True, type=lambda x: int(x) != 0,
                        help="If False, time limits will be considered at the same way of terminal states")
    parser.add_argument('--use-xavier', default=False, type=lambda x: int(x) != 0,
                        help="true if xavier init will be used in policy, false if orthogonal init will be used")
    parser.add_argument('--use-feature-extractor', default=True, type=lambda x: int(x) != 0)
    parser.add_argument('--state-extractor-dim', default=80, type=int)
    parser.add_argument('--latent-extractor-dim', default=46, type=int)
    parser.add_argument('--uncertainty-extractor-dim', default=2, type=int)
    parser.add_argument('--use-huber-loss', default=True, type=lambda x: int(x) != 0,
                        help="True if Huber loss should be used in RL training")

    parser.add_argument('--use-rms-obs', type=lambda x: int(x) != 0, default=True,
                        help="True if states should be smoothed when fed to the policy")
    parser.add_argument('--use-rms-latent', type=lambda x: int(x) != 0, default=True,
                        help="True if latent space shuold be smoothed when fed to the policy")
    parser.add_argument('--decouple-rms-latent', type=lambda x: int(x) != 0, default=False,
                        help="True if 2 different smoothers should be used "
                             "to smooth mean and variance of the latent space (only for ours algo)")
    parser.add_argument('--use-rms-rew', type=lambda x: int(x) != 0, default=True,
                        help="True if rewards should be smoothed in RL training")

    # GP parameters
    parser.add_argument('--n-restarts-gp', default=10, type=int, help="number of restarts for GP at meta-test time")
    parser.add_argument('--sw-size', default=10000, type=int, help="GP will use only the last sw number of samples")

    # Variational inference
    parser.add_argument('--init-vae-steps', default=7500, type=int, help="initial number of inference training step")
    parser.add_argument('--vae-smart', type=lambda x: int(x) != 0, default=True,
                        help="True if samples collected using bad priors will be used to train the network")
    parser.add_argument('--vae-lr', type=float, default=1e-3, help="learning rate used to train VAE network")
    parser.add_argument('--use-decay-kld', type=lambda x: int(x) != 0, default=True,
                        help="whether to make the KLD loss decay as more samples are used to produce the inference")
    parser.add_argument('--decay-kld-rate', type=float, default=0.1,
                        help="decay parameter of the KLD loss used in inference training")
    parser.add_argument('--vae-max-steps', type=int, default=None,
                        help="Maximum number of steps per batch that will be used to train VAE")
    parser.add_argument('--vae-state-emb-dim', type=int, default=64,
                        help="Dimension of the embedding layer concerning the state in VAE network")
    parser.add_argument('--vae-action-emb-dim', type=int, default=32,
                        help="Dimension of the embedding layer concerning the action in VAE network")
    parser.add_argument('--vae-reward-emb-dim', type=int, default=32,
                        help="Dimension of the embedding layer concerning the reward in VAE network")
    parser.add_argument('--vae-prior-emb-dim', type=int, default=2,
                        help="Dimension of the embedding layer concerning the prior in VAE network")
    parser.add_argument('--vae-gru-dim', type=int, default=128, help="Number of units in the GRU dim. of VAE network")
    parser.add_argument('--detach-every', type=int, default=50, help="Break VAE back-propagation through time"
                                                                     "after this number steps. If None, "
                                                                     "back-propagation won't be truncated")

    # General settings
    parser.add_argument('--training-iter', default=31250, type=int, help="number of training iterations")
    parser.add_argument('--eval-interval', type=int, default=100, help="evaluate agent every x iteration")
    parser.add_argument('--log-dir', type=str, default=".")
    parser.add_argument('--num-random-task-to-eval', type=int, default=32, help="number of random task to evalute")
    parser.add_argument('--num-test-processes', type=int, default=1,
                        help="number of processes to be used at meta-test time")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--num-processes', type=int, default=16, help="number of envs that will be run in parallalel")
    parser.add_argument('--verbose', type=lambda x: int(x) != 0, default=True)
    parser.add_argument('--folder', type=str, default="")
    parser.add_argument('--task-len', type=int, default=1, help="Number of episodes at meta-test time of the same task")

    # Cuda parameters
    parser.add_argument('--no-cuda', action='store_true', default=True, help='disables CUDA training')
    parser.add_argument('--cuda-deterministic', action='store_true', default=False,
                        help="sets flags for determinism when using CUDA (potentially slow!)")

    args = parser.parse_args(rest_args)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.vae_max_steps is None:
        args.vae_max_steps = args.num_steps

    return args
