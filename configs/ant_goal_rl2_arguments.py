import argparse

import torch

"""
Arguments fixed to...

RUNNING taskset -c 0-43 python3.5 train_ant_goal.py --algo rl2 --folder rl2huber --training-iter 31250 --gamma 0.995 --num-processes 16 --num-test-processes 1 --hidden-size 128 --ppo-lr 0.0005 --ppo-eps 1e-8 --ppo-epoch 2 --num-mini-batch 1 --use-elu 0 --num-steps 200 --clip-param 0.05 --decay-kld-rate 1 --eval-interval 100 --num-random-task-to-eval 32 --entropy-coef 0.01 --use-proper-time-limits 1 --use-gae 1 --gae_lambda 0.95 --task-len 1 --use-done 0 --use-feature-extractor 1 --rl2-reward-emb-dim 32 --rl2-action-emb-dim 32 --rl2-state-emb-dim 64 --use-rms-rew 1 --use-rms-obs 0 --use-rms-act 0 --use-rms-rew-in-policy 0 --use-huber-loss 1 --seed 
"""


def get_args(rest_args):
    parser = argparse.ArgumentParser(description="AL")

    # RL parameters
    parser.add_argument('--gamma', default=0.995, type=float, help="RL discount factor")

    # PPO-parameters
    parser.add_argument('--num-steps', type=int, default=200, help="number of steps before updating")
    parser.add_argument('--ppo-epoch', type=int, default=2, help="number of ppo epochs during update")
    parser.add_argument('--clip-param', type=float, default=0.05, help="clip parameters for ppo")
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
    parser.add_argument('--state-extractor-dim', default=None, type=int)
    parser.add_argument('--latent-extractor-dim', default=None, type=int)
    parser.add_argument('--uncertainty-extractor-dim', default=None, type=int)
    parser.add_argument('--use-huber-loss', default=True, type=lambda x: int(x) != 0,
                        help="True if Huber loss should be used in RL training")

    parser.add_argument('--use-rms-obs', type=lambda x: int(x) != 0, default=False,
                        help="True if states should be smoothed when fed to the policy")
    parser.add_argument('--use-rms-latent', type=lambda x: int(x) != 0, default=False,
                        help="True if latent space shuold be smoothed when fed to the policy")
    parser.add_argument('--decouple-rms-latent', type=lambda x: int(x) != 0, default=False,
                        help="True if 2 different smoothers should be used "
                             "to smooth mean and variance of the latent space (only for ours algo)")
    parser.add_argument('--use-rms-rew', type=lambda x: int(x) != 0, default=True,
                        help="True if rewards should be smoothed in RL training")
    parser.add_argument('--use-rms-act', type=lambda x: int(x) != 0, default=False,
                        help="True if actions should be smoothed when fed to the Policy (only for RL2)")

    # RL2
    parser.add_argument('--rl2-state-emb-dim', type=int, default=64,
                        help="RL2 state embedding dimension")
    parser.add_argument('--rl2-action-emb-dim', type=int, default=32,
                        help="RL2 action embedding dimension")
    parser.add_argument('--rl2-reward-emb-dim', type=int, default=32,
                        help="RL2 reward embedding dimension")
    parser.add_argument('--rl2-done-emb-dim', type=int, default=None,
                        help="RL done embedding dimension")
    parser.add_argument('--use-done', type=lambda x: int(x) != 0, default=False,
                        help="True if RL2 policy should be conditioned on the done signal")
    parser.add_argument('--use-rms-rew-in-policy', type=lambda x: int(x) != 0, default=False,
                        help="True if reward should be smoothed when fed to the policy (RL2 only)")
    parser.add_argument('--rl2-latent-dim', type=int, default=None)

    # GP parameters
    parser.add_argument('--n-restarts-gp', default=10, type=int, help="number of restarts for GP at meta-test time")
    parser.add_argument('--sw-size', default=10000, type=int, help="GP will use only the last sw number of samples")

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

    return args
