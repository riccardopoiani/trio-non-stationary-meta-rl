import numpy as np
import torch
import gym_sin
import os
from gym import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.ant_task_generator import AntTaskGenerator
from utilities.plots import view_results
from utilities.test_arguments import get_test_args

folder_list = ["result/"]
algo_list = ["ours"]
has_track_list = [True]
store_history_list = [True]

# Task family parameters
env_name = "antfriction-v0"

friction_var_min = 0.00001
friction_var_max = 0.04
latent_dim = 8

high_act = np.ones(8, dtype=np.float32)

low_act = -np.ones(8, dtype=np.float32)

action_space = spaces.Box(low=low_act, high=high_act)

task_generator = AntTaskGenerator(n_frictions=8,
                                  friction_var_min=friction_var_min,
                                  friction_var_max=friction_var_max)
prior_std_max = task_generator.latent_max_std.tolist()
prior_std_min = task_generator.latent_min_std.tolist()

# Other settings
noise_seq_var = 0.001


def get_sin_task_sequence_full_range(alpha, n_restarts, num_test_processes, std):
    return [], [], []


def get_sequences(alpha, n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_sin, prior_seq_sin, init_prior_sin = get_sin_task_sequence_full_range(alpha, n_restarts, num_test_processes,
                                                                                  std)

    # Fill lists
    prior_sequences = [prior_seq_sin]
    gp_list_sequences = [gp_list_sin]
    init_prior = [init_prior_sin]
    return prior_sequences, gp_list_sequences, init_prior


def get_meta_test(f, algo, gp_list_sequences, sw_size, prior_sequences, init_prior_sequences,
                  num_eval_processes, task_generator, store_history, seed, log_dir,
                  device):
    res_list = []
    if algo == "rl2":
        dirs_containing_res = os.listdir(f)
        model_list = []

        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/rl2_actor_critic"))

        for i, model in enumerate(model_list):
            print("Evaluating RL2 {} / {}".format(i, len(model_list)))
            agent = RL2(hidden_size=8,
                        use_elu=True,
                        clip_param=0.2,
                        ppo_epoch=4,
                        num_mini_batch=8,
                        value_loss_coef=0.5,
                        entropy_coef=0,
                        lr=0.0005,
                        eps=1e-6,
                        max_grad_norm=0.5,
                        action_space=action_space,
                        obs_shape=(3,),
                        use_obs_env=True,
                        num_processes=32,
                        gamma=0.99,
                        device="cpu",
                        num_steps=50,
                        action_dim=1,
                        use_gae=False,
                        gae_lambda=0.95,
                        use_proper_time_limits=False)
            agent.actor_critic = model
            res = agent.meta_test(prior_sequences, task_generator, num_eval_processes, env_name, seed, log_dir)
            res_list.append(res)
    if algo == "ours":

        model_list = []
        vi_list = []
        dirs_containing_res = os.listdir(f)

        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/agent_ac"))
            vi_list.append(torch.load(f + d + "/agent_vi"))

        i = 0
        for model, vi in zip(model_list, vi_list):
            if i < 5:
                print("Evaluating Ours {} / {}".format(i, len(model_list)))
                i += 1
                agent = OursAgent(action_space=action_space,
                                  device=device,
                                  gamma=0.99,
                                  num_steps=15,
                                  num_processes=32,
                                  clip_param=0.2,
                                  ppo_epoch=4,
                                  num_mini_batch=8,
                                  value_loss_coef=0.5,
                                  entropy_coef=0.,
                                  lr=0.00005,
                                  eps=1e-5,
                                  max_grad_norm=0.5,
                                  use_linear_lr_decay=False,
                                  use_gae=None,
                                  gae_lambda=0.95,
                                  use_proper_time_limits=False,
                                  obs_shape=(2,),
                                  latent_dim=1,
                                  recurrent_policy=False,
                                  hidden_size=10,
                                  use_elu=False,
                                  variational_model=None,
                                  vae_optim=None,
                                  rescale_obs=False,
                                  max_old=None,
                                  min_old=None,
                                  vae_min_seq=None,
                                  vae_max_seq=None,
                                  max_action=None,
                                  min_action=None,
                                  max_sigma=prior_std_max,
                                  min_sigma=prior_std_min,
                                  use_decay_kld=None,
                                  decay_kld_rate=None,
                                  env_dim=111,
                                  action_dim=8
                                  )
                agent.actor_critic = model
                agent.vae = vi
                res = agent.meta_test_sequences(gp_list_sequences=gp_list_sequences,
                                                sw_size=sw_size,
                                                env_name=env_name,
                                                seed=seed,
                                                log_dir=log_dir,
                                                prior_sequences=prior_sequences,
                                                init_prior_sequences=init_prior_sequences,
                                                use_env_obs=False,
                                                num_eval_processes=1,
                                                task_generator=task_generator,
                                                store_history=store_history,
                                                verbose=True,
                                                task_len=4)
                res_list.append(res)

    if algo == "ts_opt":
        model_list = []
        vi_list = []
        dirs_containing_res = os.listdir(f)

        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/agent_ac"))
            vi_list.append(torch.load(f + d + "/agent_vi"))

        for model, vi in zip(model_list, vi_list):
            agent = PosteriorOptTSAgent(vi=None,
                                        vi_optim=None,
                                        num_steps=15,
                                        num_processes=32,
                                        device=device,
                                        gamma=0.99,
                                        latent_dim=1,
                                        use_env_obs=True,
                                        min_action=None,
                                        max_action=None,
                                        max_sigma=prior_std_max,
                                        min_sigma=prior_std_min,
                                        action_space=action_space,
                                        obs_shape=(1,),
                                        clip_param=0.2,
                                        ppo_epoch=4,
                                        num_mini_batch=8,
                                        value_loss_coef=0.5,
                                        entropy_coef=0.,
                                        lr=0.00005,
                                        eps=1e-6,
                                        max_grad_norm=0.5,
                                        use_linear_lr_decay=False,
                                        use_gae=False,
                                        gae_lambda=0.95,
                                        use_proper_time_limits=False,
                                        recurrent_policy=False,
                                        hidden_size=16,
                                        use_elu=True,
                                        rescale_obs=False,
                                        max_old=None,
                                        min_old=None,
                                        use_decay_kld=None,
                                        decay_kld_rate=None,
                                        env_dim=1,
                                        action_dim=1,
                                        vae_max_steps=60)
            agent.actor_critic = model
            agent.vi = vi

            res = agent.meta_test_sequences(gp_list_sequences=gp_list_sequences,
                                            sw_size=sw_size,
                                            env_name=env_name,
                                            seed=seed,
                                            log_dir=log_dir,
                                            prior_sequences=prior_sequences,
                                            init_prior_sequences=init_prior_sequences,
                                            num_eval_processes=num_eval_processes,
                                            task_generator=task_generator,
                                            store_history=store_history,
                                            )
            res_list.append(res)
    return res_list


def main():
    # Get arguments
    args = get_test_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    prior_sequences, gp_list_sequences, init_prior = get_sequences(alpha=args.alpha_gp,
                                                                   n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_var ** (1 / 2))

    task_generator = AntTaskGenerator(n_frictions=8,
                                      friction_var_min=friction_var_min,
                                      friction_var_max=friction_var_max)

    meta_test_results = []
    for f, algo, store_history in zip(folder_list, algo_list, store_history_list):
        r = get_meta_test(f=f, algo=algo, sw_size=args.sw_size, prior_sequences=prior_sequences,
                          init_prior_sequences=init_prior, gp_list_sequences=gp_list_sequences,
                          num_eval_processes=args.num_test_processes, task_generator=task_generator,
                          store_history=store_history, seed=args.seed, log_dir=args.log_dir,
                          device=device)
        r = np.array(r)
        meta_test_results.append(r)

    # Create python plots from meta-test results
    view_results(meta_test_results, algo_list, has_track_list, len(init_prior), prior_sequences,
                 init_priors=init_prior,
                 rescale_latent=None)

    # Dump results on CSV for report purpose


if __name__ == "__main__":
    main()
