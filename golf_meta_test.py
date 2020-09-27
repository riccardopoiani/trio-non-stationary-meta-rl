import gym_sin
import numpy as np
import torch
import os
from gym import spaces
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.mini_golf_task_generator import MiniGolfTaskGenerator
from utilities.folder_management import handle_folder_creation
from utilities.plots import view_results, create_csv_rewards, create_csv_tracking
from utilities.test_arguments import get_test_args

import warnings
warnings.filterwarnings(action='ignore')

folder = "result/metatest/minigolf/"
env_name = "gym_sin:minigolf-v0"
folder_list = ["result/golf_fixed_final/ours/",
               "result/golf_fixed_final/tsopt/",
               "result/golf_fixed_final/rl2low/"]
algo_list = ['ours', 'ts_opt', 'rl2']
label_list = ['ours', 'ts_opt', 'rl2']
has_track_list = [True, True, False]
store_history_list = [True, True, False]
prior_var_min = 0.001
prior_var_max = 0.2
noise_seq_var = 0.001
min_action = 1e-5
max_action = 10.
action_space = spaces.Box(low=min_action,
                          high=max_action,
                          shape=(1,))

num_seq = 2
seq_len_list = [100, 110]
sequence_name_list = ['sin', 'sawtooth']


def f_sin(x, freq=0.1, offset=-0.7, a=-0.2):
    t = a * np.sin(freq * x) + offset
    return t


def f_sawtooth(x, period=50):
    saw_tooth = 2 * (x / period - np.floor(0.5 + x / period))
    saw_tooth = (-0.6 - (-1)) / (1 - (-1)) * (saw_tooth - 1) - 0.6
    return saw_tooth


def get_sin_task_sequence_full_range(n_restarts, num_test_processes, std):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[0.], [0.2 ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 100):
        friction = f_sin(idx)
        prior_seq.append(torch.tensor([[friction], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sawtooth_wave(n_restarts, num_test_processes, std):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[0.], [0.2 ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 110):
        friction = f_sawtooth(idx)
        prior_seq.append(torch.tensor([[friction], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_sin, prior_seq_sin, init_prior_sin = get_sin_task_sequence_full_range(n_restarts, num_test_processes, std)
    gp_list_saw, prior_seq_saw, init_prior_saw = get_sawtooth_wave(n_restarts, num_test_processes, std)

    # Fill lists
    p = [prior_seq_sin, prior_seq_saw]
    gp = [gp_list_sin, gp_list_saw]
    ip = [init_prior_sin, init_prior_saw]
    return p, gp, ip


def get_meta_test(algo, gp_list_sequences, sw_size, prior_sequences, init_prior_sequences,
                  num_eval_processes, task_generator, store_history, seed, log_dir,
                  device, task_len, model, vi):
    res = None
    if algo == "rl2":
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
                    num_steps=20,
                    action_dim=1,
                    use_gae=False,
                    gae_lambda=0.95,
                    use_proper_time_limits=False,
                    use_xavier=False)
        agent.actor_critic = model
        res = agent.meta_test(prior_sequences, task_generator, num_eval_processes, env_name, seed, log_dir,
                              task_len)
    elif algo == "ours":
        agent = OursAgent(action_space=action_space,
                          device=device,
                          gamma=0.99,
                          num_steps=20,
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
                          obs_shape=(3,),
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
                          max_sigma=[prior_var_max ** (1 / 2)],
                          min_sigma=[prior_var_min ** (1 / 2)],
                          use_decay_kld=None,
                          decay_kld_rate=None,
                          env_dim=1,
                          action_dim=1,
                          use_xavier=False
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
                                        use_env_obs=True,
                                        num_eval_processes=num_eval_processes,
                                        task_generator=task_generator,
                                        store_history=store_history,
                                        task_len=task_len)
    elif algo == "ts_opt":
        agent = PosteriorOptTSAgent(vi=None,
                                    vi_optim=None,
                                    num_steps=20,
                                    num_processes=32,
                                    device=device,
                                    gamma=0.99,
                                    latent_dim=1,
                                    use_env_obs=True,
                                    min_action=None,
                                    max_action=None,
                                    max_sigma=[prior_var_max ** (1 / 2)],
                                    min_sigma=[prior_var_min ** (1 / 2)],
                                    action_space=action_space,
                                    obs_shape=(2,),
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
                                    use_xavier=False,
                                    vae_max_steps=None)
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
                                        task_len=task_len)

    else:
        raise RuntimeError()

    return res


def main(args, model, vi, algo, store, seed):
    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_var ** (1 / 2))

    task_generator = MiniGolfTaskGenerator(prior_var_min=prior_var_min, prior_var_max=prior_var_max)

    return get_meta_test(algo=algo, sw_size=args.sw_size, prior_sequences=prior_sequences,
                         init_prior_sequences=init_prior, gp_list_sequences=gp_list_sequences,
                         num_eval_processes=args.num_test_processes, task_generator=task_generator,
                         store_history=store, seed=seed, log_dir=args.log_dir,
                         device=device, task_len=args.task_len, model=model, vi=vi)


def run(id, seed, args, model, vi, algo, store):
    # Eventually fix here the seeds for additional sources of randomness (e.g. tensorflow)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Starting run {} algo {} seed {}".format(id, algo, seed))
    r = main(args=args, model=model, vi=vi, algo=algo, store=store, seed=seed)
    print("Done run {} algo {} seed {}".format(id, algo, seed))
    return r


# Scheduling runs: ENTRY POINT
args = get_test_args()

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

seeds = [np.random.randint(1000000) for _ in range(100)]

r_ours = []
r_ts = []
r_rl2 = []

for algo, f, sh in zip(algo_list, folder_list, store_history_list):
    if algo == 'ours':
        model_list = []
        vi_list = []
        dirs_containing_res = os.listdir(f)
        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/agent_ac"))
            vi_list.append(torch.load(f + d + "/agent_vi"))
        r_ours = (Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=vi, algo=algo, store=sh)
            for id, seed, model, vi in zip(range(len(model_list)), seeds, model_list, vi_list)))
    elif algo == "ts_opt":
        model_list = []
        vi_list = []
        dirs_containing_res = os.listdir(f)
        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/agent_ac"))
            vi_list.append(torch.load(f + d + "/agent_vi"))
        r_ts = (Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=vi, algo=algo, store=sh)
            for id, seed, model, vi in zip(range(len(model_list)), seeds, model_list, vi_list)))
    elif algo == "rl2":
        model_list = []
        dirs_containing_res = os.listdir(f)
        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/rl2_actor_critic"))
        r_rl2 = (Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=None, algo=algo, store=sh)
            for id, seed, model in zip(range(len(model_list)), seeds, model_list)))

print("END ALL RUNS")

meta_test_res = [r_ours, r_ts, r_rl2]
with open("temp.pkl", "wb") as output:
    import pickle

    pickle.dump(meta_test_res, output)

print(np.array(meta_test_res[0]).shape)
print(np.array(meta_test_res[0]).shape)

# Create python plots from meta-test results
prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                               num_test_processes=args.num_test_processes,
                                                               std=noise_seq_var ** (1 / 2))

fd, folder_path_with_date = handle_folder_creation(result_path=folder)
view_results(meta_test_res, label_list, has_track_list, len(init_prior), prior_sequences,
             init_priors=init_prior,
             rescale_latent=[0.01, 2],
             dump_data=args.dump_data,
             save_fig=args.save_fig,
             folder=folder_path_with_date,
             view_tracking=True)

create_csv_rewards(r_list=meta_test_res,
                   label_list=label_list,
                   has_track_list=has_track_list,
                   num_seq=num_seq,
                   seq_len_list=seq_len_list,
                   sequence_name_list=sequence_name_list,
                   folder_path_with_date=folder_path_with_date,
                   prior_seqs=prior_sequences)

create_csv_tracking(r_list=meta_test_res,
                    label_list=label_list,
                    has_track_list=has_track_list,
                    num_seq=len(init_prior),
                    prior_seqs=prior_sequences,
                    rescale_latent=[0.01, 2],
                    sequence_name_list=sequence_name_list,
                    folder_path_with_date=folder_path_with_date,
                    init_priors=init_prior,
                    seq_len_list=seq_len_list)

fd.close()
