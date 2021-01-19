import numpy as np
import torch
import os
import pickle
import envs
from gym import spaces
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

from configs import golf_bayes_arguments
from learner.bayes import BayesAgent
from task.mini_golf_with_signals_generator import MiniGolfSignalsTaskGenerator
from utilities.folder_management import handle_folder_creation
from utilities.plots.plots import create_csv_rewards
from utilities.test_arguments import get_test_args

# General parameters
folder = "result/metatest/minigolfrobust/"
env_name = 'golfsignals-v0'
folder_list = ["result/golf_robust/golf_sig_3/bayessig3/",
               "result/golf_robust/golf_sig_2/bayessig2/",
               "result/golf_robust/golf_sig_1/bayessig1/",
               "result/golf/bayes/"]
algo_list = ['bayes', 'bayes', 'bayes', 'bayes']
label_list = ['bayes_3', 'bayes_2', 'bayes_1', 'bayes_0']
action_dim = 1
use_env_obs = True
num_signals_list = [3, 2, 1, 0]
has_track_list = [True, True, True, True]
store_history_list = [True, True, True, True]
prior_var_min = 0.001
prior_var_max = 0.2
noise_seq_var = 0.001
min_action = 1e-5
max_action = 10.
action_space = spaces.Box(low=min_action,
                          high=max_action,
                          shape=(1,))

num_seq = 3
seq_len_list = [100, 110]
sequence_name_list = ['sin', 'sawtooth']


def f_sin(x, freq=0.1, offset=-0.7, a=-0.2):
    t = a * np.sin(freq * x) + offset
    return t


def f_sawtooth(x, period=50):
    saw_tooth = 2 * (x / period - np.floor(0.5 + x / period))
    saw_tooth = (-0.6 - (-1)) / (1 - (-1)) * (saw_tooth - 1) - 0.6
    return saw_tooth + 0.3


def get_sin_task_sequence_full_range(n_restarts, num_test_processes, std, latent_dim):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for _ in range(latent_dim):
        curr_dim_list = []
        for _ in range(num_test_processes):
            curr_dim_list.append(GaussianProcessRegressor(kernel=kernel,
                                                          normalize_y=False,
                                                          n_restarts_optimizer=n_restarts))
        gp_list.append(curr_dim_list)

    p_mean = []
    p_var = []
    for dim in range(latent_dim):
        p_mean.append(0.)
        p_var.append(0.2 ** (1 / 2))
    init_prior_test = [torch.tensor([p_mean, p_var], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 100):
        friction = f_sin(idx)
        p_mean = []
        p_var = []

        for dim in range(latent_dim):
            if dim == 0:
                p_mean.append(friction)
            else:
                p_mean.append(np.random.uniform(low=-1, high=1))
            p_var.append(std ** 2)

        prior_seq.append(torch.tensor([p_mean, p_var], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sawtooth_wave(n_restarts, num_test_processes, std, latent_dim):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for _ in range(latent_dim):
        curr_dim_list = []
        for _ in range(num_test_processes):
            curr_dim_list.append(GaussianProcessRegressor(kernel=kernel,
                                                          normalize_y=False,
                                                          n_restarts_optimizer=n_restarts))
        gp_list.append(curr_dim_list)

    p_mean = []
    p_var = []
    for dim in range(latent_dim):
        p_mean.append(0.)
        p_var.append(0.2 ** (1 / 2))
    init_prior_test = [torch.tensor([p_mean, p_var], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, 110):
        friction = f_sawtooth(idx)
        p_mean = []
        p_var = []

        for dim in range(latent_dim):
            if dim == 0:
                p_mean.append(friction)
            else:
                p_mean.append(np.random.uniform(low=-1, high=1))
            p_var.append(std ** 2)

        prior_seq.append(torch.tensor([p_mean, p_var], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, std, num_sig):
    # Retrieve task
    gp_list_sin, prior_seq_sin, init_prior_sin = get_sin_task_sequence_full_range(n_restarts, num_test_processes,
                                                                                  std, num_sig+1)
    gp_list_saw, prior_seq_saw, init_prior_saw = get_sawtooth_wave(n_restarts, num_test_processes, std,
                                                                   num_sig+1)

    # Fill lists
    p = [prior_seq_sin, prior_seq_saw]
    gp = [gp_list_sin, gp_list_saw]
    ip = [init_prior_sin, init_prior_saw]
    return p, gp, ip


def get_meta_test(algo, gp_list_sequences, sw_size, prior_sequences, init_prior_sequences,
                  num_eval_processes, task_generator, store_history, seed, log_dir,
                  device, task_len, model, vi, rest_args, num_signals):
    if algo == "bayes":
        algo_args = golf_bayes_arguments.get_args(rest_args)
        state_dim = 1 + num_signals
        latent_dim = 1 + num_signals
        agent = BayesAgent(action_space=action_space,
                           device=device,
                           gamma=algo_args.gamma,
                           num_steps=algo_args.num_steps,
                           num_processes=algo_args.num_processes,
                           clip_param=algo_args.clip_param,
                           ppo_epoch=algo_args.ppo_epoch,
                           num_mini_batch=algo_args.num_mini_batch,
                           value_loss_coef=algo_args.value_loss_coef,
                           entropy_coef=algo_args.entropy_coef,
                           lr=algo_args.ppo_lr,
                           eps=algo_args.ppo_eps,
                           max_grad_norm=algo_args.max_grad_norm,
                           use_linear_lr_decay=algo_args.use_linear_lr_decay,
                           use_gae=algo_args.use_gae,
                           gae_lambda=algo_args.gae_lambda,
                           use_proper_time_limits=algo_args.use_proper_time_limits,
                           obs_shape=(1 + (latent_dim * 2),),
                           latent_dim=latent_dim,
                           recurrent_policy=algo_args.recurrent,
                           hidden_size=algo_args.hidden_size,
                           use_elu=algo_args.use_elu,
                           variational_model=None,
                           vae_optim=None,
                           vae_min_seq=1,
                           vae_max_seq=algo_args.vae_max_steps,
                           max_sigma=[prior_var_max ** (1 / 2) for _ in range(latent_dim)],
                           use_decay_kld=algo_args.use_decay_kld,
                           decay_kld_rate=algo_args.decay_kld_rate,
                           env_dim=state_dim,
                           action_dim=action_dim,
                           min_sigma=[prior_var_min ** (1 / 2) for _ in range(latent_dim)],
                           use_xavier=algo_args.use_xavier,
                           use_rms_obs=algo_args.use_rms_obs,
                           use_rms_latent=algo_args.use_rms_latent,
                           use_feature_extractor=algo_args.use_feature_extractor,
                           state_extractor_dim=algo_args.state_extractor_dim,
                           latent_extractor_dim=algo_args.latent_extractor_dim,
                           uncertainty_extractor_dim=algo_args.uncertainty_extractor_dim,
                           use_huber_loss=algo_args.use_huber_loss,
                           detach_every=algo_args.detach_every,
                           use_rms_rew=algo_args.use_rms_rew,
                           decouple_rms=algo_args.decouple_rms_latent
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
                                        use_env_obs=use_env_obs,
                                        num_eval_processes=num_eval_processes,
                                        task_generator=task_generator,
                                        store_history=store_history,
                                        task_len=task_len)
    else:
        raise RuntimeError()

    return res


def main(args, model, vi, algo, store, seed, rest_args, num_signals):
    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_var ** (1 / 2),
                                                                   num_sig=num_signals)

    task_generator = MiniGolfSignalsTaskGenerator(prior_var_min=prior_var_min,
                                                  prior_var_max=prior_var_max,
                                                  num_signals=num_signals)

    return get_meta_test(algo=algo, sw_size=args.sw_size, prior_sequences=prior_sequences,
                         init_prior_sequences=init_prior, gp_list_sequences=gp_list_sequences,
                         num_eval_processes=args.num_test_processes, task_generator=task_generator,
                         store_history=store, seed=seed, log_dir=args.log_dir,
                         device=device, task_len=args.task_len, model=model, vi=vi, rest_args=rest_args,
                         num_signals=num_signals)


def run(id, seed, args, model, vi, algo, store, rest_args, num_signals):
    # Eventually fix here the seeds for additional sources of randomness (e.g. tensorflow)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Starting run {} algo {} seed {}".format(id, algo, seed))
    r = main(args=args, model=model, vi=vi, algo=algo, store=store, seed=seed, rest_args=rest_args,
             num_signals=num_signals)
    print("Done run {} algo {} seed {}".format(id, algo, seed))
    return r


# Scheduling runs: ENTRY POINT
args, rest_args = get_test_args()

if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

torch.set_num_threads(1)
device = torch.device("cuda:0" if args.cuda else "cpu")

seeds = [np.random.randint(1000000) for _ in range(100)]

r_bayes_0 = []
r_bayes_1 = []
r_bayes_2 = []
r_bayes_3 = []

for algo, f, sh, num_sig in zip(algo_list, folder_list, store_history_list, num_signals_list):
    if algo == 'bayes':
        model_list = []
        vi_list = []
        dirs_containing_res = os.listdir(f)
        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/agent_ac"))
            vi_list.append(torch.load(f + d + "/agent_vi"))
        r_bayes = (Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=vi, algo=algo, store=sh, rest_args=rest_args,
                         num_signals=num_sig)
            for id, seed, model, vi in zip(range(len(model_list)), seeds, model_list, vi_list)))
        if num_sig == 0:
            r_bayes_0 = r_bayes
        elif num_sig == 1:
            r_bayes_1 = r_bayes
        elif num_sig == 2:
            r_bayes_2 = r_bayes
        elif num_sig == 3:
            r_bayes_3 = r_bayes
        else:
            raise RuntimeError("Maximum number of signals is {}".format(3))

print("END ALL RUNS")

meta_test_res = [r_bayes_3, r_bayes_2, r_bayes_1, r_bayes_0]

# Create python plots from meta-test results
prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                               num_test_processes=args.num_test_processes,
                                                               std=noise_seq_var ** (1 / 2),
                                                               num_sig=0)

fd, folder_path_with_date = handle_folder_creation(result_path=folder)

if args.dump_data:
    with open("{}data_results.pkl".format(folder_path_with_date), "wb") as output:
        pickle.dump(meta_test_res, output)

create_csv_rewards(r_list=meta_test_res,
                   label_list=['Bayes3', 'Bayes2', 'Bayes1', 'Bayes0'],
                   has_track_list=[True, True, True, True],
                   num_seq=num_seq,
                   prior_seqs=prior_sequences,
                   seq_len_list=seq_len_list,
                   sequence_name_list=sequence_name_list,
                   folder_path_with_date=folder_path_with_date)

fd.close()
