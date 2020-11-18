import numpy as np
import torch
import os
import envs
from gym import spaces
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

from configs import ant_goal_ts_arguments, ant_goal_bayes_arguments, ant_goal_rl2_arguments
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.ant_goal_task_generator import AntGoalTaskGenerator
from utilities.folder_management import handle_folder_creation
from utilities.plots.plots import view_results_multiple_dim, create_csv_rewards, create_csv_tracking, view_results_multiple_dim
from utilities.test_arguments import get_test_args

import warnings

warnings.filterwarnings(action='ignore')

folder = "result/metatest/antgoal/"
env_name = "antgoal-v0"
folder_list = ["result/antgoal/ours/", "result/antgoal/rl2/", "result/antgoal/ts/"]
algo_list = ['ours', 'rl2', 'ts_opt']
label_list = ['ours', 'rl2', 'ts_opt']
has_track_list = [True, False, True]
store_history_list = [True, False, True]

# Task family parameters
prior_var_min = 0.1
prior_var_max = 0.4
latent_dim = 2
use_env_obs = True
state_dim = 113
action_dim = 8
noise_seq_var = 0.01
high_act = np.ones(8, dtype=np.float32)
low_act = -np.ones(8, dtype=np.float32)
action_space = spaces.Box(low=low_act, high=high_act)
prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]

num_seq = 3
seq_len_list = [20, 30, 40]
sequence_name_list = ['circuit', 'constant', 'step']


def f_radius(theta, latent_dim):
    if latent_dim == 0:
        return np.cos(theta)
    else:
        return np.sin(theta)


def linear_theta(x):
    return x * 2 * np.pi / 20


def get_circuit_sequences(n_restarts, num_test_processes, var_seq):
    std = var_seq ** (1 / 2)
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)
    gp_list = []

    for _ in range(latent_dim):
        curr_dim_list = []
        for _ in range(num_test_processes):
            curr_dim_list.append(GaussianProcessRegressor(kernel=kernel,
                                                          normalize_y=False,
                                                          n_restarts_optimizer=n_restarts))
        gp_list.append(curr_dim_list)

    # Creating prior distribution
    p_mean = []
    p_var = []
    for dim in range(latent_dim):
        p_mean.append(f_radius(theta=0, latent_dim=dim))
        p_var.append(std ** 2)
    init_prior_test = [torch.tensor([p_mean, p_var], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    # Create prior sequence
    prior_seq = []
    for idx in range(20):
        p_mean = []
        p_var = []
        theta = linear_theta(idx)
        for dim in range(latent_dim):
            p_mean.append(f_radius(theta=theta, latent_dim=dim))
            p_var.append(std ** 2)

        prior_seq.append(torch.tensor([p_mean, p_var], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_constant_sequence(n_restarts, num_test_processes, var_seq):
    std = var_seq ** (1 / 2)
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)
    gp_list = []

    for _ in range(latent_dim):
        curr_dim_list = []
        for _ in range(num_test_processes):
            curr_dim_list.append(GaussianProcessRegressor(kernel=kernel,
                                                          normalize_y=False,
                                                          n_restarts_optimizer=n_restarts))
        gp_list.append(curr_dim_list)

    # Creating prior distribution
    theta_const = np.pi / 4
    p_mean = []
    p_var = []
    for dim in range(latent_dim):
        p_mean.append(f_radius(theta=theta_const, latent_dim=dim))
        p_var.append(std ** 2)
    init_prior_test = [torch.tensor([p_mean, p_var], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    # Create prior sequence
    prior_seq = []
    for idx in range(30):
        p_mean = []
        p_var = []
        for dim in range(latent_dim):
            p_mean.append(f_radius(theta=theta_const, latent_dim=dim))
            p_var.append(std ** 2)

        prior_seq.append(torch.tensor([p_mean, p_var], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_step_sequence(n_restarts, num_test_processes, var_seq):
    std = var_seq ** (1 / 2)
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)
    gp_list = []

    for _ in range(latent_dim):
        curr_dim_list = []
        for _ in range(num_test_processes):
            curr_dim_list.append(GaussianProcessRegressor(kernel=kernel,
                                                          normalize_y=False,
                                                          n_restarts_optimizer=n_restarts))
        gp_list.append(curr_dim_list)

    # Creating prior distribution
    theta = np.pi / 4
    p_mean = []
    p_var = []
    for dim in range(latent_dim):
        p_mean.append(f_radius(theta=theta, latent_dim=dim))
        p_var.append(std ** 2)
    init_prior_test = [torch.tensor([p_mean, p_var], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    # Create prior sequence
    prior_seq = []
    for idx in range(40):
        p_mean = []
        p_var = []
        theta = np.pi / 4 if idx <= 20 else -np.pi / 4
        for dim in range(latent_dim):
            p_mean.append(f_radius(theta=theta, latent_dim=dim))
            p_var.append(std ** 2)

        prior_seq.append(torch.tensor([p_mean, p_var], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_circuit, prior_seq_circuit, init_prior_circuit = get_circuit_sequences(
        num_test_processes=num_test_processes,
        n_restarts=n_restarts,
        var_seq=std ** 2
    )

    gp_list_constant, prior_seq_constant, init_prior_constant = get_constant_sequence(
        num_test_processes=num_test_processes,
        n_restarts=n_restarts,
        var_seq=std ** 2
    )

    gp_list_step, prior_seq_step, init_prior_step = get_step_sequence(
        num_test_processes=num_test_processes,
        n_restarts=n_restarts,
        var_seq=std ** 2
    )

    # Fill lists
    p = [prior_seq_circuit, prior_seq_constant, prior_seq_step]
    gp = [gp_list_circuit, gp_list_constant, gp_list_step]
    ip = [init_prior_circuit, init_prior_constant, init_prior_step]

    return p, gp, ip


def get_meta_test(algo, gp_list_sequences, sw_size, prior_sequences, init_prior_sequences,
                  num_eval_processes, task_generator, store_history, seed, log_dir,
                  device, task_len, model, vi, rest_args):
    if algo == "rl2":
        algo_args = ant_goal_rl2_arguments.get_args(rest_args)
        agent = RL2(hidden_size=algo_args.hidden_size,
                    use_elu=algo_args.use_elu,
                    clip_param=algo_args.clip_param,
                    ppo_epoch=algo_args.ppo_epoch,
                    num_mini_batch=algo_args.num_mini_batch,
                    value_loss_coef=algo_args.value_loss_coef,
                    entropy_coef=algo_args.entropy_coef,
                    lr=algo_args.ppo_lr,
                    eps=algo_args.ppo_eps,
                    max_grad_norm=algo_args.max_grad_norm,
                    action_space=action_space,
                    obs_shape=(state_dim+action_dim+1,),
                    use_obs_env=use_env_obs,
                    num_processes=algo_args.num_processes,
                    gamma=algo_args.gamma,
                    device=device,
                    num_steps=algo_args.num_steps,
                    action_dim=action_dim,
                    use_gae=algo_args.use_gae,
                    gae_lambda=algo_args.gae_lambda,
                    use_proper_time_limits=algo_args.use_proper_time_limits,
                    use_xavier=algo_args.use_xavier,
                    use_huber_loss=algo_args.use_huber_loss,
                    use_extractor=algo_args.use_feature_extractor,
                    reward_extractor_dim=algo_args.rl2_reward_emb_dim,
                    action_extractor_dim=algo_args.rl2_action_emb_dim,
                    state_extractor_dim=algo_args.rl2_state_emb_dim,
                    done_extractor_dim=algo_args.rl2_done_emb_dim,
                    use_done=algo_args.use_done,
                    use_rms_rew=algo_args.use_rms_rew,
                    use_rms_state=algo_args.use_rms_obs,
                    use_rms_act=algo_args.use_rms_act,
                    use_rms_rew_in_policy=algo_args.use_rms_rew_in_policy,
                    state_dim=state_dim,
                    latent_dim=algo_args.rl2_latent_dim
                    )
        agent.actor_critic = model
        res = agent.meta_test(prior_sequences, task_generator, num_eval_processes, env_name, seed, log_dir,
                              task_len)
    elif algo == "ours":
        algo_args = ant_goal_bayes_arguments.get_args(rest_args)
        agent = OursAgent(action_space=action_space,
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
                          obs_shape=(2*latent_dim + state_dim,),
                          latent_dim=latent_dim,
                          recurrent_policy=algo_args.recurrent,
                          hidden_size=algo_args.hidden_size,
                          use_elu=algo_args.use_elu,
                          variational_model=None,
                          vae_optim=None,
                          vae_min_seq=1,
                          vae_max_seq=algo_args.vae_max_steps,
                          max_sigma=prior_std_max,
                          use_decay_kld=algo_args.use_decay_kld,
                          decay_kld_rate=algo_args.decay_kld_rate,
                          env_dim=state_dim,
                          action_dim=action_dim,
                          min_sigma=prior_std_min,
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
                                        use_env_obs=True,
                                        num_eval_processes=num_eval_processes,
                                        task_generator=task_generator,
                                        store_history=store_history,
                                        task_len=task_len)
    elif algo == "ts_opt":
        algo_args = ant_goal_ts_arguments.get_args(rest_args)
        agent = PosteriorOptTSAgent(vi=None,
                                    vi_optim=None,
                                    num_steps=algo_args.num_steps,
                                    num_processes=algo_args.num_processes,
                                    device=device,
                                    gamma=algo_args.gamma,
                                    latent_dim=latent_dim,
                                    use_env_obs=use_env_obs,
                                    max_sigma=prior_std_max,
                                    min_sigma=prior_std_min,
                                    action_space=action_space,
                                    obs_shape=(state_dim + latent_dim,),
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
                                    recurrent_policy=algo_args.recurrent,
                                    hidden_size=algo_args.hidden_size,
                                    use_elu=algo_args.use_elu,
                                    use_decay_kld=algo_args.use_decay_kld,
                                    decay_kld_rate=algo_args.decay_kld_rate,
                                    env_dim=state_dim,
                                    action_dim=action_dim,
                                    vae_max_steps=algo_args.vae_max_steps,
                                    use_xavier=algo_args.use_xavier,
                                    use_rms_obs=algo_args.use_rms_obs,
                                    use_rms_latent=algo_args.use_rms_latent,
                                    use_feature_extractor=algo_args.use_feature_extractor,
                                    state_extractor_dim=algo_args.state_extractor_dim,
                                    latent_extractor_dim=algo_args.latent_extractor_dim,
                                    use_huber_loss=algo_args.use_huber_loss,
                                    detach_every=algo_args.detach_every,
                                    use_rms_rew=algo_args.use_rms_rew
                                    )
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


def main(args, model, vi, algo, store, seed, rest_args):
    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_var ** (1 / 2))

    task_generator = AntGoalTaskGenerator(prior_var_max=prior_var_max,
                                          prior_var_min=prior_var_min)

    return get_meta_test(algo=algo, sw_size=args.sw_size, prior_sequences=prior_sequences,
                         init_prior_sequences=init_prior, gp_list_sequences=gp_list_sequences,
                         num_eval_processes=args.num_test_processes, task_generator=task_generator,
                         store_history=store, seed=seed, log_dir=args.log_dir,
                         device=device, task_len=args.task_len, model=model, vi=vi, rest_args=rest_args)


def run(id, seed, args, model, vi, algo, store, rest_args):
    # Eventually fix here the seeds for additional sources of randomness (e.g. tensorflow)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("Starting run {} algo {} seed {}".format(id, algo, seed))
    r = main(args=args, model=model, vi=vi, algo=algo, store=store, seed=seed, rest_args=rest_args)
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
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=vi, algo=algo, store=sh, rest_args=rest_args)
            for id, seed, model, vi in zip(range(len(model_list)), seeds, model_list, vi_list)))
    elif algo == "ts_opt":
        model_list = []
        vi_list = []
        dirs_containing_res = os.listdir(f)
        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/agent_ac"))
            vi_list.append(torch.load(f + d + "/agent_vi"))
        r_ts = (Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=vi, algo=algo, store=sh, rest_args=rest_args)
            for id, seed, model, vi in zip(range(len(model_list)), seeds, model_list, vi_list)))
    elif algo == "rl2":
        model_list = []
        dirs_containing_res = os.listdir(f)
        for d in dirs_containing_res:
            model_list.append(torch.load(f + d + "/rl2_actor_critic"))
        r_rl2 = (Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(id=id, seed=seed, args=args, model=model, vi=None, algo=algo, store=sh, rest_args=rest_args)
            for id, seed, model in zip(range(len(model_list)), seeds, model_list)))

print("END ALL RUNS")

meta_test_res = [r_ours, r_ts, r_rl2]
with open("temp.pkl", "wb") as output:
    import pickle

    pickle.dump(meta_test_res, output)

# Create python plots from meta-test results
prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                               num_test_processes=args.num_test_processes,
                                                               std=noise_seq_var ** (1 / 2))

fd, folder_path_with_date = handle_folder_creation(result_path=folder)

if args.dump_data:
    with open("{}data_results.pkl".format(folder_path_with_date), "wb") as output:
        pickle.dump(meta_test_res, output)

create_csv_rewards(r_list=meta_test_res,
                   label_list=['Ours', 'TS', 'RL'],
                   has_track_list=[True, True, False],
                   num_seq=num_seq,
                   prior_seqs=prior_sequences,
                   seq_len_list=seq_len_list,
                   sequence_name_list=sequence_name_list,
                   folder_path_with_date=folder_path_with_date)

create_csv_tracking(r_list=meta_test_res,
                    label_list=['Ours', 'TS', 'RL2'],
                    has_track_list=[True, True, False],
                    num_seq=num_seq,
                    prior_seqs=prior_sequences,
                    seq_len_list=seq_len_list,
                    sequence_name_list=sequence_name_list,
                    folder_path_with_date=folder_path_with_date,
                    init_priors=init_prior,
                    rescale_latent=[-3.0, 3.0],
                    num_dim=latent_dim)

fd.close()
