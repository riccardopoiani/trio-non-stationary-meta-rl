import numpy as np
import torch
import os
import envs
from gym import spaces
from joblib import Parallel, delayed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, DotProduct

from configs import cheetah_ts_arguments, cheetah_bayes_arguments, cheetah_rl2_arguments
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.cheetah_vel_task_generator import CheetahVelTaskGenerator
from utilities.folder_management import handle_folder_creation
from utilities.plots.plots import view_results_multiple_dim, create_csv_rewards, create_csv_tracking
from utilities.test_arguments import get_test_args

folder = "result/metatest/cheetahvelv2/"
env_name = "cheetahvel-v2"
folder_list = ["result/cheetahvelv2/oursinitnewp/", "result/cheetahvelv2/rl2init/", "result/cheetahvelv2/tsinitsmooth/"]
algo_list = ['ours', 'rl2', 'ts_opt']
label_list = ['ours', 'rl2', 'ts_opt']
has_track_list = [True, False, True]
store_history_list = [True, False, True]

# Task family parameters
prior_var_min = 0.01
prior_var_max = 0.3
noise_seq_var = 0.00001
latent_dim = 1
state_dim = 20
action_dim = 6
use_env_obs = True
high_act = np.ones(6, dtype=np.float32)
low_act = -np.ones(6, dtype=np.float32)
action_space = spaces.Box(low=low_act, high=high_act)
prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]

num_seq = 3
seq_len_list = [15, 20, 30]
sequence_name_list = ['const_acc', 'const', 'quad_dec']


def f_linear(x, m=0.13, q=-0.9):
    return x * m + q


def f_quadratic(x):
    a = -2 / 900
    b = 0
    c = 1
    return a * (x ** 2) + b * x + c


def get_linear_deceleration(num_test_processes, n_restarts, std, seq_len):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(latent_dim):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    target_vel = f_quadratic(0)
    init_prior_test = [torch.tensor([[target_vel], [prior_var_min ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, seq_len):
        target_vel = f_quadratic(x=idx)
        prior_seq.append(torch.tensor([[target_vel], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_constant_speed(num_test_processes, n_restarts, std, seq_len):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(latent_dim):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    target_vel = 0.8

    init_prior_test = [torch.tensor([[target_vel], [prior_var_min ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, seq_len):
        prior_seq.append(torch.tensor([[target_vel], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_constant_acceleration(num_test_processes, n_restarts, std, seq_len):
    kernel = C(1) * RBF(1) + WhiteKernel(0.01, noise_level_bounds="fixed") + DotProduct(1)

    gp_list = []
    for i in range(latent_dim):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    init_prior_test = [torch.tensor([[f_linear(0)], [prior_var_min ** (1 / 2)]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(0, seq_len):
        vel = f_linear(idx)
        prior_seq.append(torch.tensor([[vel], [std ** 2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_const_acc, prior_seq_const_acc, init_prior_const_acc = get_constant_acceleration(
        num_test_processes=num_test_processes,
        std=std,
        n_restarts=n_restarts,
        seq_len=15)

    gp_list_const, prior_seq_const, init_prior_const = get_constant_speed(
        num_test_processes=num_test_processes,
        std=std,
        n_restarts=n_restarts,
        seq_len=20
    )

    gp_list_quad, prior_seq_quad, init_prior_quad = get_linear_deceleration(
        num_test_processes=num_test_processes,
        std=std,
        n_restarts=n_restarts,
        seq_len=30
    )

    # Fill lists
    p = [prior_seq_const_acc, prior_seq_const, prior_seq_quad]
    gp = [gp_list_const_acc, gp_list_const, gp_list_quad]
    ip = [init_prior_const_acc, init_prior_const, init_prior_quad]

    return p, gp, ip


def get_meta_test(algo, gp_list_sequences, sw_size, prior_sequences, init_prior_sequences,
                  num_eval_processes, task_generator, store_history, seed, log_dir,
                  device, task_len, model, vi, rest_args):
    if algo == "rl2":
        algo_args = cheetah_rl2_arguments.get_args(rest_args)
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
                    obs_shape=(27,),
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
        algo_args = cheetah_bayes_arguments.get_args(rest_args)
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
                          obs_shape=(2 * latent_dim + state_dim,),
                          latent_dim=latent_dim,
                          recurrent_policy=algo_args.recurrent,
                          hidden_size=algo_args.hidden_size,
                          use_elu=algo_args.use_elu,
                          variational_model=None,
                          vae_optim=None,
                          vae_min_seq=1,
                          vae_max_seq=algo_args.vae_max_steps,
                          max_sigma=[prior_var_max ** (1 / 2)],
                          use_decay_kld=algo_args.use_decay_kld,
                          decay_kld_rate=algo_args.decay_kld_rate,
                          env_dim=state_dim,
                          action_dim=action_dim,
                          min_sigma=[prior_var_min ** (1 / 2)],
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
    elif algo == "ts_opt":
        algo_args = cheetah_ts_arguments.get_args(rest_args)
        agent = PosteriorOptTSAgent(vi=None,
                                    vi_optim=None,
                                    num_steps=algo_args.num_steps,
                                    num_processes=algo_args.num_processes,
                                    device=device,
                                    gamma=algo_args.gamma,
                                    latent_dim=latent_dim,
                                    use_env_obs=use_env_obs,
                                    max_sigma=[prior_var_max ** (1 / 2)],
                                    min_sigma=[prior_var_min ** (1 / 2)],
                                    action_space=action_space,
                                    obs_shape=(21,),
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

    task_generator = CheetahVelTaskGenerator(prior_var_max=prior_var_max,
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
                    rescale_latent=[0.0, 1.5],
                    num_dim=latent_dim)
fd.close()