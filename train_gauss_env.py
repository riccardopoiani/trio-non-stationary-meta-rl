import pickle

import numpy as np
import torch
import gym_sin
from gym import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from learner.posterior_thompson_sampling import PosteriorTSAgent
from utilities.arguments import get_args
from learner.gp_ts import GaussianProcessThompsonSampling
from learner.posterior_multi_task import PosteriorMTAgent
from learner.recurrent import RL2
from inference.inference_network import InferenceNetwork
from task.GuassianTaskGenerator import GaussianTaskGenerator
from utilities.folder_management import handle_folder_creation


def get_const_task_sequence(alpha, n_restarts, num_test_processes, std):
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 alpha=alpha ** 2,
                                                 normalize_y=True,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])
    init_prior_test = [torch.tensor([[-10], [5]], dtype=torch.float32) for _ in range(num_test_processes)]

    mean = -5

    prior_seq = []
    for idx in range(15):
        prior_seq.append(torch.tensor([[mean], [std]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_linear_task_sequence(alpha, n_restarts, num_test_processes, std):
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 alpha=alpha ** 2,
                                                 normalize_y=True,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])
    init_prior_test = [torch.tensor([[30], [5]], dtype=torch.float32) for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(20):
        mean = 30 - idx

        prior_seq.append(torch.tensor([[mean], [std]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_phase_task_sequence(alpha, n_restarts, num_test_processes, std):
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 alpha=alpha ** 2,
                                                 normalize_y=True,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])
    init_prior_test = [torch.tensor([[-5], [5]], dtype=torch.float32) for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(40):
        if idx < 15:
            mean = 0
        elif idx < 30:
            mean = 10
        else:
            mean = 0

        prior_seq.append(torch.tensor([[mean], [std]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_abrupt_and_smooth(alpha, n_restarts, num_test_processes, std):
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 alpha=alpha ** 2,
                                                 normalize_y=True,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])
    init_prior_test = [torch.tensor([[-30], [5]], dtype=torch.float32) for _ in range(num_test_processes)]
    prior_seq = []

    for idx in range(60):
        if idx < 15:
            mean = -30
        elif idx < 50:
            mean = -20 + (idx - 15)
        else:
            mean = -20 + 50 - 15

        prior_seq.append(torch.tensor([[mean], [std]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(alpha, n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_const, prior_seq_const, init_prior_const = get_const_task_sequence(alpha, n_restarts,
                                                                               num_test_processes, std)
    gp_list_linear, prior_seq_linear, init_prior_linear = get_const_task_sequence(alpha, n_restarts,
                                                                                  num_test_processes,
                                                                                  std)
    gp_list_phase, prior_seq_phase, init_prior_phase = get_phase_task_sequence(alpha, n_restarts, num_test_processes,
                                                                               std)
    gp_list_both, prior_seq_both, init_prior_both = get_abrupt_and_smooth(alpha, n_restarts, num_test_processes, std)

    # Fill lists
    prior_sequences = [prior_seq_const, prior_seq_linear, prior_seq_phase, prior_seq_both]
    gp_list_sequences = [gp_list_const, gp_list_linear, gp_list_phase, gp_list_both]
    init_prior = [init_prior_const, init_prior_linear, init_prior_phase, init_prior_both]
    # gp_list_sequences = [gp_list_const]
    # prior_sequences = [prior_seq_const]
    # init_prior = [init_prior_const]

    return prior_sequences, gp_list_sequences, init_prior


def main():
    # Task family settings
    folder = "result/gaussv0/"
    env_name = "gauss-v0"
    action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
    latent_dim = 1
    x_min = -100
    x_max = 100
    min_mean = -60
    max_mean = 60
    prior_mu_min = -10
    prior_mu_max = 10
    prior_std_min = 1
    prior_std_max = 10
    n_tasks = 10000
    std = 5
    amplitude = 1

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    task_generator = GaussianTaskGenerator(x_min=x_min, x_max=x_max, min_mean=min_mean, max_mean=max_mean,
                                           prior_mu_min=prior_mu_min,
                                           prior_mu_max=prior_mu_max,
                                           prior_std_min=prior_std_min,
                                           prior_std_max=prior_std_max,
                                           std=std,
                                           amplitude=amplitude)
    task_generator.create_task_family(n_tasks=n_tasks, n_batches=1, test_perc=0,
                                      batch_size=150 if args.use_data_loader else 1)

    if len(args.folder) == 0:
        folder = folder + args.algo + "/"
    else:
        folder = folder + args.folder + "/"
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    prior_sequences, gp_list_sequences, init_prior = get_sequences(alpha=args.alpha_gp,
                                                                   n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=std)

    if args.algo == 'rl2':
        obs_shape = (2,)

        agent = RL2(hidden_size=args.hidden_size,
                    use_elu=args.use_elu,
                    clip_param=args.clip_param,
                    ppo_epoch=args.ppo_epoch,
                    num_mini_batch=args.num_mini_batch,
                    value_loss_coef=args.value_loss_coef,
                    entropy_coef=args.entropy_coef,
                    lr=args.ppo_lr,
                    eps=args.ppo_eps,
                    max_grad_norm=args.max_grad_norm,
                    action_space=action_space,
                    obs_shape=obs_shape,
                    use_obs_env=False,
                    num_processes=args.num_processes,
                    gamma=args.gamma,
                    device=device,
                    num_steps=args.num_steps,
                    action_dim=1,
                    use_gae=args.use_gae,
                    gae_lambda=args.gae_lambda,
                    use_proper_time_limits=args.use_proper_time_limits)

        eval_list, test_list = agent.train(n_iter=args.training_iter,
                                           env_name=env_name,
                                           seed=args.seed,
                                           task_generator=task_generator,
                                           eval_interval=args.eval_interval,
                                           log_dir=args.log_dir,
                                           num_test_processes=args.num_test_processes,
                                           verbose=args.verbose,
                                           num_random_task_to_eval=args.num_random_task_to_eval,
                                           prior_task_sequences=prior_sequences)

        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(eval_list, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)

        torch.save(agent.actor_critic, "{}rl2_actor_critic".format(folder_path_with_date))
    elif args.algo == "ours":
        max_old = [100, 10]
        min_old = [-100, 0]
        vae_min_seq = 1
        vae_max_seq = 150

        obs_shape = (2,)

        vi = InferenceNetwork(n_in=4, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = PosteriorMTAgent(action_space=action_space, device=device, gamma=args.gamma,
                                 num_steps=args.num_steps, num_processes=args.num_processes,
                                 clip_param=args.clip_param, ppo_epoch=args.ppo_epoch,
                                 num_mini_batch=args.num_mini_batch,
                                 value_loss_coef=args.value_loss_coef,
                                 entropy_coef=args.entropy_coef,
                                 lr=args.ppo_lr,
                                 eps=args.ppo_eps, max_grad_norm=args.max_grad_norm,
                                 use_linear_lr_decay=args.use_linear_lr_decay,
                                 use_gae=args.use_gae,
                                 gae_lambda=args.gae_lambda,
                                 use_proper_time_limits=args.use_proper_time_limits,
                                 obs_shape=obs_shape,
                                 latent_dim=latent_dim,
                                 recurrent_policy=args.recurrent,
                                 hidden_size=args.hidden_size,
                                 use_elu=args.use_elu,
                                 variational_model=vi,
                                 vae_optim=vi_optim,
                                 rescale_obs=args.rescale_obs,
                                 max_old=max_old,
                                 min_old=min_old,
                                 vae_min_seq=vae_min_seq,
                                 vae_max_seq=vae_max_seq,
                                 max_action=x_max,
                                 min_action=x_min,
                                 use_time=False,
                                 rescale_time=None,
                                 max_time=None,
                                 max_sigma=prior_std_max)

        res_eval, res_vae, test_list = agent.train(training_iter=args.training_iter,
                                                   env_name=env_name,
                                                   seed=args.seed,
                                                   task_generator=task_generator,
                                                   eval_interval=args.eval_interval,
                                                   log_dir=args.log_dir,
                                                   use_env_obs=False,
                                                   num_vae_steps=args.num_vae_steps,
                                                   init_vae_steps=args.init_vae_steps,
                                                   sw_size=args.sw_size,
                                                   num_random_task_to_eval=args.num_random_task_to_eval,
                                                   num_test_processes=args.num_test_processes,
                                                   use_data_loader=args.use_data_loader,
                                                   gp_list_sequences=gp_list_sequences,
                                                   prior_sequences=prior_sequences,
                                                   init_prior_test_sequences=init_prior,
                                                   verbose=args.verbose,
                                                   vae_smart=args.vae_smart
                                                   )

        with open("{}vae.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_vae, output)
        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_eval, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)

        torch.save(agent.vae, "{}agent_vi".format(folder_path_with_date))
        torch.save(agent.actor_critic, "{}agent_ac".format(folder_path_with_date))

    elif args.algo == "gp_ts":
        x_space = np.linspace(x_min, x_max, 10000)
        agent = GaussianProcessThompsonSampling(arms=x_space, alpha=0.25, n_restart_opt=10, init_std_dev=1e2,
                                                normalized=True)
        r = agent.meta_test(task_generator=task_generator,
                            prior_sequences=prior_sequences,
                            env_name=env_name,
                            verbose=args.verbose)
        with open("{}test".format(folder_path_with_date), "wb") as output:
            pickle.dump(r, output)
    elif args.algo == "ts_posterior":
        vi = InferenceNetwork(n_in=4, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = PosteriorTSAgent(vi=vi,
                                 vi_optim=vi_optim,
                                 num_steps=args.num_steps,
                                 num_processes=args.num_processes,
                                 device=device,
                                 gamma=args.gamma,
                                 latent_dim=latent_dim,
                                 use_env_obs=False,
                                 min_action=x_min,
                                 max_action=x_max,
                                 max_sigma=prior_std_max)
        vi_loss, eval_list, test_list = agent.train(n_train_iter=args.training_iter,
                                                    eval_interval=args.eval_interval,
                                                    task_generator=task_generator,
                                                    env_name=env_name,
                                                    seed=args.seed,
                                                    log_dir=args.log_dir,
                                                    verbose=args.verbose,
                                                    num_random_task_to_evaluate=args.num_random_task_to_eval,
                                                    gp_list_sequences=gp_list_sequences,
                                                    sw_size=args.sw_size,
                                                    prior_sequences=prior_sequences,
                                                    init_prior_sequences=init_prior,
                                                    num_eval_processes=args.num_test_processes,
                                                    use_true_sigma=args.use_true_sigma,
                                                    use_data_loader=args.use_data_loader,
                                                    vae_smart=args.vae_smart)

        with open("{}vi_loss".format(folder_path_with_date), "wb") as output:
            pickle.dump(vi_loss, output)
        with open("{}eval".format(folder_path_with_date), "wb") as output:
            pickle.dump(eval_list, output)
        with open("{}test".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)

        torch.save(agent.vi, "{}vi".format(folder_path_with_date))

    else:
        raise NotImplementedError("Agent {} not available".format(args.algo))


if __name__ == "__main__":
    main()
