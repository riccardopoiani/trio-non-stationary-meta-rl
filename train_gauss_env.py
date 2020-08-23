import pickle

import numpy as np
import torch
from gym import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from active_learning.PosteriorTS import PosteriorTSAgent
from active_learning.arguments import get_args
from active_learning.gp_ts import GaussianProcessThompsonSampling
from active_learning.posterior_multi_task import PosteriorMTAgent
from active_learning.recurrent import RL2
from network.vae import InferenceNetworkNoPrev
from task.GuassianTaskGenerator import GaussianTaskGenerator
from utilities.folder_management import handle_folder_creation


def get_task_sequence(alpha, n_restarts, num_test_processes):
    kernel = C(1.0, (1e-5, 1e5)) * RBF(1, (1e-5, 1e5))

    gp_list = []
    for i in range(2):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 alpha=alpha ** 2,
                                                 normalize_y=True,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])
    test_kwargs = []
    init_prior_test = [torch.tensor([[15, 30], [4, 4]], dtype=torch.float32) for _ in range(num_test_processes)]

    for idx in range(50):
        if idx < 15:
            mean = 15
            std = 30
        elif idx > 40:
            mean = 40
            std = 30
        else:
            mean = 15 + idx
            std = 30 - idx / 16

        test_kwargs.append({'amplitude': 1,
                            'mean': mean,
                            'std': std,
                            'noise_std': 0.001,
                            'scale_reward': False})

    return gp_list, test_kwargs, init_prior_test


def get_sequences():
    return [], [], []


def main():
    # Task family settings
    folder = "result/gaussv0/"
    env_name = "gauss-v0"
    action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
    latent_dim = 1
    x_min = -100
    x_max = 100
    min_mean = -70
    max_mean = 70
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

    task_generator = GaussianTaskGenerator(x_min, x_max, min_mean, max_mean,
                                           prior_mu_min, prior_mu_max, prior_std_min, prior_std_max, std, amplitude)
    task_generator.create_task_family(n_tasks=n_tasks, n_batches=1, test_perc=0, batch_size=1)

    folder = folder + args.algo
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    prior_sequences, gp_list_sequences, init_prior = get_sequences()

    if args.algo == 'rl2':
        obs_shape = (2,)

        agent = RL2(hidden_size=args.hidden_size,
                    use_elu=args.use_elu,
                    clip_param=args.clip_param,
                    ppo_epoch=args.ppo_epoch,
                    num_mini_batch=args.num_mini_batch,
                    value_loss_coef=args.value_loss_coef,
                    entropy_coef=args.entropy_coef,
                    lr=args.lr,
                    eps=args.eps,
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
        max_old = [100, 50, 20, 20]
        min_old = [-100, 0, 0, 0]
        vae_min_seq = 1
        vae_max_seq = 150

        obs_shape = (4,)

        vi = InferenceNetworkNoPrev(n_in=4, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters())

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
                                                   log_dir=args.lod_dir,
                                                   use_env_obs=False,
                                                   num_vae_steps=args.num_vae_steps,
                                                   init_vae_steps=args.init_vae_steps,
                                                   sw_size=args.sw_size,
                                                   num_random_task_to_eval=args.num_random_task_to_eval,
                                                   num_test_processes=args.num_test_processes,
                                                   use_true_sigma=args.use_true_sigma,
                                                   use_data_loader=False,
                                                   gp_list_sequences=gp_list_sequences,
                                                   prior_sequences=prior_sequences,
                                                   init_prior_test_sequences=init_prior,
                                                   verbose=args.verbose,
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
        vi = InferenceNetworkNoPrev(n_in=4, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters())

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
                                                    num_eval_processes=args.num_eval_processes,
                                                    use_true_sigma=args.use_true_sigma)

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
