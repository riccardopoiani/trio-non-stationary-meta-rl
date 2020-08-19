import torch
import numpy as np
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, ConstantKernel as C

import gym_sin
from gym import spaces

from active_learning.FixedIdentification import FixedIDAgent
from active_learning.arguments import get_args
from active_learning.new_pt_multi_task import DecoupledPMTAgent
from active_learning.oracle import OracleAgent
from active_learning.posterior_multi_task import PosteriorMTAgent
from network.vae import InferenceNetwork, InferenceNetwork2, InferenceNetworkNoPrev
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


def main():
    # Task family settings
    folder = "result/gaussv0/"
    env_name = "gauss-v0"
    action_space = spaces.Box(low=np.array([-1]), high=np.array([1]))
    latent_dim = 2
    x_min = -100
    x_max = 100
    min_mean = -40
    max_mean = 40
    min_std = 15
    max_std = 35
    prior_mean_mu_min = -10
    prior_mean_mu_max = 10
    prior_mean_std_min = 3
    prior_mean_std_max = 5

    prior_std_mu_min = -10
    prior_std_mu_max = 10
    prior_std_std_min = 3
    prior_std_std_max = 5

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    task_generator = GaussianTaskGenerator(x_min=x_min, x_max=x_max, min_mean=min_mean,
                                           max_mean=max_mean, min_std=min_std, max_std=max_std,
                                           prior_mean_mu_min=prior_mean_mu_min, prior_mean_mu_max=prior_mean_mu_max,
                                           prior_mean_std_min=prior_mean_std_min, prior_mean_std_max=prior_mean_std_max,
                                           prior_std_mu_min=prior_std_mu_min, prior_std_mu_max=prior_std_mu_max,
                                           prior_std_std_min=prior_std_std_min, prior_std_std_max=prior_std_std_max)
    task_generator.create_task_family(n_tasks=5000, n_batches=1, test_perc=0, batch_size=160)

    folder = folder + args.algo
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    if args.algo == 'oracle':
        obs_shape = (2,)

        agent = OracleAgent(action_space=action_space,
                            device=device,
                            gamma=args.gamma,
                            num_steps=args.num_steps,
                            num_processes=args.num_processes,
                            clip_param=args.clip_param,
                            ppo_epoch=args.ppo_epoch,
                            num_mini_batch=args.num_mini_batch,
                            value_loss_coef=args.value_loss_coef,
                            entropy_coef=args.entropy_coef,
                            lr=args.ppo_lr,
                            eps=args.ppo_eps,
                            max_grad_norm=args.max_grad_norm,
                            use_linear_lr_decay=args.use_linear_lr_decay,
                            use_gae=args.use_gae,
                            gae_lambda=args.gae_lambda,
                            use_proper_time_limits=args.use_proper_time_limits,
                            obs_shape=obs_shape,
                            latent_dim=latent_dim,
                            recurrent_policy=args.recurrent,
                            hidden_size=args.hidden_size,
                            use_elu=args.use_elu)

        result = agent.train(training_iter=args.training_iter,
                             env_name=env_name,
                             seed=args.seed,
                             task_generator=task_generator,
                             num_update_per_meta_training_iter=args.num_update_per_meta_training_iter,
                             eval_interval=args.eval_interval,
                             log_dir=args.log_dir,
                             num_task_to_eval=args.num_task_to_eval,
                             use_env_obs=False)

        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(result, output)
    elif args.algo == "fixed_id":
        max_old = [100, 50, 20, 20]
        min_old = [-100, 0, 0, 0]
        vae_min_seq = 1
        vae_max_seq = 150

        obs_shape_opt = (2,)
        obs_shape_id = (8,)

        vi = InferenceNetworkNoPrev(n_in=6, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters())

        agent = FixedIDAgent(action_space=action_space, device=device, gamma=args.gamma,
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
                             obs_shape_opt=obs_shape_opt,
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
                             max_action=100,
                             min_action=-100,
                             obs_shape_id=obs_shape_id,
                             hidden_size_id=args.hidden_size_id,
                             use_elu_id=args.use_elu_id,
                             clip_param_id=args.clip_param_id,
                             ppo_epoch_id=args.ppo_epoch_id,
                             value_loss_coef_id=args.value_loss_coef_id,
                             lr_id=args.lr_id,
                             eps_id=args.eps_id,
                             max_grad_norm_id=args.max_grad_norm_id,
                             num_steps_id=args.num_step_id,
                             gamma_identification=args.gamma_id,
                             recurrent_policy_id=args.recurrent_id,
                             num_mini_batch_id=args.num_mini_batch_id,
                             entropy_coef_id=args.entropy_coef_id)

        gp_list, test_kwargs, init_prior_test = get_task_sequence(alpha=args.alpha_gp,
                                                                  n_restarts=args.n_restarts_gp,
                                                                  num_test_processes=args.num_test_processes)

        agent.train(training_iter_id=args.training_iter_id,
                    training_iter_optimal=args.training_iter_opt,
                    env_name=env_name,
                    seed=args.seed,
                    task_generator=task_generator,
                    eval_interval=args.eval_interval,
                    num_random_task_to_eval=args.num_random_task_to_eval,
                    num_vae_steps=args.num_vae_steps,
                    num_test_processes=args.num_test_processes,
                    max_id_iteration=args.max_id_iteration,
                    gp_list=gp_list,
                    sw_size=args.sw_gp,
                    test_kwargs=test_kwargs,
                    init_prior_test=init_prior_test,
                    log_dir=args.log_dir,
                    use_env_obs=False)

    elif args.algo == "new_posterior_multi_task":
        max_old = [100, 50, 20, 20]
        min_old = [-100, 0, 0, 0]
        vae_min_seq = 1
        vae_max_seq = 150

        obs_shape = (2,)

        vi = InferenceNetworkNoPrev(n_in=6, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters())

        agent = DecoupledPMTAgent(action_space=action_space, device=device, gamma=args.gamma,
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
                                  max_action=100,
                                  min_action=-100)
        gp_list, test_kwargs, init_prior_test = get_task_sequence(alpha=args.alpha_gp,
                                                                  n_restarts=args.n_restarts_gp,
                                                                  num_test_processes=args.num_test_processes)

        res_eval, res_vae, test_list = agent.train(training_iter=args.training_iter,
                                                   env_name=env_name,
                                                   seed=args.seed,
                                                   task_generator=task_generator,
                                                   eval_interval=args.eval_interval,
                                                   log_dir=args.log_dir,
                                                   use_env_obs=False,
                                                   num_vae_steps=args.num_vae_steps,
                                                   gp_list=gp_list,
                                                   sw_size=args.sw_gp,
                                                   test_kwargs=test_kwargs,
                                                   init_prior_test=init_prior_test,
                                                   num_random_task_to_eval=args.num_random_task_to_eval,
                                                   num_test_processes=args.num_test_processes)

        with open("{}vae.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_vae, output)
        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_eval, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)
    elif args.algo == "posterior_multi_task":
        max_old = [100, 50, 20, 20]
        min_old = [-100, 0, 0, 0]
        vae_min_seq = 1
        vae_max_seq = 150

        obs_shape = (4,)

        vi = InferenceNetworkNoPrev(n_in=6, z_dim=latent_dim)
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
                                 max_action=100,
                                 min_action=-100)

        gp_list, test_kwargs, init_prior_test = get_task_sequence(alpha=args.alpha_gp,
                                                                  n_restarts=args.n_restarts_gp,
                                                                  num_test_processes=args.num_test_processes)

        res_eval, res_vae, test_list = agent.train(training_iter=args.training_iter,
                                                   env_name=env_name,
                                                   seed=args.seed,
                                                   task_generator=task_generator,
                                                   eval_interval=args.eval_interval,
                                                   log_dir=args.log_dir,
                                                   use_env_obs=False,
                                                   num_vae_steps=args.num_vae_steps,
                                                   init_vae_steps=args.init_vae_steps,
                                                   gp_list=gp_list,
                                                   sw_size=args.sw_gp,
                                                   test_kwargs=test_kwargs,
                                                   init_prior_test=init_prior_test,
                                                   num_random_task_to_eval=args.num_random_task_to_eval,
                                                   num_test_processes=args.num_test_processes)

        with open("{}vae.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_vae, output)
        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_eval, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)


if __name__ == "__main__":
    main()
