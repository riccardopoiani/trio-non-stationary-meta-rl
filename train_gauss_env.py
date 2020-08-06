import torch
import numpy as np
import pickle

import gym_sin
from gym import spaces

from active_learning.arguments import get_args
from active_learning.oracle import OracleAgent
from active_learning.posterior_multi_task import PosteriorMTAgent
from network.vae import InferenceNetwork, InferenceNetwork2
from task.GuassianTaskGenerator import GaussianTaskGenerator
from utilities.folder_management import handle_folder_creation


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
    prior_mu_min = -10
    prior_mu_max = 10
    prior_std_min = 0.1
    prior_std_max = 5

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
                                           prior_mu_min=prior_mu_min, prior_mu_max=prior_mu_max,
                                           prior_std_min=prior_std_min, prior_std_max=prior_std_max)
    task_generator.get_task_family(n_tasks=1000, n_batches=1, test_perc=0, batch_size=160)

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

    elif args.algo == "posterior_multi_task":
        max_old = [100, 50, 20, 20]
        min_old = [-100, 0, 0, 0]
        vae_min_seq = 1
        vae_max_seq = 150

        obs_shape = (4,)

        vi = InferenceNetwork(n_in=8, z_dim=latent_dim)
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

        res_eval, res_vae, = agent.train(training_iter=args.training_iter,
                                         env_name=env_name,
                                         seed=args.seed,
                                         task_generator=task_generator,
                                         eval_interval=args.eval_interval,
                                         log_dir=args.log_dir,
                                         num_task_to_eval=args.num_task_to_eval,
                                         use_env_obs=False,
                                         num_vae_steps=args.num_vae_steps,
                                         init_vae_steps=args.init_vae_steps)

        with open("{}vae.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_vae, output)
        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(res_eval, output)


if __name__ == "__main__":
    main()
