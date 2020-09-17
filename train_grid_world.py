import pickle

import torch
import numpy as np
import gym_sin
from gym import spaces

from inference.inference_network import InferenceNetwork
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.gridworld_task_generator import GridWorldTaskGenerator
from utilities.arguments import get_args
from utilities.folder_management import handle_folder_creation


def get_sequences(alpha, n_restarts, num_test_processes, std):
    return [], [], []


def main():
    print("Starting...")
    # Task family settings
    folder = "result/gridwolrdv0/"
    env_name = "gridworld-v0"

    # Task family parameters
    size = 8
    goal_radius = 1
    charge_forward_min = 0.5
    charge_forward_max = 1
    charge_rotation_min = 0.2
    charge_rotation_max = 1
    prior_goal_std_min = 0.2
    prior_goal_std_max = 2
    prior_charge_std_min = 0.01
    prior_charge_std_max = 0.1
    latent_dim = 5

    high_act = np.array([
        1,
        np.pi / 2,
    ], dtype=np.float32)

    low_act = np.array([
        0,
        -np.pi / 2
    ], dtype=np.float32)

    action_space = spaces.Box(low=low_act, high=high_act)

    # Other settings
    noise_seq_std = 0.001

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    task_generator = GridWorldTaskGenerator(size=size,
                                            goal_radius=goal_radius,
                                            charge_forward_min=charge_forward_min,
                                            charge_forward_max=charge_forward_max,
                                            charge_rotation_min=charge_rotation_min,
                                            charge_rotation_max=charge_rotation_max,
                                            prior_goal_std_min=prior_goal_std_min,
                                            prior_goal_std_max=prior_goal_std_max,
                                            prior_charge_std_min=prior_charge_std_min,
                                            prior_charge_std_max=prior_charge_std_max)
    prior_std_max = task_generator.latent_max_std.tolist()

    if len(args.folder) == 0:
        folder = folder + args.algo + "/"
    else:
        folder = folder + args.folder + "/"
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    prior_sequences, gp_list_sequences, init_prior = get_sequences(alpha=args.alpha_gp,
                                                                   n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_std)

    print("Algorithm start..")
    if args.algo == 'rl2':
        obs_shape = (6,)  # 3 obs_shape + 2 action_shape + 1 reward

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
                    use_obs_env=True,
                    num_processes=args.num_processes,
                    gamma=args.gamma,
                    device=device,
                    num_steps=args.num_steps,
                    action_dim=2,
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
    elif args.algo == 'ts_opt':
        max_old = None
        min_old = None
        obs_shape = (8,) # latent dim + obs

        vi = InferenceNetwork(n_in=16, z_dim=latent_dim) # 2 action + 3 obs + 1 reward + 10 prior (latent dim * 2)
        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = PosteriorOptTSAgent(vi=vi,
                                    vi_optim=vi_optim,
                                    num_steps=args.num_steps,
                                    num_processes=args.num_processes,
                                    device=device,
                                    gamma=args.gamma,
                                    latent_dim=latent_dim,
                                    use_env_obs=True,
                                    min_action=None,
                                    max_action=None,
                                    max_sigma=prior_std_max,
                                    action_space=action_space,
                                    obs_shape=obs_shape,
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
                                    use_proper_time_limits=False,
                                    recurrent_policy=args.recurrent,
                                    hidden_size=args.hidden_size,
                                    use_elu=args.use_elu,
                                    rescale_obs=False,
                                    max_old=max_old,
                                    min_old=min_old,
                                    use_decay_kld=args.use_decay_kld,
                                    decay_kld_rate=args.decay_kld_rate,
                                    env_dim=3,
                                    action_dim=2)

        vi_loss, eval_list, test_list, final_test = agent.train(n_train_iter=args.training_iter,
                                                                init_vae_steps=args.init_vae_steps,
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
                                                                vae_smart=args.vae_smart)
        with open("{}vae.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(vi_loss, output)
        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(eval_list, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)
        with open("{}final_test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(final_test, output)

        torch.save(agent.vi, "{}agent_vi".format(folder_path_with_date))
        torch.save(agent.actor_critic, "{}agent_ac".format(folder_path_with_date))

    elif args.algo == "ours":
        max_old = None
        min_old = None
        vae_min_seq = 1
        vae_max_seq = args.num_steps

        obs_shape = (13,)

        vi = InferenceNetwork(n_in=16, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = OursAgent(action_space=action_space, device=device, gamma=args.gamma,
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
                          rescale_obs=False,
                          max_old=max_old,
                          min_old=min_old,
                          vae_min_seq=vae_min_seq,
                          vae_max_seq=vae_max_seq,
                          max_action=None,
                          min_action=None,
                          max_sigma=prior_std_max,
                          use_decay_kld=args.use_decay_kld,
                          decay_kld_rate=args.decay_kld_rate,
                          env_dim=3,
                          action_dim=2
                          )

        res_eval, res_vae, test_list, final_test = agent.train(training_iter=args.training_iter,
                                                               env_name=env_name,
                                                               seed=args.seed,
                                                               task_generator=task_generator,
                                                               eval_interval=args.eval_interval,
                                                               log_dir=args.log_dir,
                                                               use_env_obs=True,
                                                               init_vae_steps=args.init_vae_steps,
                                                               sw_size=args.sw_size,
                                                               num_random_task_to_eval=args.num_random_task_to_eval,
                                                               num_test_processes=args.num_test_processes,
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
        with open("{}final_test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(final_test, output)

        torch.save(agent.vae, "{}agent_vi".format(folder_path_with_date))
        torch.save(agent.actor_critic, "{}agent_ac".format(folder_path_with_date))
    else:
        raise NotImplementedError("Agent {} not available".format(args.algo))


if __name__ == "__main__":
    main()