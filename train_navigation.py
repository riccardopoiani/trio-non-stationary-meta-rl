import os
import pickle

import numpy as np
import torch
import custom_env
from gym import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from inference.inference_network import EmbeddingInferenceNetwork
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.navigation_task_generator import NavigationTaskGenerator
from utilities.arguments import get_args
from utilities.folder_management import handle_folder_creation

os.environ["OPENBLAS_NUM_THREADS"] = "1"


def get_alternating_sequences(n_restarts, num_test_processes, var_seq, num_signals):
    std = var_seq ** (1 / 2)
    # Creating GPs
    kernel = C(1.0, (1e-8, 1e8)) * RBF(1, (1e-8, 1e8))
    gp_list = []

    for _ in range(1 + 2 * num_signals):
        curr_dim_list = []
        for _ in range(num_test_processes):
            curr_dim_list.append(GaussianProcessRegressor(kernel=kernel,
                                                          normalize_y=False,
                                                          n_restarts_optimizer=n_restarts)
                                 )
        gp_list.append(curr_dim_list)

    # Creating prior distribution
    p_min = [-1]
    p_var = [std ** 2]
    for _ in range(2 * num_signals):
        p_min.append(0)
        p_var.append(std ** 2)
    init_prior_test = [torch.tensor([p_min, p_var], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    # Create prior sequence
    prior_seq = []
    for idx in range(10):
        p_min = []
        p_var = []
        if idx <= 20:
            p_min.append(-1 + idx / 10)
            p_var.append(std ** 2)
        elif 20 < idx < 30:
            p_min.append(1)
            p_var.append(std ** 2)
        else:
            p_min.append(1 - (idx - 30) / 10)
            p_var.append(std ** 2)

        for _ in range(2 * num_signals):
            if idx <= 20:
                p_min.append(1 - idx / 10)
            else:
                p_min.append(-1)
            p_var.append(std ** 2)

        prior_seq.append(torch.tensor([p_min, p_var], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, var_seq, num_signals):
    gp_list, prior_seq, init_prior_test = get_alternating_sequences(n_restarts=n_restarts,
                                                                    num_test_processes=num_test_processes,
                                                                    var_seq=var_seq,
                                                                    num_signals=num_signals)
    # return [prior_seq], [gp_list], [init_prior_test]
    return [], [], []


def main():
    print("Starting...")
    args = get_args()

    # Task family settings
    folder = "result/navigationv0/"
    env_name = "goalnavigation-v0"

    # Task family parameters
    prior_goal_var_min = 0.0001
    prior_goal_var_max = 0.2
    prior_signal_var_min = 0.01
    prior_signal_var_max = 0.4
    signals_dim = args.num_signals
    latent_dim = 1 + signals_dim * 2  # (x,y) + (bal_x, bal_y) + num_sig * (x,y)

    high_act = np.array([
        1,
        1,
    ], dtype=np.float32)

    low_act = np.array([
        -1,
        -1
    ], dtype=np.float32)

    action_space = spaces.Box(low=low_act, high=high_act)

    # Other settings
    var_seq = 0.00001

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    task_generator = NavigationTaskGenerator(prior_goal_var_min=prior_goal_var_min,
                                             prior_goal_var_max=prior_goal_var_max,
                                             prior_signal_var_min=prior_signal_var_min,
                                             prior_signal_var_max=prior_signal_var_max,
                                             signals_dim=signals_dim)

    prior_std_max = task_generator.latent_max_std.tolist()
    prior_std_min = task_generator.latent_min_std.tolist()

    if len(args.folder) == 0:
        folder = folder + args.algo + "/"
    else:
        folder = folder + args.folder + "/"
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   var_seq=var_seq,
                                                                   num_signals=signals_dim)

    print("Algorithm start..")
    if args.algo == 'rl2':
        obs_shape = (2 + 1 + 2 + signals_dim,)  # obs_shape + action_shape + 1 reward

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
                    use_proper_time_limits=args.use_proper_time_limits,
                    use_xavier=args.use_xavier)

        eval_list, test_list = agent.train(n_iter=args.training_iter,
                                           env_name=env_name,
                                           seed=args.seed,
                                           task_generator=task_generator,
                                           eval_interval=args.eval_interval,
                                           log_dir=args.log_dir,
                                           num_test_processes=args.num_test_processes,
                                           verbose=args.verbose,
                                           num_random_task_to_eval=args.num_random_task_to_eval,
                                           prior_task_sequences=prior_sequences,
                                           task_len=args.task_len)

        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(eval_list, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)

        torch.save(agent.actor_critic, "{}rl2_actor_critic".format(folder_path_with_date))
    elif args.algo == 'ts_opt':
        max_old = None
        min_old = None
        obs_shape = (latent_dim + 2 + signals_dim,)  # latent dim + obs

        # 2 action + 2 obs + 1 reward + prior (latent dim * 2)
        vi = EmbeddingInferenceNetwork(n_in=2 + 2 + 1 + signals_dim + latent_dim * 2, z_dim=latent_dim)
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
                                    env_dim=2 + signals_dim,
                                    action_dim=2,
                                    min_sigma=prior_std_min,
                                    vae_max_steps=args.vae_max_steps,
                                    use_xavier=args.use_xavier)

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
                                                                vae_smart=args.vae_smart,
                                                                task_len=args.task_len)
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

        # 2 * latent_dim + obs
        obs_shape = (2 * latent_dim + 2 + signals_dim,)

        # 2 action + 2 obs + 1 reward + 4 prior (latent dim * 2)
        vi = EmbeddingInferenceNetwork(n_in=obs_shape[0] + 1 + 2, z_dim=latent_dim)
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
                          vae_max_seq=args.vae_max_steps,
                          max_action=None,
                          min_action=None,
                          max_sigma=prior_std_max,
                          use_decay_kld=args.use_decay_kld,
                          decay_kld_rate=args.decay_kld_rate,
                          env_dim=2 + signals_dim,
                          action_dim=2,
                          min_sigma=prior_std_min,
                          use_xavier=args.use_xavier
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
                                                               vae_smart=args.vae_smart,
                                                               task_len=args.task_len)

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
