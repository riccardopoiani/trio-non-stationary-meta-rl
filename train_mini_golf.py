import pickle

import torch
import numpy as np
import gym_sin
from gym import spaces
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

from inference.inference_network import InferenceNetwork
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.mini_golf_task_generator import MiniGolfTaskGenerator
from utilities.arguments import get_args
from utilities.folder_management import handle_folder_creation


def get_sin_task_sequence(n_restarts, num_test_processes, std):
    kernel = C(1.0, (1e-8, 1e8)) * RBF(1, (1e-8, 1e8))

    gp_list = []
    for i in range(num_test_processes):
        gp_list.append([GaussianProcessRegressor(kernel=kernel,
                                                 normalize_y=False,
                                                 n_restarts_optimizer=n_restarts)
                        for _ in range(num_test_processes)])

    f = 0.05
    offset = -0.7
    a = -0.2

    init_prior_test = [torch.tensor([[a * np.sin(0) + offset], [std**2]], dtype=torch.float32)
                       for _ in range(num_test_processes)]

    prior_seq = []
    for idx in range(300):
        friction = a * np.sin(f * idx) + offset
        prior_seq.append(torch.tensor([[friction], [std**2]], dtype=torch.float32))

    return gp_list, prior_seq, init_prior_test


def get_sequences(n_restarts, num_test_processes, std):
    # Retrieve task
    gp_list_sin, prior_seq_sin, init_prior_sin = get_sin_task_sequence(n_restarts, num_test_processes, std)

    # Fill lists
    prior_sequences = [prior_seq_sin]
    gp_list_sequences = [gp_list_sin]
    init_prior = [init_prior_sin]
    return prior_sequences, gp_list_sequences, init_prior


def main():
    print("Starting...")
    # Task family settings
    folder = "result/minigolfv0/"
    env_name = "minigolf-v0"
    prior_var_min = 0.001
    prior_var_max = 0.2
    noise_seq_var = 0.001
    latent_dim = 1
    min_action = 1e-5
    max_action = 10.
    action_space = spaces.Box(low=min_action,
                              high=max_action,
                              shape=(1,))
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    task_generator = MiniGolfTaskGenerator(prior_var_min=prior_var_min,
                                           prior_var_max=prior_var_max)

    if len(args.folder) == 0:
        folder = folder + args.algo + "/"
    else:
        folder = folder + args.folder + "/"
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_var ** (1 / 2))

    print("Algorithm start..")
    if args.algo == 'rl2':
        obs_shape = (4,)

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
                    action_dim=1,
                    use_gae=args.use_gae,
                    gae_lambda=args.gae_lambda,
                    use_proper_time_limits=args.use_proper_time_limits,
                    use_xavier=args.use_xavier,
                    use_obs_rms=args.use_rms_obs)

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
        obs_shape = (2,)

        vi = InferenceNetwork(n_in=5, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = PosteriorOptTSAgent(vi=vi,
                                    vi_optim=vi_optim,
                                    num_steps=args.num_steps,
                                    num_processes=args.num_processes,
                                    device=device,
                                    gamma=args.gamma,
                                    latent_dim=latent_dim,
                                    use_env_obs=True,
                                    max_sigma=[prior_var_max ** (1 / 2)],
                                    min_sigma=[prior_var_min ** (1 / 2)],
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
                                    use_decay_kld=args.use_decay_kld,
                                    decay_kld_rate=args.decay_kld_rate,
                                    env_dim=1,
                                    action_dim=1,
                                    vae_max_steps=args.vae_max_steps,
                                    use_xavier=args.use_xavier,
                                    use_rms_latent=args.use_rms_latent,
                                    use_rms_obs=args.use_rms_obs)

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
        vae_min_seq = 1
        vae_max_seq = args.vae_max_steps

        obs_shape = (3,)

        vi = InferenceNetwork(n_in=5, z_dim=latent_dim)
        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = OursAgent(action_space=action_space,
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
                          use_elu=args.use_elu,
                          variational_model=vi,
                          vae_optim=vi_optim,
                          vae_min_seq=vae_min_seq,
                          vae_max_seq=vae_max_seq,
                          max_sigma=[prior_var_max ** (1 / 2)],
                          min_sigma=[prior_var_min ** (1 / 2)],
                          use_decay_kld=args.use_decay_kld,
                          decay_kld_rate=args.decay_kld_rate,
                          env_dim=1,
                          action_dim=1,
                          use_xavier=args.use_xavier,
                          use_rms_latent=args.use_rms_latent,
                          use_rms_obs=args.use_rms_obs
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
