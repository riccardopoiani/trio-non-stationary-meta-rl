import pickle

import torch
import numpy as np
import argparse
from gym import spaces

from configs import cheetah_bayes_arguments, cheetah_rl2_arguments, cheetah_ts_arguments, \
    golf_bayes_arguments, golf_rl2_arguments, \
    golf_ts_arguments, ant_goal_bayes_arguments, ant_goal_rl2_arguments, ant_goal_ts_arguments, \
    golf_with_signals_bayes_arguments

from inference.inference_network import EmbeddingInferenceNetwork, InferenceNetwork
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.cheetah_vel_task_generator import CheetahVelTaskGenerator
from task.mini_golf_task_generator import MiniGolfTaskGenerator
from task.ant_goal_task_generator import AntGoalTaskGenerator
from task.mini_golf_with_signals_generator import MiniGolfSignalsTaskGenerator
from utilities.folder_management import handle_folder_creation


def get_sequences(n_restarts, num_test_processes, std):
    return [], [], []


def main():
    # Task family settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-type')
    parser.add_argument('--algo', default='rl2', help="choose in {'rl2', 'ours', 'ts'}")
    parser.add_argument('--golf-num-signals', type=int, default=None)
    args, rest_args = parser.parse_known_args()
    env = args.env_type
    algo = args.algo
    golf_num_signals = args.golf_num_signals

    if env != "golf_signals":
        assert golf_num_signals is None

    # Retrieve general arguments
    if env == "cheetah_vel":
        if algo == "rl2":
            args = cheetah_rl2_arguments.get_args(rest_args)
        elif algo == "ours":
            args = cheetah_bayes_arguments.get_args(rest_args)
        elif algo == "ts":
            args = cheetah_ts_arguments.get_args(rest_args)
    elif env == "golf" or env == "golf_signals":
        if algo == "rl2":
            args = golf_rl2_arguments.get_args(rest_args)
        elif algo == "ours":
            if env == "golf" or golf_num_signals <= 1:
                args = golf_bayes_arguments.get_args(rest_args)
            else:
                args = golf_with_signals_bayes_arguments.get_args(rest_args)
        elif algo == "ts":
            args = golf_ts_arguments.get_args(rest_args)
    elif env == "ant_goal":
        if algo == "rl2":
            args = ant_goal_rl2_arguments.get_args(rest_args)
        elif algo == "ours":
            args = ant_goal_bayes_arguments.get_args(rest_args)
        elif algo == "ts":
            args = ant_goal_ts_arguments.get_args(rest_args)
    else:
        raise NotImplemented()

    # Retrieve environment settings
    if env == "cheetah_vel":
        use_simple_inference = False
        use_env_obs = True
        state_dim = 20
        action_dim = 6
        latent_dim = 1
        prior_var_min = 0.01
        prior_var_max = 0.3
        high_act = np.ones(6, dtype=np.float32)
        low_act = -np.ones(6, dtype=np.float32)
        action_space = spaces.Box(low=low_act, high=high_act)
        env_name = "cheetahvel-v2"
        task_generator = CheetahVelTaskGenerator(prior_var_min=prior_var_min,
                                                 prior_var_max=prior_var_max)
        prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
        prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]
        folder = "result/cheetahvelv2/"
    elif env == "ant_goal":
        use_simple_inference = False
        use_env_obs = True
        folder = "result/antgoal/"
        env_name = "antgoal-v0"
        prior_var_min = 0.1
        prior_var_max = 0.4
        latent_dim = 2
        state_dim = 113
        action_dim = 8
        high_act = np.ones(8, dtype=np.float32)
        low_act = -np.ones(8, dtype=np.float32)
        action_space = spaces.Box(low=low_act, high=high_act)
        prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
        prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]
        task_generator = AntGoalTaskGenerator(prior_var_min=prior_var_min,
                                              prior_var_max=prior_var_max)
    elif env == "golf":
        use_simple_inference = True
        folder = "result/minigolfv0/"
        env_name = "golf-v0"
        use_env_obs = True
        prior_var_min = 0.001
        prior_var_max = 0.2
        latent_dim = 1
        min_action = 1e-5
        max_action = 10.
        action_dim = 1
        state_dim = 1
        action_space = spaces.Box(low=min_action,
                                  high=max_action,
                                  shape=(1,))
        task_generator = MiniGolfTaskGenerator(prior_var_min=prior_var_min,
                                               prior_var_max=prior_var_max)
        prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
        prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]
    elif env == "golf_signals":
        if golf_num_signals <= 1:
            use_simple_inference = True
        else:
            use_simple_inference = False
        folder = "result/golf_sig_{}/".format(golf_num_signals)
        env_name = "golfsignals-v0"
        use_env_obs = True
        prior_var_min = 0.001
        prior_var_max = 0.2
        latent_dim = 1 + golf_num_signals
        min_action = 1e-5
        max_action = 10.
        action_dim = 1
        state_dim = 1 + golf_num_signals
        action_space = spaces.Box(low=min_action,
                                  high=max_action,
                                  shape=(1,))
        task_generator = MiniGolfSignalsTaskGenerator(prior_var_min=prior_var_min,
                                                      prior_var_max=prior_var_max,
                                                      num_signals=golf_num_signals)
        prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
        prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]

    else:
        raise RuntimeError("Env {} not available".format(env))

    noise_seq_var = 0

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    if len(args.folder) == 0:
        folder = folder + algo + "/"
    else:
        folder = folder + args.folder + "/"
    fd, folder_path_with_date = handle_folder_creation(result_path=folder)

    prior_sequences, gp_list_sequences, init_prior = get_sequences(n_restarts=args.n_restarts_gp,
                                                                   num_test_processes=args.num_test_processes,
                                                                   std=noise_seq_var ** (1 / 2))

    print("Algorithm start..")
    if algo == 'rl2':
        if use_env_obs:
            obs_dim = state_dim + action_dim + 1
        else:
            obs_dim = action_dim + 1
        obs_shape = (obs_dim,)  # 2 obs_shape + 2 action_shape + 1 reward

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
                    use_obs_env=use_env_obs,
                    num_processes=args.num_processes,
                    gamma=args.gamma,
                    device=device,
                    num_steps=args.num_steps,
                    action_dim=action_dim,
                    state_dim=state_dim,
                    use_gae=args.use_gae,
                    gae_lambda=args.gae_lambda,
                    use_proper_time_limits=args.use_proper_time_limits,
                    use_xavier=args.use_xavier,
                    use_huber_loss=args.use_huber_loss,
                    use_extractor=args.use_feature_extractor,
                    reward_extractor_dim=args.rl2_reward_emb_dim,
                    action_extractor_dim=args.rl2_action_emb_dim,
                    state_extractor_dim=args.rl2_state_emb_dim,
                    done_extractor_dim=args.rl2_done_emb_dim,
                    use_done=args.use_done,
                    use_rms_rew=args.use_rms_rew,
                    use_rms_state=args.use_rms_obs,
                    use_rms_act=args.use_rms_act,
                    use_rms_rew_in_policy=args.use_rms_rew_in_policy,
                    latent_dim=args.rl2_latent_dim
                    )

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
    elif algo == 'ts':
        if use_env_obs:
            dim = state_dim + latent_dim
        else:
            dim = latent_dim
        obs_shape = (dim,)  # latent dim + obs

        if use_simple_inference:
            vi = InferenceNetwork(n_in=action_dim+state_dim+1+(latent_dim*2), z_dim=latent_dim)
        else:
            vi = EmbeddingInferenceNetwork(z_dim=latent_dim,
                                           action_dim=action_dim,
                                           action_embedding_dim=args.vae_action_emb_dim,
                                           state_dim=state_dim,
                                           state_embedding_dim=args.vae_state_emb_dim,
                                           reward_embedding_dim=args.vae_reward_emb_dim,
                                           prior_embedding_dim=args.vae_prior_emb_dim,
                                           hidden_size_dim=args.vae_gru_dim)

        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = PosteriorOptTSAgent(vi=vi,
                                    vi_optim=vi_optim,
                                    num_steps=args.num_steps,
                                    num_processes=args.num_processes,
                                    device=device,
                                    gamma=args.gamma,
                                    latent_dim=latent_dim,
                                    use_env_obs=use_env_obs,
                                    max_sigma=prior_std_max,
                                    min_sigma=prior_std_min,
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
                                    use_proper_time_limits=args.use_proper_time_limits,
                                    recurrent_policy=args.recurrent,
                                    hidden_size=args.hidden_size,
                                    use_elu=args.use_elu,
                                    use_decay_kld=args.use_decay_kld,
                                    decay_kld_rate=args.decay_kld_rate,
                                    env_dim=state_dim,
                                    action_dim=action_dim,
                                    vae_max_steps=args.vae_max_steps,
                                    use_xavier=args.use_xavier,
                                    use_feature_extractor=args.use_feature_extractor,
                                    state_extractor_dim=args.state_extractor_dim,
                                    latent_extractor_dim=args.latent_extractor_dim,
                                    use_huber_loss=args.use_huber_loss,
                                    detach_every=args.detach_every,
                                    use_rms_latent=args.use_rms_latent,
                                    use_rms_obs=args.use_rms_obs,
                                    use_rms_rew=args.use_rms_rew
                                    )

        vi_loss, eval_list, test_list, final_test, eval_opt = agent.train(n_train_iter=args.training_iter,
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
                                                                          task_len=args.task_len,
                                                                          is_eval_optimal=True)

        torch.save(agent.vi, "{}agent_vi".format(folder_path_with_date))
        torch.save(agent.actor_critic, "{}agent_ac".format(folder_path_with_date))

        with open("{}vae.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(vi_loss, output)
        with open("{}eval.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(eval_list, output)
        with open("{}test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(test_list, output)
        with open("{}final_test.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(final_test, output)
        with open("{}eval_opt.pkl".format(folder_path_with_date), "wb") as output:
            pickle.dump(eval_opt, output)

    elif algo == "ours":
        vae_min_seq = 1
        vae_max_seq = args.vae_max_steps

        # 2 * latent_dim + obs
        if use_env_obs:
            dim = 2 * latent_dim + state_dim
        else:
            dim = 2 * latent_dim
        obs_shape = (dim,)

        # 8 action + (111 obs + 2 * latent_dim) + 1 reward
        if use_simple_inference:
            vi = InferenceNetwork(n_in=action_dim+state_dim+1+(latent_dim*2), z_dim=latent_dim)
        else:
            vi = EmbeddingInferenceNetwork(z_dim=latent_dim,
                                           action_dim=action_dim,
                                           action_embedding_dim=args.vae_action_emb_dim,
                                           state_dim=state_dim,
                                           state_embedding_dim=args.vae_state_emb_dim,
                                           reward_embedding_dim=args.vae_reward_emb_dim,
                                           prior_embedding_dim=args.vae_prior_emb_dim,
                                           hidden_size_dim=args.vae_gru_dim)
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
                          vae_min_seq=vae_min_seq,
                          vae_max_seq=vae_max_seq,
                          max_sigma=prior_std_max,
                          use_decay_kld=args.use_decay_kld,
                          decay_kld_rate=args.decay_kld_rate,
                          env_dim=state_dim,
                          action_dim=action_dim,
                          min_sigma=prior_std_min,
                          use_xavier=args.use_xavier,
                          use_rms_obs=args.use_rms_obs,
                          use_rms_latent=args.use_rms_latent,
                          use_feature_extractor=args.use_feature_extractor,
                          state_extractor_dim=args.state_extractor_dim,
                          latent_extractor_dim=args.latent_extractor_dim,
                          uncertainty_extractor_dim=args.uncertainty_extractor_dim,
                          use_huber_loss=args.use_huber_loss,
                          detach_every=args.detach_every,
                          use_rms_rew=args.use_rms_rew,
                          decouple_rms=args.decouple_rms_latent
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
        raise NotImplementedError("Agent {} not available".format(algo))


if __name__ == "__main__":
    main()
