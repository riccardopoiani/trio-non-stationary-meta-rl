import pickle

import torch
import numpy as np
import gym_sin
from gym import spaces

from inference.inference_network import EmbeddingInferenceNetwork
from learner.ours import OursAgent
from learner.posterior_ts_opt import PosteriorOptTSAgent
from learner.recurrent import RL2
from task.ant_goal_task_generator import AntGoalTaskGenerator
from utilities.arguments import get_args
from utilities.folder_management import handle_folder_creation


def get_sequences(n_restarts, num_test_processes, std):
    return [], [], []


def main():
    print("Starting...")
    # Task family settings
    folder = "result/antgoal/"
    env_name = "antgoal-v0"

    # Task family parameters
    prior_var_min = 0.1
    prior_var_max = 0.4
    latent_dim = 2

    state_dim = 113
    action_dim = 8

    high_act = np.ones(8, dtype=np.float32)

    low_act = -np.ones(8, dtype=np.float32)

    action_space = spaces.Box(low=low_act, high=high_act)

    # Other settings
    noise_seq_var = 0.001

    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    task_generator = AntGoalTaskGenerator(prior_var_max=prior_var_max,
                                          prior_var_min=prior_var_min)

    prior_std_max = [prior_var_max ** (1 / 2) for _ in range(latent_dim)]
    prior_std_min = [prior_var_min ** (1 / 2) for _ in range(latent_dim)]

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
        obs_shape = (state_dim + action_dim + 1,)  # 2 obs_shape + 2 action_shape + 1 reward

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
    elif args.algo == 'ts_opt':
        obs_shape = (state_dim + latent_dim,)  # latent dim + obs

        vi = EmbeddingInferenceNetwork(z_dim=latent_dim,
                                       action_dim=action_dim,
                                       action_embedding_dim=args.vae_action_emb_dim,
                                       state_dim=state_dim,
                                       state_embedding_dim=args.vae_state_emb_dim,
                                       reward_embedding_dim=args.vae_reward_emb_dim,
                                       prior_embedding_dim=args.vae_prior_emb_dim,
                                       hidden_size_dim=args.vae_gru_dim)  # 2 action + 2 obs + 1 reward + 10 prior (latent dim * 2)

        vi_optim = torch.optim.Adam(vi.parameters(), lr=args.vae_lr)

        agent = PosteriorOptTSAgent(vi=vi,
                                    vi_optim=vi_optim,
                                    num_steps=args.num_steps,
                                    num_processes=args.num_processes,
                                    device=device,
                                    gamma=args.gamma,
                                    latent_dim=latent_dim,
                                    use_env_obs=True,
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

    elif args.algo == "ours":
        vae_min_seq = 1
        vae_max_seq = args.vae_max_steps

        # 2 * latent_dim + obs
        obs_shape = (2 * latent_dim + state_dim,)

        # 8 action + (111 obs + 2 * latent_dim) + 1 reward
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
        raise NotImplementedError("Agent {} not available".format(args.algo))


if __name__ == "__main__":
    main()
