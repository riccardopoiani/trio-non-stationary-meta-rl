import time
from collections import deque

import torch
import gym_sin
from gym import spaces
from network.vae import InferenceNetwork2
from ppo_a2c.algo.ppo import PPO
from ppo_a2c.envs import AdvancedRewardSmoother
from ppo_a2c.envs import make_vec_envs_multi_task
from ppo_a2c.model import Policy, MLPBase
from ppo_a2c.storage import RolloutStorage
from task.TaskGenerator import GaussianTaskGenerator

from train_active_support import sample_task, sample_prior_dist, identification_evaluate, al_augment_obs, \
    get_reward, get_posterior, ObsSmootherTemp

inference_net = torch.load("notebooks/inference_2")

# Task creation
# Problem param
z_dim = 2
n_tasks = 1000

# Input space range
x_min = -100
x_max = 100

# Training task latent space range
min_mean = -40
max_mean = 40

min_std = 15
max_std = 35

# Prior on the offset task range
mu_min = -10
mu_max = 10

var_min = 0.1
var_max = 5

# Task generator creation
task_gen = GaussianTaskGenerator(x_min=x_min, x_max=x_max)

# Dataset creation
data_set = []
a_set = []
mean_set = []
std_set = []
param = []
for _ in range(n_tasks):
    data, mean, std = sample_task(task_gen, n_batches=1, test_perc=0, batch_size=180, min_mean=min_mean,
                                  max_mean=max_mean, min_std=min_std, max_std=max_std)
    data_set.append(data)
    mean_set.append(mean)
    std_set.append(std)
    param.append((mean.item(), std.item()))

# Prior distribution on the next task
prior_dist = []
for _ in range(n_tasks):
    prior_dist.append(torch.Tensor(sample_prior_dist(z_dim, mu_min=mu_min, mu_max=mu_max, var_min=var_min,
                                                     var_max=var_max)))

# General parameters
device = "cpu"
env_name = "gauss-v0"
seed = 0
gamma = 0.9
log_dir = "."

# Training parameters
num_steps = 30
num_processes = 32

# PPO parametrs
clip_param = 0.2
ppo_epoch = 4
num_mini_batch = 8
value_loss_coef = 0.5
entropy_coef = 0.001
lr = 0.00001
eps = 1e-6
max_grad_norm = 0.5

# Training parameters
use_linear_lr_decay = False
use_gae = False
gae_lambda = 0.95
use_proper_time_limits = False

# A2C
base = MLPBase
obs_shape = (4,)  # state + latent space of the current model
action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype="float")
actor_critic = Policy(obs_shape,
                      action_space, base=base,
                      base_kwargs={'recurrent': False, 'hidden_size': 16, 'use_elu': True})

# PPO
agent = PPO(actor_critic,
            clip_param,
            ppo_epoch,
            num_mini_batch,
            value_loss_coef,
            entropy_coef,
            lr=lr,
            eps=eps,
            max_grad_norm=max_grad_norm,
            use_clipped_value_loss=True)

# Reward log structure
episode_rewards = deque(maxlen=10)

start = time.time()

meta_training_iter = 10000
num_update_per_meta_training_iter = 1
num_task_to_evaluate = num_processes

latent_dim = 2
env_obs_shape = 1
obs_dim = 4

num_processes = 32
log_interval = 5
eval_interval = 1
eval_list = []

variational_model = inference_net

num_replicas = 2
num_different = num_processes // num_replicas

for k in range(meta_training_iter):
    # Sample previous task and prior distribution on the next task
    prev_task_idx = torch.randint(low=0, high=n_tasks, size=(num_different,))
    prev_task = []
    for i in range(num_different):
        for j in range(num_replicas):
            prev_task.append(param[prev_task_idx[i]])

    prior_idx = torch.randint(low=0, high=n_tasks, size=(num_different,))
    prior = []
    for i in range(num_different):
        for j in range(num_replicas):
            prior.append(prior_dist[prior_idx[i]].clone().detach())

    # Sample current task from the prior
    mu = [prior[i][0].clone().detach() for i in range(num_processes)]
    var = [prior[i][1].clone().detach() for i in range(num_processes)]

    sample = []
    for i in range(num_different):
        sample.append(torch.normal(mu[i * num_replicas], var[i * num_replicas]).tolist())

    offset_param = []
    for i in range(num_different):
        for j in range(num_replicas):
            offset_param.append(sample[i])
    offset_param = torch.tensor(offset_param)

    # Modify the prior
    for i in range(num_processes):
        prior[i][0, :] = prior[i][0, :] + torch.tensor(prev_task[i])

    mu = [mu[i] + torch.tensor(prev_task[i]) for i in range(num_processes)]
    new_tasks = offset_param + torch.tensor(prev_task)

    # Sample new task
    envs_kwargs = [{'amplitude': 1,
                    'mean': new_tasks[i][0].item(),
                    'std': new_tasks[i][1].item(),
                    'noise_std': 0.001,
                    'scale_reward': False} for i in range(num_processes)]

    envs = make_vec_envs_multi_task(env_name,
                                    seed,
                                    num_processes,
                                    None,
                                    log_dir,
                                    device,
                                    False,
                                    envs_kwargs,
                                    num_frame_stack=None)

    obs = envs.reset()
    obs = al_augment_obs(obs, prev_task, latent_dim, env_obs_shape, prior, prior)

    rollouts = RolloutStorage(num_steps, num_processes,
                              obs_shape, action_space,
                              actor_critic.recurrent_hidden_state_size)

    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    use_prev_state = False

    rms = AdvancedRewardSmoother(num_processes, num_replicas, gamma=gamma)
    # rms = RewardSmoother(num_envs=num_processes, cliprew=10, gamma=gamma)
    obs_rms = ObsSmootherTemp(obs_shape=(7,))
    obs = obs_rms.step(obs)

    for _ in range(num_update_per_meta_training_iter):
        # Collect observations and store them into the storage
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            posterior = get_posterior(variational_model, action, reward, prior, prev_task,
                                      num_processes, use_prev_state=use_prev_state)
            use_prev_state = True

            reward_prev = get_reward(posterior, new_tasks, latent_dim, num_processes)
            reward = rms.step(reward_prev, done)
            obs = al_augment_obs(obs, prev_task, latent_dim, env_obs_shape, posterior, prior)
            obs = obs_rms.step(obs)

            # If done then clean the history of observations.
            if done.any():
                use_prev_state = False
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, use_gae, gamma,
                                 gae_lambda, use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

    rms.reset()
    eval_interval = 10
    if eval_interval is not None and k % eval_interval == 0 and k > 1:
        e = identification_evaluate(actor_critic, inference_net, env_name, seed, num_processes, ".", device,
                                    32, latent_dim, env_obs_shape,param=param, prior_dist=prior_dist,
                                    n_tasks=n_tasks)
        eval_list.append(e)