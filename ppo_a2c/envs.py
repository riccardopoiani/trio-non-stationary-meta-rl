"""
Code taken from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
"""
import os

import gym
import numpy as np
import torch
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper, ShmemVecEnv
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from gym.spaces.box import Box

from envs.utils.running_mean_std import RunningMeanStd

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass

"""
class LatentSpaceSmoother:
    def __init__(self, shape, clipob=10., epsilon=1e-8):
        self.clipob = clipob
        self.epsilon = epsilon
        self.ob_rms = RunningMeanStd(shape=shape)

    def step(self, obs):
        obs = obs.numpy()
        self.ob_rms.update(obs)
        obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
        return torch.tensor(obs, dtype=torch.float32)
"""


def make_env_multi_task(env_id, seed, rank, log_dir, allow_early_resets, kwargs):
    def _thunk():
        if env_id.startswith("dm"):
            raise NotImplementedError
        else:
            env = gym.make(env_id, **kwargs)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        if env_id.startswith("dm"):
            _, domain, task = env_id.split('.')
            env = dm_control2gym.make(domain_name=domain, task_name=task)
        else:
            env = gym.make(env_id)

        is_atari = hasattr(gym.envs, 'atari') and isinstance(
            env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
        if is_atari:
            env = make_atari(env_id)

        env.seed(seed + rank)

        if str(env.__class__.__name__).find('TimeLimit') >= 0:
            env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)

        if is_atari:
            if len(env.observation_space.shape) == 3:
                env = wrap_deepmind(env)
        elif len(env.observation_space.shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs_multi_task(env_name,
                             seed,
                             num_processes,
                             gamma,
                             log_dir,
                             device,
                             allow_early_resets,
                             env_kwargs_list,
                             normalize_reward,
                             num_frame_stack=None):
    envs = [
        make_env_multi_task(env_name, seed, i, log_dir, allow_early_resets, env_kwargs_list[i])
        for i in range(num_processes)
    ]

    envs = DummyVecEnv(envs)

    if gamma is None:
        envs = VecNormalize(envs, normalize_rew=normalize_reward, ret_rms=None)
    else:
        envs = VecNormalize(envs, normalize_rew=normalize_reward, ret_rms=None, gamma=gamma)

    envs = VecPyTorch(envs, device)

    if num_frame_stack is not None:
        envs = VecPyTorchFrameStack(envs, num_frame_stack, device)
    elif len(envs.observation_space.shape) == 3:
        envs = VecPyTorchFrameStack(envs, 4, device)

    return envs


def get_vec_envs_multi_task(env_name,
                            seed,
                            num_processes,
                            gamma,
                            log_dir,
                            device,
                            allow_early_resets,
                            env_kwargs_list,
                            envs,
                            normalize_rew,
                            num_frame_stack=None,
                            ):
    if envs is None:
        return make_vec_envs_multi_task(env_name=env_name, seed=seed, num_processes=num_processes, gamma=gamma,
                                        log_dir=log_dir, device=device, allow_early_resets=allow_early_resets,
                                        env_kwargs_list=env_kwargs_list, num_frame_stack=num_frame_stack,
                                        normalize_reward=normalize_rew)
    else:
        for i in range(num_processes):
            envs.envs[i].set_latent(**env_kwargs_list[i])
        return envs


# Checks whether done was caused my timit limits or not
class TimeLimitMask(gym.Wrapper):
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


# Can be used to test recurrent policies for Reacher-v2
class MaskGoal(gym.ObservationWrapper):
    def observation(self, observation):
        if self.env._elapsed_steps > 0:
            observation[-2:] = 0
        return observation


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)

        if isinstance(reward, list):
            reward = [torch.from_numpy(r).unsqueeze(dim=1).float() for r in reward]
        else:
            reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecEnvWrapper):
    def __init__(self, venv, ret_rms, normalize_rew, clipob=10., cliprew=10., gamma=0.99, epsilon=1e-8):
        super(VecNormalize, self).__init__(venv)
        self.clipob = clipob
        self.cliprew = cliprew
        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon
        self.ret_rms = ret_rms
        self.normalize_rew = normalize_rew

        if self.normalize_rew:
            if ret_rms is None:
                self.ret_rms = RunningMeanStd(shape=1)
            else:
                self.ret_rms = ret_rms

        self.training = True

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return obs

    def step_wait(self):
        # execute action
        obs, rews, news, infos = self.venv.step_wait()
        # update discounted return
        self.ret = self.ret * self.gamma + rews
        self.ret[news] = 0.
        # normalise
        rews = self._rewfilt(rews)
        return obs, rews, news, infos

    def _rewfilt(self, rews):
        if self.normalize_rew:
            # update rolling mean / std
            if self.training:
                self.ret_rms.update(self.ret)
            # normalise
            rews_norm = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
            return [rews, rews_norm]
        else:
            return [rews, rews]

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')
        self.stacked_obs = torch.zeros((venv.num_envs,) +
                                       low.shape).to(device)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = \
            self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()


class MyShmemVecEnv(ShmemVecEnv):

    def __init__(self, env_fns, spaces=None, context='spawn'):
        super(MyShmemVecEnv, self).__init__(env_fns, spaces, context)
        self.envs = [e() for e in env_fns]
