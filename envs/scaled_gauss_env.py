import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class ScaledGaussEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    NOISE_STD = 0.05
    ENV_STD = 0.25

    def __init__(self, mean=0):
        super(ScaledGaussEnv, self).__init__()

        # Env. parameters
        self.mean = ((0.8 - (-0.8))/(1 - (-1))) * (mean - 1) + 0.8
        self.state = 0  # There is a single state

        self.action_space = spaces.Box(low=-1., high=1., shape=(1,), dtype="float32")
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([0]))

    def set_latent(self, mean):
        self.mean = ((0.8 - (-0.8))/(1 - (-1))) * (mean - 1) + 0.8

    def reset(self):
        self.state = 0
        return np.array([self.state])

    def get_state(self):
        return np.array([self.state])

    def step(self, action):
        if action < -1:
            action = np.array([-1.])
        elif action > 1:
            action = np.array([1.])

        noise = np.random.normal(loc=0, scale=self.NOISE_STD, size=1)
        reward = np.exp(-((action - self.mean) ** 2) / (self.ENV_STD ** 2)) + noise
        reward = reward[0]

        return np.array([self.state]), reward, False, {}

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
