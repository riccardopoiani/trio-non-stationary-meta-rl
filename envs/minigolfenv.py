import gym
import numpy as np
from gym import spaces
from gym.utils import seeding

"""
Minigolf task.
References
----------
  - Penner, A. R. "The physics of putting." Canadian Journal of Physics 80.2 (2002): 83-96.
"""


class MiniGolf(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, friction=0):
        super(MiniGolf, self).__init__()

        self.horizon = 20
        self.gamma = 0.99

        self.min_pos = 0.0
        self.max_pos = 20.0
        self.min_action = 1e-5
        self.max_action = 10.0
        self.putter_length = 1.0  # [0.7:1.0]
        self.friction = (2 - 0.01) / (1 - (-1)) * (friction - 1) + 2  # [0.065:0.196]
        self.hole_size = 0.10  # [0.10:0.15]
        self.sigma_noise = 0.3
        self.ball_radius = 0.02135
        self.min_variance = 1e-2  # Minimum variance for computing the densities

        # gym attributes
        self.viewer = None
        low = np.array([self.min_pos])
        high = np.array([self.max_pos])
        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(1,))
        self.observation_space = spaces.Box(low=low, high=high)

        # initialize state
        self.seed()
        self.reset()

    def set_latent(self, friction=0):
        self.friction = (2 - 0.01) / (1 - (-1)) * (friction - 1) + 2

    def step(self, action, render=False):
        action = np.clip(action, self.min_action, self.max_action / 2)

        noise = 10
        while abs(noise) > 1:
            noise = self.np_random.randn() * self.sigma_noise
        u = action * self.putter_length * (1 + noise)

        v_min = np.sqrt(10 / 7 * self.friction * 9.81 * self.state)
        v_max = np.sqrt((2 * self.hole_size - self.ball_radius) ** 2 * (9.81 / (2 * self.ball_radius)) + v_min ** 2)

        deceleration = 5 / 7 * self.friction * 9.81

        t = u / deceleration
        xn = self.state - u * t + 0.5 * deceleration * t ** 2

        reward = 0
        done = True
        if u < v_min:
            reward = -1
            done = False
        elif u > v_max:
            reward = -100

        self.state = xn

        # TODO the last three values should not be used
        return self.get_state(), float(reward), done, {}

    def reset(self, state=None):
        self.state = np.array([self.np_random.uniform(low=self.min_pos,
                                                      high=self.max_pos)])

        return self.get_state()

    def get_state(self):
        return np.array(self.state)

    def render(self, mode='human'):
        raise NotImplementedError

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        pass
