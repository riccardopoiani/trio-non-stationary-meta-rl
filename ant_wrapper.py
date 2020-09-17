import gym
import numpy as np


class AntWrapper(gym.Env):

    def __init__(self):
        self.env = gym.make('Ant-v2')

        # ALTERNATIVE 1
        # "friction" for each actuator that effectively clips the maximum actions
        # friction=1 corresponds to the actuator working normally
        # friction=0 corresponds to the actuator not working at all
        # self.frictions = np.ones(8)  # default parameters

        # ALTERNATIVE 2
        # More "realistic" alternative
        # Each friction \alpha induces a clip value of 1/(1+\alpha)
        # So \alpha=0 means no friction and \alpha=inf means infinite friction
        self.frictions = np.zeros(8)

    def reset(self):
        self.env.reset()

    def step(self, action):

        # ALTERNATIVE 1
        # clip_vals = self.frictions

        # ALTERNATIVE 2
        clip_vals = 1 / (1 + self.frictions)

        action = np.clip(action, -clip_vals, clip_vals)

        return self.env.step(action)

    def render(self, mode='human'):
        self.env.render(mode=mode)

    def set_latent(self, frictions):
        assert frictions.shape == (8,), "You must specify 8 friction paramers"

        # ALTERNATIVE 1
        # assert np.all(0 <= frictions <= 1), "Frictions must be numbers from 0 to 1"

        # ALTERNATIVE 2
        assert np.all(0 <= frictions), "Frictions must be positive numbers"

        self.frictions = frictions


env = AntWrapper()
obs = env.reset()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

print (obs_dim,act_dim)

for i in range(1000):
    env.render()
    action = np.random.randn(act_dim,1)
    #action = action.reshape((1,-1)).astype(np.float32)
    obs, reward, done, _ = env.step(np.squeeze(action, axis=0))