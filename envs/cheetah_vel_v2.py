import numpy as np

from gym.envs.mujoco import HalfCheetahEnv


class HalfCheetahVelEnvV2(HalfCheetahEnv):
    def __init__(self, goal_velocity):
        self.goal_velocity = ((1.5 - 0.0) / (1 - (-1))) * (goal_velocity - 1) + 1.5
        self.task_dim = 1
        super(HalfCheetahVelEnvV2, self).__init__()

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        if abs(forward_vel - self.goal_velocity) < 0.5:
            forward_reward = -1.0 * abs(forward_vel - self.goal_velocity)
        else:
            forward_reward = -10.0 + (-1.0) * abs(forward_vel - self.goal_velocity)
        ctrl_cost = 0.1 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost,
                     task=self.get_latent())
        return observation, reward, done, infos

    def set_latent(self, goal_velocity):
        self.goal_velocity = ((1.5 - 0.0) / (1 - (-1))) * (goal_velocity - 1) + 1.5

    def get_latent(self):
        return np.array([self.goal_velocity])
