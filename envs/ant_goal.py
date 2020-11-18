import random

import numpy as np
from gym.envs.mujoco import mujoco_env


class AntGoal(mujoco_env.MujocoEnv):
    def __init__(self, goal_x, goal_y):
        # Rescale goal
        goal_x = ((3 - (-3)) / (1 - (-1))) * (goal_x - 1) + 3
        goal_y = ((3 - (-3)) / (1 - (-1))) * (goal_y - 1) + 3
        self.goal_pos = np.array([goal_x, goal_y])

        mujoco_env.MujocoEnv.__init__(self, model_path='ant.xml', frame_skip=5)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = 0.1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward
        )

    def set_latent(self, goal_x, goal_y):
        goal_x = ((3 - (-3)) / (1 - (-1))) * (goal_x - 1) + 3
        goal_y = ((3 - (-3)) / (1 - (-1))) * (goal_y - 1) + 3
        self.goal_pos = np.array([goal_x, goal_y])

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()
