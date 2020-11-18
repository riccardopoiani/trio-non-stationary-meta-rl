from gym.envs.registration import register

register(
    'scalegauss-v0',
    entry_point='envs.scaled_gauss_env:ScaledGaussEnv',
    max_episode_steps=15
)

register(
    'golf-v0',
    entry_point='envs.minigolfenv:MiniGolf',
    max_episode_steps=20
)

register(
    'antgoal-v0',
    entry_point='envs.ant_goal:AntGoal',
    max_episode_steps=200
)

register(
    'cheetahvel-v0',
    entry_point="envs.cheetahvel.HalfCheetahVel",
    max_episode_steps=200
)

