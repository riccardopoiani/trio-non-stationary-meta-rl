from gym.envs.registration import register

register(
    'golf-v0',
    entry_point='envs.minigolfenv:MiniGolf',
    max_episode_steps=20
)

register(
    'golfsignals-v0',
    entry_point='envs.golf_with_signals:MiniGolfWithSignals',
    max_episode_steps=20
)

register(
    'antgoal-v0',
    entry_point='envs.ant_goal:AntGoal',
    max_episode_steps=200
)

register(
    'cheetahvel-v2',
    entry_point="envs.cheetah_vel_v2:HalfCheetahVelEnvV2",
    max_episode_steps=200
)

