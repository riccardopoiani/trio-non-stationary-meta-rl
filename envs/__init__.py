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
    'cheetahvel-v2',
    entry_point="envs.cheetah_vel_v2:HalfCheetahVelEnvV2",
    max_episode_steps=200
)

register(
    'antgoalsignal-v0',
    entry_point='envs.ant_goal_with_signals_states:AntGoalWithSignal',
    max_episode_steps=200
)

