from gym.envs.registration import register

register(
    id='parametric-continuous-stump-tracks-v0',
    entry_point='TeachMyAgent.environments.envs.parametric_continuous_stump_tracks:ParametricContinuousStumpTracks'
)

register(
    id='parametric-continuous-parkour-v0',
    entry_point='TeachMyAgent.environments.envs.parametric_continuous_parkour:ParametricContinuousParkour'
)