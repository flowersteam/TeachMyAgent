import numpy as np
from TeachMyAgent.students.openai_baselines.common.wrappers import ClipActionsWrapper
from TeachMyAgent.students.openai_baselines.common.vec_env import VecNormalize, DummyVecEnv

class CustomVecNormalizeEnv(VecNormalize):
    '''
        Custom inherited class that gives access to raw observation and reward (i.e. unscaled).
    '''
    def _rewsfilt(self, rews, news):
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return rews

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        for i in range(len(rews)):
            infos[i]["original_reward"] = rews
            infos[i]["original_obs"] = obs
        obs = self._obfilt(obs)
        rews = self._rewsfilt(rews, news)
        return obs, rews, news, infos

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return obs, self._obfilt(obs)

    def get_raw_env(self):
        return self.envs[0].env

    def __load_rms__(self, ob_rms, ret_rms):
        self.ob_rms = ob_rms
        self.ret_rms = ret_rms

class TwinCustomVecNormalizeEnv(CustomVecNormalizeEnv):
    '''
        Share reward and observation normalization with a twin class (i.e. for the test env).

        Use this to make the test env scale reward and observation the same way the training env does.
    '''
    def __init__(self, twin_env, venv, **kwargs):
        super().__init__(venv, **kwargs)
        self.twin = twin_env

    def _rewsfilt(self, rews, news):
        self.ret = self.ret * self.gamma + rews
        if self.ret_rms:
            rews = np.clip(rews / np.sqrt(self.twin.ret_rms.var + self.epsilon), -self.cliprew, self.cliprew)
        self.ret[news] = 0.
        return rews

    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.twin.ob_rms.mean) / np.sqrt(self.twin.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

def create_clipped_env(env_f):
    vanilla_env = env_f()
    return ClipActionsWrapper(vanilla_env)

def create_custom_vec_normalized_envs(env_f):
    '''
        Create two vectorized, normalized and clipped envs (i.e. training env and test env).
    '''
    clipped_env = DummyVecEnv([lambda: create_clipped_env(env_f)])
    vec_normalized_env = CustomVecNormalizeEnv(clipped_env, use_tf=False)

    clipped_test_env = DummyVecEnv([lambda: create_clipped_env(env_f)])
    vec_normalized_test_env = TwinCustomVecNormalizeEnv(vec_normalized_env, clipped_test_env, use_tf=False)

    return vec_normalized_env, vec_normalized_test_env
