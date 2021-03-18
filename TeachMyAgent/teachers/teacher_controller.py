import numpy as np
import pickle
import copy
from TeachMyAgent.teachers.algos.riac import RIAC
from TeachMyAgent.teachers.algos.alp_gmm import ALPGMM
from TeachMyAgent.teachers.algos.covar_gmm import CovarGMM
from TeachMyAgent.teachers.algos.adr import ADR
from TeachMyAgent.teachers.algos.self_paced_teacher import SelfPacedTeacher
from ACL_bench.teachers.algos.truncated_self_paced import SelfPacedTeacher as TruncatedSelfPacedTeacher
from TeachMyAgent.teachers.algos.goal_gan import GoalGAN
from TeachMyAgent.teachers.algos.setter_solver import SetterSolver
from TeachMyAgent.teachers.algos.random_teacher import RandomTeacher
from TeachMyAgent.teachers.utils.dimensions_shuffler import DimensionsShuffler
from collections import OrderedDict

# Utils functions to convert task vector into dictionary (or the opposite)
def param_vec_to_param_dict(param_env_bounds, param):
    param_dict = OrderedDict()
    cpt = 0
    for i,(name, bounds) in enumerate(param_env_bounds.items()):
        if type(bounds[0]) is list:
            nb_dims = len(bounds)
            param_dict[name] = param[cpt:cpt+nb_dims]
            cpt += nb_dims
        else:
            if len(bounds) == 2:
                param_dict[name] = param[cpt]
                cpt += 1
            elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                nb_dims = bounds[2]
                param_dict[name] = param[cpt:cpt+nb_dims]
                cpt += nb_dims

    return param_dict

def param_dict_to_param_vec(param_env_bounds, param_dict):
    param_vec = []
    for name, bounds in param_env_bounds.items():
        if isinstance(param_dict[name], list) or isinstance(param_dict[name], np.ndarray):
            param_vec.extend(param_dict[name])
        else:
            param_vec.append(param_dict[name])

    return np.array(param_vec)

# Class controlling the interactions between ACL methods and DeepRL students
class TeacherController(object):
    def __init__(self, teacher, nb_test_episodes, param_env_bounds, seed=None, test_set=None,
                 keep_periodical_task_samples=None, shuffle_dimensions=False, scale_reward=False, **teacher_params):
        self.teacher = teacher
        self.nb_test_episodes = nb_test_episodes
        self.test_set = test_set
        self.test_ep_counter = 0
        self.train_step_counter = 0
        self.eps= 1e-03
        self.param_env_bounds = copy.deepcopy(param_env_bounds)
        self.keep_periodical_task_samples = keep_periodical_task_samples
        self.scale_reward = scale_reward

        # figure out parameters boundaries vectors
        mins, maxs = [], []
        for name, bounds in param_env_bounds.items():
            if type(bounds[0]) is list:
                try:
                    # Define min / max for each dim
                    for dim in bounds:
                        mins.append(dim[0])
                        maxs.append(dim[1])
                except:
                    print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                    exit(1)
            else:
                if len(bounds) == 2:
                    mins.append(bounds[0])
                    maxs.append(bounds[1])
                elif len(bounds) == 3:  # third value is the number of dimensions having these bounds
                    mins.extend([bounds[0]] * bounds[2])
                    maxs.extend([bounds[1]] * bounds[2])
                else:
                    print("ill defined boundaries, use [min, max, nb_dims] format or [min, max] if nb_dims=1")
                    exit(1)
        self.task_dim = len(mins)
        if shuffle_dimensions:
            self.dimensions_shuffler = DimensionsShuffler(mins, maxs, seed=seed)
            if "initial_dist" in teacher_params:
                teacher_params["initial_dist"]["mean"] = self.dimensions_shuffler.inverse_interpolate_task(
                    teacher_params["initial_dist"]["mean"])
            if "target_dist" in teacher_params:
                    teacher_params["target_dist"]["mean"] = self.dimensions_shuffler.inverse_interpolate_task(
                        teacher_params["target_dist"]["mean"])
        else:
            self.dimensions_shuffler = None

        # setup tasks generator
        if teacher == 'Random':
            self.task_generator = RandomTeacher(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'RIAC':
            self.task_generator = RIAC(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'ALP-GMM':
            self.task_generator = ALPGMM(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'Covar-GMM':
            self.task_generator = CovarGMM(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'ADR':
            self.task_generator = ADR(mins, maxs, seed=seed, scale_reward=scale_reward, **teacher_params)
        elif teacher == 'Self-Paced':
            self.task_generator = SelfPacedTeacher(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'Truncated-Self-Paced':
            self.task_generator = TruncatedSelfPacedTeacher(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'GoalGAN':
            self.task_generator = GoalGAN(mins, maxs, seed=seed, **teacher_params)
        elif teacher == 'Setter-Solver':
            self.task_generator = SetterSolver(mins, maxs, seed=seed, **teacher_params)
        else:
            print('Unknown teacher')
            raise NotImplementedError

        # Generate test set
        ## Use evenly distributed tasks for StumpTracks
        ## Use uniform sampling otherwise
        ## Or load a saved test set
        test_param_vec = None
        if test_set is None:
            if self.task_dim == 2 and "stump_height" in param_env_bounds and "obstacle_spacing" in param_env_bounds: # StumpTracks
                print("Using random test set for two fixed dimensions.")
                # select <nb_test_episodes> parameters choosen uniformly in the task space
                nb_steps = int(nb_test_episodes ** (1 / self.task_dim))
                d1 = np.linspace(mins[0], maxs[0], nb_steps, endpoint=True)
                d2 = np.linspace(mins[1], maxs[1], nb_steps, endpoint=True)
                test_param_vec = np.transpose([np.tile(d1, len(d2)), np.repeat(d2, len(d1))])  # cartesian product
            else:
                print("Using random test set.")
                test_random_state = np.random.RandomState(
                    31)  # Seed a new random generator not impacting the global one to always get the same test set
                test_param_vec = test_random_state.uniform(mins, maxs, size=(nb_test_episodes, self.task_dim))
        else:
            test_param_vec = np.array(pickle.load(open("TeachMyAgent/teachers/test_sets/"+test_set+".pkl", "rb")))
            self.nb_test_episodes = len(test_param_vec)
            print('fixed set of {} tasks loaded'.format(len(test_param_vec)))
        test_param_dicts = [param_vec_to_param_dict(param_env_bounds, vec) for vec in test_param_vec]
        self.test_env_list = test_param_dicts

        # Data recording
        self.env_params_train = []
        self.env_train_rewards = []
        self.env_train_norm_rewards = []
        self.env_train_len = []
        self.periodical_task_samples = []
        self.periodical_task_infos = []

        self.env_params_test = []
        self.env_test_rewards = []
        self.env_test_len = []

    def _get_last_task(self):
        params = self.env_params_train[-1]
        if self.dimensions_shuffler is not None:
            params = self.dimensions_shuffler.last_raw_task
        return params

    def set_value_estimator(self, estimator):
        self.task_generator.value_estimator = estimator

    def record_train_task_initial_state(self, initial_state):
        self.task_generator.record_initial_state(self._get_last_task(), initial_state)

    def record_train_step(self, state, action, reward, next_state, done):
        self.train_step_counter += 1
        self.task_generator.step_update(state, action, reward, next_state, done)
        # Monitor curriculum
        if self.keep_periodical_task_samples is not None \
                and self.train_step_counter % self.keep_periodical_task_samples == 0:
            tasks = []
            infos = []
            if self.task_generator.is_non_exploratory_task_sampling_available():
                for i in range(100):
                    task_and_infos = self.task_generator.non_exploratory_task_sampling()
                    tasks.append(task_and_infos["task"])
                    infos.append(task_and_infos["infos"])
            self.periodical_task_samples.append(np.array(tasks))
            self.periodical_task_infos.append(np.array(infos))

    def record_train_episode(self, ep_reward, ep_len, is_success=False):
        self.env_train_rewards.append(ep_reward)
        self.env_train_len.append(ep_len)
        if self.scale_reward and self.teacher != 'Oracle':
            ep_reward = np.interp(ep_reward,
                                  (self.task_generator.env_reward_lb, self.task_generator.env_reward_ub),
                                  (0, 1))
            self.env_train_norm_rewards.append(ep_reward)
        self.task_generator.episodic_update(self._get_last_task(), ep_reward, is_success)

    def record_test_episode(self, reward, ep_len):
        self.env_test_rewards.append(reward)
        self.env_test_len.append(ep_len)

    def dump(self, filename):
        with open(filename, 'wb') as handle:
            dump_dict = {'env_params_train': self.env_params_train,
                         'env_train_rewards': self.env_train_rewards,
                         'env_train_len': self.env_train_len,
                         'env_params_test': self.env_params_test,
                         'env_test_rewards': self.env_test_rewards,
                         'env_test_len': self.env_test_len,
                         'env_param_bounds': list(self.param_env_bounds.items()),
                         'periodical_samples': self.periodical_task_samples,
                         'periodical_infos': self.periodical_task_infos}
            dump_dict = self.task_generator.dump(dump_dict)
            pickle.dump(dump_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def set_env_params(self, env):
        params = copy.copy(self.task_generator.sample_task())
        if self.dimensions_shuffler is not None:
            params = self.dimensions_shuffler.interpolate_task(params)
        self.env_params_train.append(params)
        if len(params) > 0:
            assert type(params[0]) == np.float32
            param_dict = param_vec_to_param_dict(self.param_env_bounds, params)
            env.set_environment(**param_dict)
        return params

    def set_test_env_params(self, test_env):
        self.test_ep_counter += 1
        test_param_dict = self.test_env_list[self.test_ep_counter - 1]

        if self.test_set == "hexagon_test_set":
            # removing legacy parameters from test_set, don't pay attention
            legacy = ['tunnel_height', 'gap_width', 'step_height', 'step_number']
            keys = test_param_dict.keys()
            for env_param in legacy:
                if env_param in keys:
                    del test_param_dict[env_param]

        test_param_vec = param_dict_to_param_vec(self.param_env_bounds, test_param_dict)
        if len(test_param_vec) > 0:
            self.env_params_test.append(test_param_vec)
            test_env.set_environment(**test_param_dict)

        if self.test_ep_counter == self.nb_test_episodes:
            self.test_ep_counter = 0
        return test_param_dict