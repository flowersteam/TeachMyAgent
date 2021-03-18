import argparse
import os
import time
from copy import deepcopy

from TeachMyAgent.run_utils.student_args_handler import StudentArgsHandler
from TeachMyAgent.run_utils.environment_args_handler import EnvironmentArgsHandler
from TeachMyAgent.run_utils.teacher_args_handler import TeacherArgsHandler

##### USE THIS FILE TO MONITOR GENERATION TIME OF THE ENVIRONMENTS #####

def dict_to_args_str(dictionary):
    args_str = []
    for key in dictionary:
        args_str.append("--{}".format(key))
        if dictionary[key] is not None:
            args_str.append("{}".format(dictionary[key]))

    return args_str

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

STEPS = 10000
REPEAT_STEPS = 3
STEPS_PER_EPISODE = 2000

# Set first value to the minimum so that the env is always the same, set the second as the upper bound
envs = [
    # {
    #     'env':'bipedal-walker-continuous-v0',
    #     "walker_type": ["default"],
    #     "max_stump_h": [0, 3.0],
    #     "max_stump_w": None,
    #     "max_stump_r": None,
    #     "roughness": None,
    #     "max_obstacle_spacing": [0, 6.0],
    #     "max_gap_w": None,
    #     "step_h": None,
    #     "step_nb": None,
    #     "hexa_shape": None,
    #     "stump_seq": None
    # },
    # {
    #     'env':'parametric-continuous-walker-v0',
    #     "walker_type": ["old_classic_bipedal", "classic_bipedal", "profile_chimpanzee"],
    #     "motors_torque": 80,
    #     "max_stump_h": [0, 3.0],
    #     "max_stump_w": None,
    #     "max_stump_r": None,
    #     "roughness": None,
    #     "max_obstacle_spacing": [0, 6.0],
    #     "max_gap_w": None,
    #     "step_h": None,
    #     "step_nb": None,
    #     "hexa_shape": None,
    #     "stump_seq": None
    # },
    # {
    #     'env':'parametric-continuous-climber-v0',
    #     "motors_torque": 300,
    #     "walker_type": ["back_chimpanzee"],
    #     "lidar_group": 6,
    #     "max_splitting_height": [1.5,5],
    #     "max_circle_radius": [0.18, 1],
    #     "max_space_between_grips": [3.3, 5],
    #     "max_y_variations_std": [0, 0.5],
    #     "max_group_sections": [2, 5],
    #     "max_group_x_variations_std": [0, 1.5]
    # },
    {
        'env':'parametric-continuous-parkour-v0',
        "walker_type": ["old_classic_bipedal"],
    },
{
        'env':'parametric-continuous-parkour-v0',
        "walker_type": ["old_classic_bipedal"],
        "movable_creepers": None,
    },
    # {
    #     'env':'parametric-continuous-flat-parkour-v0',
    #     "walker_type": ["classic_bipedal", "fish"],
    #     "motors_torque": 80,
    #     "water_level": 4,
    #     "dummy_param": [0, 1]
    # }
]

parser = argparse.ArgumentParser()
parser.add_argument('--seed', '-s', type=int, default=0)
StudentArgsHandler.set_parser_arguments(parser)
EnvironmentArgsHandler.set_parser_arguments(parser)
TeacherArgsHandler.set_parser_arguments(parser)
for _env in envs:
    _env["teacher"] = 'Random'
    _env["nb_test_episodes"] = 1
    _env["seed"] = 43
    current_args_dict = deepcopy(_env)
    for walker_type in _env["walker_type"]:
        print("##### Benchmarking {0} with {1} body #####".format(_env['env'], walker_type))
        current_args_dict["walker_type"] = walker_type
        args_str = dict_to_args_str(current_args_dict)
        args = parser.parse_args(args_str)

        for j in range(2):
            if j == 0:
                print("## Fixed env ##") # Because the first value equals to the min possible
            else:
                print("## Random env ##")

            env_fn, param_env_bounds = EnvironmentArgsHandler.get_object_from_arguments(args)
            Teacher = TeacherArgsHandler.get_object_from_arguments(args, param_env_bounds)

            t = time.time()
            env = env_fn()
            print("Env make took {0} s".format(time.time() - t))

            t = time.time()
            Teacher.set_env_params(env)
            env.reset()
            print("First reset took {0} s".format(time.time() - t))

            for w in range(REPEAT_STEPS):
                nb_of_resets = 0
                t = time.time()
                for i in range(STEPS):
                    if hasattr(env, "action_space"):
                        a = env.action_space.sample()
                    else:
                        a = env.env.action_space.sample()

                    _, _, d, _ = env.step(a)

                    if d or i % STEPS_PER_EPISODE == 0:
                        nb_of_resets+=1
                        Teacher.set_env_params(env)
                        env.reset()
                print("Playing {0} steps for the {1} time  took for {2} s".format(STEPS, w+1, time.time() - t))
                print("During this, the environment was reset {0} times".format(nb_of_resets))

            t = time.time()
            Teacher.set_env_params(env)
            env.reset()
            print("Last reset took {0} s".format(time.time() - t))

            env.close()