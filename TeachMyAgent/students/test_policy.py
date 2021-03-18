import json
import os
from os import listdir
import os.path as osp
import pickle
import time
from collections import OrderedDict, Counter
import argparse
import numpy as np
import pathlib
import joblib

from gym.wrappers.monitoring.video_recorder import VideoRecorder

from TeachMyAgent.run_utils.environment_args_handler import EnvironmentArgsHandler
from TeachMyAgent.run_utils.student_args_handler import StudentArgsHandler
from TeachMyAgent.teachers.teacher_controller import param_vec_to_param_dict

from TeachMyAgent.students.spinup.utils.test_policy import load_policy_and_env as spinup_load_policy
from TeachMyAgent.students.openai_baselines.ppo2.ppo2 import get_model as get_baselines_model
from TeachMyAgent.students.ppo_utils import create_custom_vec_normalized_envs

def get_student_type(save_path):
    for root, _, files in os.walk(save_path):
        if 'progress.txt' in files: # Spinup
            return 'spinup'
        elif 'progress.csv' in files: # OpenAI Baselines
            return 'baselines'

def load_training_infos(save_path):
    with open(osp.join(save_path, 'config.json')) as json_file:
        training_config = json.load(json_file)
    return training_config

def get_baselines_last_checkpoint(path):
    last_checkpoint = -1
    for f in listdir(path):
        if osp.isfile(osp.join(path, f)):
            try:
                checkpoint = int(f)
                last_checkpoint = f if checkpoint > int(last_checkpoint) else last_checkpoint
            except Exception:
                continue
    return last_checkpoint

def load_env_params(save_path):
    with open(osp.join(save_path, 'env_params_save.pkl'), "rb") as file:
        teacher_dict = pickle.load(file)
    return teacher_dict
def get_training_test_size(teacher_dict):
    param_to_count = teacher_dict["env_params_test"][0]
    nb_of_epochs = 0
    for param in teacher_dict["env_params_test"]:
        if (param_to_count == param).all():
            nb_of_epochs += 1

    return int(len(teacher_dict["env_params_test"]) / nb_of_epochs)

def load_training_test_set(save_path, order_by_best_rewards=None):
    ### Get last training test episodes and sort them by total reward
    teacher_dict = load_env_params(save_path)
    test_set_size = get_training_test_size(teacher_dict)

    test_params_to_use = teacher_dict["env_params_test"][-test_set_size:]  # nth last
    test_rewards_to_use = teacher_dict["env_test_rewards"][-test_set_size:]
    if order_by_best_rewards is not None:
        print("Getting test set tasks ordered by last return from {} to {} ..."
              .format("greatest" if order_by_best_rewards else "lowest",
                      "lowest" if order_by_best_rewards else "greatest"))
        sorted_indexes_of_test_episodes = sorted(range(test_set_size),
                                                 key=lambda k: test_rewards_to_use[k],
                                                 reverse=order_by_best_rewards)  # Sort with best results first
    else:
        print("Getting test set tasks as defined...")
        sorted_indexes_of_test_episodes = range(test_set_size)

    teacher_param_env_bounds = OrderedDict(teacher_dict["env_param_bounds"])
    env_params_list = [param_vec_to_param_dict(teacher_param_env_bounds, test_params_to_use[i])
                       for i in sorted_indexes_of_test_episodes]
    associated_rewards_list = [test_rewards_to_use[i] for i in sorted_indexes_of_test_episodes]
    return env_params_list, associated_rewards_list

def load_fixed_test_set(save_path, test_set_name):
    teacher_dict = load_env_params(save_path)
    teacher_param_env_bounds = OrderedDict(teacher_dict["env_param_bounds"])
    test_param_vec = np.array(pickle.load(open("TeachMyAgent/teachers/test_sets/" + test_set_name + ".pkl", "rb")))

    return [param_vec_to_param_dict(teacher_param_env_bounds, vec) for vec in test_param_vec]

def load_env(save_path, load_test_env=False):
    try:
        filename = osp.join(save_path, 'vars.pkl')
        state = joblib.load(filename)
        if load_test_env:
            env = state['test_env']
        else:
            env = state['env']
    except Exception as err:
        print("Unable to load envs : {}".format(err))
        env = None

    return env

def load_vectorized_env(save_path, env):
    try:
        filename = osp.join(save_path, 'vars.pkl')
        state = joblib.load(filename)
        env.__load_rms__(state["ob_rms"], state["ret_rms"])
    except Exception as err:
        print("Unable to load Running Mean Stds : {}".format(err))

def run_policy(env, get_action, env_params_list, max_ep_len=None, episode_id=0, record=False, recording_path=None,
               no_render=False, use_baselines=False):
    if record:
        if os.name == "nt":
            full_path = os.path.join(pathlib.Path().absolute(), recording_path)
            full_path_len = len(full_path)
            nb_char_to_remove = full_path_len - 245
            if nb_char_to_remove > 0:
                recording_path = recording_path[:-nb_char_to_remove]
        video_recorder = VideoRecorder(env, recording_path + "_ep" + str(episode_id) + ".mp4", enabled=True)

    if use_baselines:
        env.get_raw_env().set_environment(**env_params_list[episode_id])
    else:
        env.set_environment(**env_params_list[episode_id])

    if use_baselines:
        _, o = env.reset()
    else:
        o = env.reset()

    r, d, ep_ret, ep_len, n = 0, False, 0, 0, 0
    while True:
        if record and video_recorder.enabled:
            video_recorder.capture_frame()
        if not record and not no_render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)
        o, r, d, i = env.step(a)
        if use_baselines:
            ep_ret += i[0]["original_reward"][0]
        else:
            ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(episode_id, ep_ret, ep_len))
            if record and video_recorder.enabled:
                video_recorder.close()
                video_recorder.enabled = False
            break
    return ep_ret

def main(args):
    if args.fixed_test_set is None:
        # training_config = load_training_infos(args.fpath)
        # nb_test_episodes_during_training = training_config["num_test_episodes"] \
        #     if "num_test_episodes" in training_config \
        #     else training_config["nb_test_episodes"]
        test_set_params, _ = load_training_test_set(args.fpath, args.bests)
    else:
        test_set_params = load_fixed_test_set(args.fpath, args.fixed_test_set)

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    student_type = get_student_type(args.fpath)

    env = None
    if args.load_env:
        env = load_env(args.fpath, args.use_test_env is not None)

    if env is None:
        env_fn, _, _, _ = EnvironmentArgsHandler.get_object_from_arguments(args)
        if student_type == "spinup":
            env = env_fn()
        elif student_type == "baselines":
            env, _ = create_custom_vec_normalized_envs(env_fn)
            load_vectorized_env(args.fpath, env)

    if student_type == 'spinup':
        get_action = spinup_load_policy(args.fpath,
                                        args.itr if args.itr >= 0 else 'last',
                                        args.deterministic)
        env._SET_RENDERING_VIEWPORT_SIZE(600, 400)
    elif student_type == 'baselines':
        ac_kwargs = dict()
        ac_kwargs['hidden_sizes'] = [int(layer) for layer in args.hidden_sizes.split("/")]
        nbatch_train = args.nb_env_steps * 1e6 // int(args.sample_size//args.batch_size)

        model = get_baselines_model(network=args.network, nbatch_train=nbatch_train, ob_space=env.observation_space,
                                    ac_space=env.action_space, env=env, nsteps=args.sample_size, ent_coef=args.ent_coef,
                                    vf_coef=args.vf_coef, hidden_sizes=ac_kwargs['hidden_sizes'])
        last_checkpoint = get_baselines_last_checkpoint(args.fpath + "/checkpoints/")
        model.load(args.fpath + "/checkpoints/" + last_checkpoint)
        # Careful : The recurrent version is not implemented here yet
        get_action = lambda o: model.step(o)[0]
        env.get_raw_env()._SET_RENDERING_VIEWPORT_SIZE(600, 400)
    else:
        raise Exception('Unknown student type.')

    if args.episode_ids == "-1":
        print("Testing the policy on the whole test set...")
        episodes = [i for i in range(len(test_set_params))]
    else:
        episodes = [int(id) for id in args.episode_ids.split("/")]

    rewards = []
    for episode_id in episodes:
        r = run_policy(env, get_action, test_set_params, args.len, episode_id, args.record, args.recording_path,
                       args.norender, use_baselines=student_type == 'baselines')
        rewards.append(r)
    env.close()
    return rewards

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--episode_ids', '-id', type=str, default="0")
    parser.add_argument('--bests', type=str2bool, default=None)
    parser.add_argument('--fixed_test_set', '-ts', type=str, default=None)
    parser.add_argument('--load_env', action='store_true')
    parser.add_argument('--use_test_env', action='store_true')

    parser.add_argument('--record', type=str2bool, default=False)
    parser.add_argument('--recording_path', type=str, default=None)
    EnvironmentArgsHandler.set_parser_arguments(parser)
    StudentArgsHandler.set_parser_arguments(parser)

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

