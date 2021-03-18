import gym
import TeachMyAgent.environments
import time
import numpy as np
import sys
import os

from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum

from gym.wrappers.monitoring.video_recorder import VideoRecorder

##### USE THIS FILE TO DEBUG YOUR EMBODIMENT WITH SEQUENCES OF ACTIONS #####

debug_folder = "XXX\DebugMorphology\MorphologyDebugSequences"

def get_full_debug_sequence(nb_of_motors):
    sequence = []
    sequence.append(np.zeros(nb_of_motors))
    for i in range(nb_of_motors):
        positive_action = np.array([1 if id == i else 0 for id in range(nb_of_motors)])
        negative_action = np.array([-1 if id == i else 0 for id in range(nb_of_motors)])
        sequence.append(positive_action)
        sequence.append(negative_action)
        sequence.append(positive_action)
        sequence.append(negative_action)

    return sequence

def get_no_actions_debug_sequence(nb_of_motors):
    sequence = []
    sequence.append(np.zeros(nb_of_motors))

    return sequence

def perform_debug_sequence(sequence_name, env, walker_type, sequence_of_actions, action_repeat = 10, has_gravity = False):
    video_folder = debug_folder + "\\" + walker_type
    if not os.path.exists(video_folder):
        os.makedirs(video_folder)

    if not has_gravity:
        env.world.gravity = (0, 0)
    else:
        env.world.gravity = (0, -10)

    video_recorder = VideoRecorder(env, video_folder + "\\" + sequence_name +".mp4",)  # Stump Tracks
    env.reset()

    for action in sequence_of_actions:
        for i in range(action_repeat):
            _, _, d, _ = env.step(action)
            video_recorder.capture_frame()
            time.sleep(0.01)
            video_recorder.capture_frame()
    video_recorder.close()

def main():
    if len(sys.argv) < 2: return
    if sys.argv[1] == "all":
        walkers_to_debug = [body.name for body in BodiesEnum]
    else:
        walkers_to_debug = [sys.argv[1]]

    for walker_type in walkers_to_debug:
        env = gym.make('parametric-continuous-walker-v0', walker_type=walker_type)
        action_space = env.action_space

        env.set_environment(stump_height=0, obstacle_spacing=0)

        perform_debug_sequence("no_actions_no_gravity",
                               env,
                               walker_type,
                               get_no_actions_debug_sequence(action_space.shape[0]),
                               action_repeat=50)

        perform_debug_sequence("no_actions",
                               env,
                               walker_type,
                               get_no_actions_debug_sequence(action_space.shape[0]),
                               action_repeat=50,
                               has_gravity=True)

        perform_debug_sequence("debug_actions_no_gravity",
                               env,
                               walker_type,
                               get_full_debug_sequence(action_space.shape[0]))

        perform_debug_sequence("debug_actions",
                               env,
                               walker_type,
                               get_full_debug_sequence(action_space.shape[0]),
                               has_gravity=True)

if __name__ == "__main__":
    # execute only if run as a script
    main()