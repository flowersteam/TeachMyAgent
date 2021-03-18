# Implemented by Clément Romac following https://arxiv.org/abs/1909.12892
# with the help of https://drive.google.com/drive/folders/1yjhztFeX67tHEImXCiP_UAQfQ-wFvV4Y

import numpy as np
import tensorflow as tf
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
from TeachMyAgent.teachers.utils.setter_solver_utils import Judge, FlatRnvp, ClippedSigmoid

class GoalBuffer(object):
    def __init__(self):
        self.buffer = {}

    def add(self, key, goal, feasibility, reward):
        if key in self.buffer:
            self.buffer[key].append({
                "goal": goal,
                "feasibility": feasibility,
                "return": reward
            })
        else:
            self.buffer[key] = [{
                "goal": goal,
                "feasibility": feasibility,
                "return": reward
            }]

    def read(self, reset=True):
        goals = []
        feasibilities = []
        returns = []

        for key in self.buffer:
            for episode in self.buffer[key]:
                goals.append(episode["goal"])
                feasibilities.append(episode["feasibility"])
                returns.append(episode["return"])

        if reset:
            self.buffer = {}

        return np.array(goals), np.array(feasibilities), np.array(returns)

class SetterSolver(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, update_frequency=100,
                 setter_loss_noise_ub=0.01, setter_hidden_size=128):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)
        self.nb_dims = len(self.mins)

        tf.set_random_seed(
            seed
        )
        tf_config = tf.ConfigProto()
        # Prevent tensorflow from taking all the gpu memory
        tf_config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(config=tf_config)

        self.update_frequency = update_frequency
        self.episode_counter = 0

        self.judge = Judge(hidden_sizes=[64, 64, 64],
                           tf_session=self.tf_session,
                           goal_size=self.nb_dims)
        self.setter = FlatRnvp(
            latent_size=self.nb_dims,
            num_blocks=3,
            num_layers_per_block=3,
            tf_session=self.tf_session,
            judge_output_op=self.judge._mlp,
            hidden_size=setter_hidden_size,
            final_non_linearity=ClippedSigmoid,
            loss_noise_ub=setter_loss_noise_ub,
            random_state=self.random_state
        )

        self.tf_session.run(tf.global_variables_initializer())

        self.goal_buffer = GoalBuffer()
        self.pending_goals = []
        self.bk = {'judge_loss': [],
                   'setter_loss': [],
                   'episodes': []}


    def episodic_update(self, task, reward, is_success):
        self.episode_counter += 1

        # Add goal and associated infos to buffer
        goal_key = task.tobytes()
        last_sampled_goal = self.pending_goals[len(self.pending_goals) - 1]
        is_success = int(is_success)
        if last_sampled_goal[0] == goal_key:
            self.goal_buffer.add(goal_key, last_sampled_goal[1], last_sampled_goal[2], is_success)
        else:
            raise Exception("Wrong goal")

        self.pending_goals = []

        if self.episode_counter % self.update_frequency == 0:
            samples, feasibilities, returns = self.goal_buffer.read()
            feasibilities = feasibilities.reshape((len(feasibilities), 1))
            returns = returns.reshape((len(returns), 1))

            judge_loss, _ = self.judge.train(returns, samples)
            setter_loss, _ = self.setter.train(samples, feasibilities, returns)
            self.bk["judge_loss"].append(judge_loss)
            self.bk["setter_loss"].append(setter_loss)
            self.bk["episodes"].append(self.episode_counter)


    def sample_task(self):
        # Sample a feasibility and and associated goal
        feasibility = self.random_state.uniform(low=1e-8, high=1.0, size=[1, 1])
        sample, log_p = self.setter.sample(1, condition=feasibility)
        new_task = sample[0]

        # Interpolate the goal that is in [0, 1] to the real task space
        rescaled_task = np.float32(self.rescale_task(new_task))

        # Store the goal and feasibility in the buffer
        task_key = rescaled_task.tobytes()
        self.pending_goals.append((task_key, new_task, feasibility))

        return rescaled_task

    def non_exploratory_task_sampling(self):
        feasibility = self.random_state.uniform(low=1e-8, high=1.0, size=[1, 1])
        sample, log_p = self.setter.sample(1, condition=feasibility)
        return {"task": np.float32(self.rescale_task(sample[0])),
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": feasibility}
                }