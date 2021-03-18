# Automatic Domain Randomization, see https://arxiv.org/abs/1910.07113 for details
# Implemented by Rémy Portelas and Clément Romac

import numpy as np
from gym.spaces import Box
from collections import deque
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher

class ADR(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, step_size, max_reward_thr, min_reward_thr,
                 initial_dist=None, boundary_sampling_p=0.5, queue_len=10, scale_reward=False):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)
        self.nb_dims = len(self.mins)

        # Boundary sampling probability p_r
        self.bound_sampling_p = boundary_sampling_p

        # ADR step size
        self.step_size = step_size

        # Max reward threshold, sampling distribution inflates if mean reward above this
        self.max_reward_threshold = max_reward_thr
        if scale_reward:
            self.max_reward_threshold = np.interp(self.max_reward_threshold,
                                                  (self.env_reward_lb, self.env_reward_ub),
                                                  (0, 1))

        # Min reward threshold, sampling distribution deflates if mean reward below this
        self.min_reward_threshold = min_reward_thr
        if scale_reward:
            self.min_reward_threshold = np.interp(self.min_reward_threshold,
                                                  (self.env_reward_lb, self.env_reward_ub),
                                                  (0, 1))

        # max queue length
        self.window_len = queue_len

        # Set initial task space to predefined calibrated task
        initial_mean, initial_variance = self.get_or_create_dist(initial_dist, mins, maxs, subspace=True)

        # Single task version (as the original paper)
        self.cur_mins = initial_mean
        self.cur_maxs = initial_mean

        self.cur_mins = np.array(self.cur_mins, dtype=np.float32)  # current min bounds
        self.cur_maxs = np.array(self.cur_maxs, dtype=np.float32)  # current max bounds
        self.task_space = Box(self.cur_mins, self.cur_maxs, dtype=np.float32)
        self.task_space.seed(self.seed)

        # Init queues, one per task space dimension
        self.min_queues = [deque(maxlen=self.window_len) for _ in range(self.nb_dims)]
        self.max_queues = [deque(maxlen=self.window_len) for _ in range(self.nb_dims)]

        # Boring book-keeping
        self.episode_nb = 0
        self.bk = {'task_space': [(self.cur_mins.copy(),self.cur_maxs.copy())],
                   'episodes': []}

    def episodic_update(self, task, reward, is_success):
        self.episode_nb += 1

        # check for updates
        for i, (min_q, max_q, cur_min, cur_max) in enumerate(zip(self.min_queues, self.max_queues, self.cur_mins, self.cur_maxs)):
            if task[i] == cur_min:  # if the proposed task has the i^th dimension set to min boundary
                min_q.append(reward)
                if len(min_q) == self.window_len:
                    if np.mean(min_q) >= self.max_reward_threshold:  # decrease min boundary (inflate sampling space)
                        self.cur_mins[i] = max(self.cur_mins[i] - self.step_size, self.mins[i])
                    elif np.mean(min_q) <= self.min_reward_threshold:  # increase min boundary (deflate sampling space)
                        self.cur_mins[i] = min(self.cur_mins[i] + self.step_size, self.cur_maxs[i])
                    self.min_queues[i] = deque(maxlen=self.window_len)  # reset queue
            if task[i] == cur_max:  # if the proposed task has the i^th dimension set to max boundary
                max_q.append(reward)
                if len(max_q) == self.window_len:  # queue is full, time to update
                    if np.mean(max_q) >= self.max_reward_threshold:  # increase max boundary
                        self.cur_maxs[i] = min(self.cur_maxs[i] + self.step_size, self.maxs[i])
                    elif np.mean(max_q) <= self.min_reward_threshold:  # decrease max boundary
                        self.cur_maxs[i] = max(self.cur_maxs[i] - self.step_size, self.cur_mins[i])
                    self.max_queues[i] = deque(maxlen=self.window_len)  # reset queue

        prev_cur_mins, prev_cur_maxs = self.bk['task_space'][-1]
        if (prev_cur_mins != self.cur_mins).any() or (prev_cur_maxs != self.cur_maxs).any():  # were boundaries changed ?
            self.task_space = Box(self.cur_mins, self.cur_maxs, dtype=np.float32)
            self.task_space.seed(self.seed)
            # book-keeping only if boundaries were updates
            self.bk['task_space'].append((self.cur_mins.copy(), self.cur_maxs.copy()))
            self.bk['episodes'].append(self.episode_nb)

    def sample_task(self):
        new_task = self.non_exploratory_task_sampling()["task"]
        if self.random_state.random() < self.bound_sampling_p:  # set random dimension to min or max bound
            idx = self.random_state.randint(0, self.nb_dims)
            is_min_max_capped = np.array([self.cur_mins[idx] == self.mins[idx], self.cur_maxs[idx] == self.maxs[idx]])
            if not is_min_max_capped.all():  # both min and max bounds can increase, choose extremum randomly
                if self.random_state.random() < 0.5:  # skip min bound if already
                    new_task[idx] = self.cur_mins[idx]
                else:
                    new_task[idx] = self.cur_maxs[idx]
            elif not is_min_max_capped[0]:
                new_task[idx] = self.cur_mins[idx]
            elif not is_min_max_capped[1]:
                new_task[idx] = self.cur_maxs[idx]
        return new_task

    def non_exploratory_task_sampling(self):
        return {"task": self.task_space.sample(),
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": None}
                }