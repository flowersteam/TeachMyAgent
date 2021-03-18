# The following code was obtained from https://github.com/florensacc/rllab-curriculum
# and only slightly modified to fit the project structure. The original license of
# the software is the following:

# The MIT License (MIT)

# Copyright (c) 2016 rllab contributors

# rllab uses a shared copyright model: each contributor holds copyright over
# their contributions to rllab. The project versioning records all such
# contribution and copyright details.
# By contributing to the rllab repository through pull-request, comment,
# or otherwise, the contributor releases their content to the license and
# copyright terms herein.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from TeachMyAgent.teachers.utils.gan.gan import FCGAN
import multiprocessing
import scipy.misc


def sample_matrix_row(M, size, replace=False):
    if size > M.shape[0]:
        return M
    if replace:
        indices = np.random.randint(0, M.shape[0], size)
    else:
        indices = np.random.choice(M.shape[0], size, replace=replace)
    return M[indices, :]


class StateGenerator(object):
    """A base class for state generation."""

    def pretrain_uniform(self):
        """Pretrain the generator distribution to uniform distribution in the limit."""
        raise NotImplementedError

    def pretrain(self, states):
        """Pretrain with state distribution in the states list."""
        raise NotImplementedError

    def sample_states(self, size):
        """Sample states with given size."""
        raise NotImplementedError

    def sample_states_with_noise(self, size):
        """Sample states with noise."""
        raise NotImplementedError

    def train(self, states, labels, outer_iters, generator_iters=None, discriminator_iters=None):
        """Train with respect to given states and labels."""
        raise NotImplementedError


class CrossEntropyStateGenerator(StateGenerator):
    """Maintain a state list and add noise to current states to generate new states."""

    def __init__(self, state_size, state_range, noise_std=1.0,
                 state_center=None):
        self.state_list = np.array([])
        self.state_range = state_range
        self.noise_std = noise_std
        self.state_center = np.array(state_center) if state_center is not None else np.zeros(state_size)

    def pretrain_uniform(self, size=1000):
        states = self.state_center + np.random.uniform(
            -self.state_range, self.state_range, size=(size, self.state_size)
        )
        return self.pretrain(states)

    def pretrain(self, states):
        self.state_list = np.array(states)

    def sample_states(self, size):
        if len(self.state_list) == 0:
            raise ValueError('Generator uninitialized!')

        states = sample_matrix_row(self.state_list, size)
        return np.clip(
            states + np.random.randn(*states.shape) * self.noise_std,
            self.state_center - self.state_range, self.state_center + self.state_range
        )

    def sample_states_with_noise(self, size):
        return self.sample_states(size)

    def train(self, states, labels, outer_iters, generator_iters=None, discriminator_iters=None):
        labels = np.mean(labels, axis=1) >= 1
        good_states = np.array(states)[labels, :]
        if len(good_states) != 0:
            self.state_list = good_states


class StateGAN(StateGenerator):
    """A GAN for generating states. It is just a wrapper for clgan.GAN.FCGAN"""

    def __init__(self, state_size, evaluater_size,
                 state_noise_level, state_range=None, state_center=None, state_bounds=None, *args, **kwargs):
        self.gan = FCGAN(
            generator_output_size=state_size,
            discriminator_output_size=evaluater_size,
            *args,
            **kwargs
        )
        self.state_size = state_size
        self.evaluater_size = evaluater_size
        self.state_center = np.array(state_center) if state_center is not None else np.zeros(state_size)
        if state_range is not None:
            self.state_range = state_range
            self.state_bounds = np.vstack(
                [-self.state_range * np.ones(self.state_size), self.state_range * np.ones(self.state_size)])
        elif state_bounds is not None:
            self.state_bounds = np.array(state_bounds)
            self.state_range = self.state_bounds[1] - self.state_bounds[0]
        self.state_noise_level = state_noise_level
        print('state_center is : ', self.state_center, 'state_range: ', self.state_range,
              'state_bounds: ', self.state_bounds)

    def pretrain_uniform(self, size=10000, report=None, *args, **kwargs):
        """
        :param size: number of uniformly sampled states (that we will try to fit as output of the GAN)
        :param outer_iters: of the GAN
        """
        states = np.random.uniform(
            self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1],
            size=(size, self.state_size)
        )
        return self.pretrain(states, *args, **kwargs)

    def pretrain(self, states, outer_iters=500, generator_iters=None, discriminator_iters=None):
        """
        Pretrain the state GAN to match the distribution of given states.
        :param states: the state distribution to match
        :param outer_iters: of the GAN
        """
        labels = np.ones((states.shape[0], self.evaluater_size))  # all state same label --> uniform
        return self.train(
            states, labels, outer_iters, generator_iters, discriminator_iters
        )

    def _add_noise_to_states(self, states):
        noise = np.random.randn(*states.shape) * self.state_noise_level
        states += noise
        return np.clip(states, self.state_center + self.state_bounds[0], self.state_center + self.state_bounds[1])

    def sample_states(self, size):  # un-normalizes the states
        normalized_states, noise = self.gan.sample_generator(size)
        states = self.state_center + normalized_states * (self.state_bounds[1] - self.state_bounds[0])
        return states, noise

    def sample_states_with_noise(self, size):
        states, noise = self.sample_states(size)
        states = self._add_noise_to_states(states)
        return states, noise

    def train(self, states, labels, outer_iters, generator_iters=None, discriminator_iters=None):
        normalized_states = (states - self.state_center) / (self.state_bounds[1] - self.state_bounds[0])
        return self.gan.train(
            normalized_states, labels, outer_iters, generator_iters, discriminator_iters
        )

    def discriminator_predict(self, states):
        return self.gan.discriminator_predict(states)


class StateCollection(object):
    """ A collection of states, with minimum distance threshold for new states. """

    def __init__(self, n_processes, distance_threshold=None, states_transform=None, idx_lim=None):
        self.distance_threshold = distance_threshold
        self.state_list = []
        self.states_transform = states_transform
        self.idx_lim = idx_lim
        self.pool = multiprocessing.Pool(n_processes)
        self.n_processes = n_processes
        if self.states_transform:
            self.transformed_state_list = []

    @property
    def size(self):
        return len(self.state_list)

    def empty(self):
        self.state_list = []

    def sample(self, size, replace=False, replay_noise=0):
        states = sample_matrix_row(np.array(self.state_list), size, replace)
        if len(states) != 0:
            states += replay_noise[None, :] * np.random.randn(*states.shape)
        return states

    def append(self, states):
        if self.states_transform:
            return self.append_states_transform(states)
        if len(states) > 0:
            states = np.array(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states = self._process_states(states)
            if states.shape[0] >= self.n_processes > 1:
                states_per_process = states.shape[0] // self.n_processes
                list_of_states = [states[i * states_per_process: (i + 1) * states_per_process, :] for i in
                                  range(self.n_processes)]
                states = self.pool.map(SelectStatesWrapper(self.idx_lim, self.state_list, self.distance_threshold),
                                       list_of_states)
                states = np.concatenate(states)
            else:
                states = self._select_states(states)
            self.state_list.extend(states.tolist())
            return states

    def _select_states(self, states):
        # print('selecting states from shape: ', states.shape)
        selected_states = states
        selected_states_idx_lim = np.array([state[:self.idx_lim] for state in states])
        # print('selecting states from shape (after idx_lim of ', self.idx_lim, ': ', selected_states_idx_lim.shape)
        state_list_idx_lim = np.array([state[:self.idx_lim] for state in self.state_list])
        # print('the state_list_idx_lim shape: ', np.shape(self.state_list))
        if self.distance_threshold is not None and self.distance_threshold > 0:
            if len(self.state_list) > 0:
                dists = scipy.spatial.distance.cdist(state_list_idx_lim, selected_states_idx_lim)
                indices = np.amin(dists, axis=0) > self.distance_threshold
                selected_states = selected_states[indices, :]
        # print('the selected states are: {}'.format(selected_states.shape))
        return selected_states

    def _process_states(self, states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        states = np.array(states)
        results = [states[0]]
        results_idx_lim = [states[0][:self.idx_lim]]
        for i, state in enumerate(states[1:]):
            # print("analyzing state : ", i)
            if np.amin(scipy.spatial.distance.cdist(results_idx_lim,
                                                    state.reshape(1, -1)[:self.idx_lim])) > self.distance_threshold:
                results.append(state)
                results_idx_lim.append(state[:self.idx_lim])
        return np.array(results)

    def _process_states_transform(self, states, transformed_states):
        "keep only the states that are at more than dist_threshold from each other"
        # adding a states transform allows you to maintain full state information while possibly disregarding some dim
        results = [states[0]]
        transformed_results = [transformed_states[0]]
        for i in range(1, len(states)):
            # checks if valid in transformed space
            if np.amin(scipy.spatial.distance.cdist(transformed_results,
                                                    transformed_states[i].reshape(1, -1))) > self.distance_threshold:
                results.append(states[i])
                transformed_results.append(transformed_states[i])
        return np.array(results), np.array(transformed_results)

    def append_states_transform(self, states):
        assert self.idx_lim is None, "Can't use state transform and idx_lim with StateCollection!"
        if len(states) > 0:
            states = np.array(states)
            transformed_states = self.states_transform(states)
            if self.distance_threshold is not None and self.distance_threshold > 0:
                states, transformed_states = self._process_states_transform(states, transformed_states)
                if len(self.state_list) > 0:
                    print("hi")
                    dists = scipy.spatial.distance.cdist(self.transformed_state_list, transformed_states)
                    indices = np.amin(dists, axis=0) > self.distance_threshold
                    states = states[indices, :]
                    transformed_states = transformed_states[indices, :]
            self.state_list.extend(states)
            self.transformed_state_list.extend(transformed_states)
            assert (len(self.state_list) == len(self.transformed_state_list))
        return states  # modifed to return added states

    # def append(self, states):
    #     if self.states_transform:
    #         return self.append_states_transform(states)
    #     if len(states) > 0:
    #         states = np.array(states)
    #         if self.distance_threshold is not None and self.distance_threshold > 0:
    #             states = self._process_states(states)
    #             if len(self.state_list) > 0:
    #                 dists = scipy.spatial.distance.cdist(self.state_list, states)
    #                 indices = np.amin(dists, axis=0) > self.distance_threshold
    #                 states = states[indices, :]
    #         self.state_list.extend(states)
    #     return states # modifed to return added states

    @property
    def states(self):
        return np.array(self.state_list)


class SelectStatesWrapper:

    def __init__(self, idx_lim, state_list, distance_threshold):
        self.idx_lim = idx_lim
        self.state_list = state_list
        self.distance_threshold = distance_threshold

    def __call__(self, states):
        # print('selecting states from shape: ', states.shape)
        selected_states = states
        selected_states_idx_lim = np.array([state[:self.idx_lim] for state in states])
        # print('selecting states from shape (after idx_lim of ', self.idx_lim, ': ', selected_states_idx_lim.shape)
        state_list_idx_lim = np.array([state[:self.idx_lim] for state in self.state_list])
        # print('the state_list_idx_lim shape: ', np.shape(self.state_list))
        if self.distance_threshold is not None and self.distance_threshold > 0:
            if len(self.state_list) > 0:
                dists = scipy.spatial.distance.cdist(state_list_idx_lim, selected_states_idx_lim)
                indices = np.amin(dists, axis=0) > self.distance_threshold
                selected_states = selected_states[indices, :]
        # print('the selected states are: {}'.format(selected_states.shape))
        return selected_states
