# Taken from https://github.com/psclklnk/spdl and wrapped in our architecture
# Modified by Clément Romac, copy of the license at TeachMyAgent/teachers/LICENSES/SPDL

import torch
import numpy as np
from copy import deepcopy
from functools import partial
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
from TeachMyAgent.teachers.utils.conjugate_gradient import cg_step
from TeachMyAgent.teachers.utils.torch import to_float_tensor
from TeachMyAgent.teachers.utils.gaussian_torch_distribution import GaussianTorchDistribution

class Buffer:

    def __init__(self, n_elements, max_buffer_size, reset_on_query):
        self.reset_on_query = reset_on_query
        self.max_buffer_size = max_buffer_size
        self.buffers = [list() for i in range(0, n_elements)]

    def update_buffer(self, datas):
        if isinstance(datas[0], list):
            for buffer, data in zip(self.buffers, datas):
                buffer.extend(data)
        else:
            for buffer, data in zip(self.buffers, datas):
                buffer.append(data)

        while len(self.buffers[0]) > self.max_buffer_size:
            for buffer in self.buffers:
                del buffer[0]

    def read_buffer(self, reset=None):
        if reset is None:
            reset = self.reset_on_query

        res = tuple([buffer for buffer in self.buffers])

        if reset:
            for i in range(0, len(self.buffers)):
                self.buffers[i] = []

        return res

    def __len__(self):
        return len(self.buffers[0])


class AbstractSelfPacedTeacher():

    def __init__(self, init_mean, flat_init_chol, target_mean, flat_target_chol, alpha_function, max_kl, cg_parameters):
        self.context_dist = GaussianTorchDistribution(init_mean, flat_init_chol, use_cuda=False)
        self.target_dist = GaussianTorchDistribution(target_mean, flat_target_chol, use_cuda=False)

        self.alpha_function = alpha_function
        self.max_kl = max_kl
        self.cg_parameters = {"n_epochs_line_search": 10, "n_epochs_cg": 10, "cg_damping": 1e-2,
                              "cg_residual_tol": 1e-10}
        if cg_parameters is not None:
            self.cg_parameters.update(cg_parameters)

        self.task = None
        self.iteration = 0

    def target_context_kl(self, numpy=True):
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist.distribution_t,
                                                      self.target_dist.distribution_t).detach()
        if numpy:
            kl_div = kl_div.numpy()

        return kl_div

    def save(self, path):
        weights = self.context_dist.get_weights()
        np.save(path, weights)

    def load(self, path):
        self.context_dist.set_weights(np.load(path))

    def _compute_context_kl(self, old_context_dist):
        return torch.distributions.kl.kl_divergence(old_context_dist.distribution_t, self.context_dist.distribution_t)

    def _compute_context_loss(self, cons_t, old_c_log_prob_t, c_val_t, alpha_cur_t):
        con_ratio_t = torch.exp(self.context_dist.log_pdf_t(cons_t) - old_c_log_prob_t)
        kl_div = torch.distributions.kl.kl_divergence(self.context_dist.distribution_t, self.target_dist.distribution_t)
        return torch.mean(con_ratio_t * c_val_t) - alpha_cur_t * kl_div


class SelfPacedTeacher(AbstractTeacher, AbstractSelfPacedTeacher):

    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, update_frequency, update_offset, alpha_function, initial_dist=None,
                 target_dist=None, max_kl=0.1, std_lower_bound=None, kl_threshold=None,  cg_parameters=None,
                 use_avg_performance=False, max_context_buffer_size=1000, reset_contexts=True, discount_factor=0.99):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)
        torch.manual_seed(self.seed)

        initial_mean, initial_variance = self.get_or_create_dist(initial_dist, mins, maxs, subspace=True) # Random subspace of the task space if no intial dist
        target_mean, target_variance = self.get_or_create_dist(target_dist, mins, maxs, subspace=False) # Full task space if no intial dist

        context_bounds = (np.array(mins), np.array(maxs))

        self.update_frequency = update_frequency
        self.update_offset = update_offset
        self.step_counter = 0
        self.discounted_sum_reward = 0
        self.discount_factor = discount_factor
        self.discounted_sum_rewards = []
        self.current_disc = 1
        self.pending_initial_state = None
        self.algorithm_iterations = 0

        # The bounds that we show to the outside are limited to the interval [-1, 1], as this is typically better for
        # neural nets to deal with
        self.context_buffer = Buffer(2, max_context_buffer_size, reset_contexts)
        self.context_dim = target_mean.shape[0]
        self.context_bounds = context_bounds
        self.use_avg_performance = use_avg_performance

        if std_lower_bound is not None and kl_threshold is None:
            raise RuntimeError("Error! Both Lower Bound on standard deviation and kl threshold need to be set")
        else:
            if std_lower_bound is not None:
                if isinstance(std_lower_bound, np.ndarray):
                    if std_lower_bound.shape[0] != self.context_dim:
                        raise RuntimeError("Error! Wrong dimension of the standard deviation lower bound")
                elif std_lower_bound is not None:
                    std_lower_bound = np.ones(self.context_dim) * std_lower_bound
            self.std_lower_bound = std_lower_bound
            self.kl_threshold = kl_threshold

        # Create the initial context distribution
        if isinstance(initial_variance, np.ndarray):
            flat_init_chol = GaussianTorchDistribution.flatten_matrix(initial_variance, tril=False)
        else:
            flat_init_chol = GaussianTorchDistribution.flatten_matrix(initial_variance * np.eye(self.context_dim),
                                                                      tril=False)

        # Create the target distribution
        if isinstance(target_variance, np.ndarray):
            flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance, tril=False)
        else:
            flat_target_chol = GaussianTorchDistribution.flatten_matrix(target_variance * np.eye(self.context_dim),
                                                                        tril=False)

        AbstractSelfPacedTeacher.__init__(self, initial_mean, flat_init_chol, target_mean, flat_target_chol,
                                               alpha_function, max_kl, cg_parameters)
        self.bk = {'mean': [],
                   'covariance': [],
                   'steps': [],
                   'algo_iterations': [],
                   'kl': []}

    def record_initial_state(self, task, initial_state):
        self.pending_initial_state = initial_state

    def episodic_update(self, task, reward, is_success):
        assert self.pending_initial_state is not None

        self.discounted_sum_rewards.append(self.discounted_sum_reward)
        self.context_buffer.update_buffer((self.pending_initial_state, task))
        self.discounted_sum_reward = 0
        self.current_disc = 1
        self.pending_initial_state = None

    def step_update(self, state, action, reward, next_state, done):
        self.step_counter += 1
        self.discounted_sum_reward += self.current_disc * reward
        self.current_disc *= self.discount_factor

        if self.step_counter >= self.update_offset and self.step_counter % self.update_frequency == 0:
            if len(self.discounted_sum_rewards) > 0 and len(self.context_buffer) > 0:
                self.algorithm_iterations += 1
                avg_performance = np.mean(self.discounted_sum_rewards)
                self.discounted_sum_rewards = []

                ins, cons = self.context_buffer.read_buffer()
                initial_states, contexts = np.array(ins), np.array(cons)
                values = self.value_estimator(initial_states)
                if values is None:
                    raise Exception("Please define a valid value estimator, this one returns None...")

                old_context_dist = deepcopy(self.context_dist)
                contexts_t = to_float_tensor(contexts, use_cuda=False)
                old_c_log_prob_t = old_context_dist.log_pdf_t(contexts_t).detach()

                # Estimate the value of the state after the policy update
                c_val_t = to_float_tensor(values, use_cuda=False)

                # Add the penalty term
                cur_kl_t = self.target_context_kl(numpy=False)
                if self.use_avg_performance:
                    alpha_cur_t = self.alpha_function(self.algorithm_iterations, avg_performance, cur_kl_t)
                else:
                    alpha_cur_t = self.alpha_function(self.algorithm_iterations, torch.mean(c_val_t).detach(), cur_kl_t)

                cg_step(partial(self._compute_context_loss, contexts_t, old_c_log_prob_t, c_val_t, alpha_cur_t),
                        partial(self._compute_context_kl, old_context_dist), self.max_kl,
                        self.context_dist.parameters, self.context_dist.set_weights,
                        self.context_dist.get_weights, **self.cg_parameters, use_cuda=False)

                cov = self.context_dist._chol_flat.detach().numpy()
                if self.std_lower_bound is not None and self.target_context_kl() > self.kl_threshold:
                    cov[0:self.context_dim] = np.log(np.maximum(np.exp(cov[0:self.context_dim]), self.std_lower_bound))
                    self.context_dist.set_weights(np.concatenate((self.context_dist.mean(), cov)))
                self.bk["mean"].append(self.context_dist.mean())
                self.bk["covariance"].append(self.context_dist.covariance_matrix())
                self.bk["steps"].append(self.step_counter)
                self.bk["algo_iterations"].append(self.algorithm_iterations)
                self.bk["kl"].append(self.target_context_kl())
            else:
                print("Skipping iteration at step {} because buffers are empty.".format(self.step_counter))


    def sample_task(self):
        sample = self.context_dist.sample().detach().numpy()
        return np.clip(sample, self.context_bounds[0], self.context_bounds[1], dtype=np.float32)

    def non_exploratory_task_sampling(self):
        return {"task": self.sample_task(),
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": None}
                }

