# Implemented by Pascal Klink and modified by Clément Romac

import emcee
import numpy as np
import torch
import scipy.linalg as scpla
from ACL_bench.teachers.algos.AbstractTeacher import AbstractTeacher
from ACL_bench.teachers.utils.gaussian_torch_distribution import GaussianTorchDistribution
from scipy.optimize import minimize, NonlinearConstraint, Bounds
from scipy.stats import multivariate_normal
import itertools

LOG_LEVEL = "warning" # no / warning / all

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

class DummyGaussian:

    def __init__(self, mean, sigma_inv):
        self._mean = mean
        self._sigma = scpla.cho_solve(scpla.cho_factor(sigma_inv), np.eye(mean.shape[0]))

    def mean(self):
        return self._mean.copy()

    def covariance_matrix(self):
        return self._sigma.copy()

    def get_weights(self):
        return np.concatenate((self._mean, self._sigma.reshape((-1,))))

    def set_weights(self, raw_array):
        self._mean = raw_array[0:self._mean.shape[0]]
        self._sigma = np.reshape(raw_array[self._mean.shape[0]:], self._sigma.shape)


class AbstractSelfPacedTeacher:

    def __init__(self, init_mean, init_sigma, bounds, alpha_function, max_kl, samples_per_iter=300, target_mean=None,
                 target_sigma=None, use_rejection_sampling=True):
        self.mean = init_mean
        self.sigma_inv = scpla.cho_solve(scpla.cho_factor(init_sigma), np.eye(init_mean.shape[0]))
        self.context_dist = DummyGaussian(self.mean, self.sigma_inv)
        self.bounds = bounds
        self.use_rejection_sampling = use_rejection_sampling

        self.target_mean = target_mean
        if target_sigma is None:
            self.target_sigma_inv = None
        else:
            self.target_sigma_inv = scpla.cho_solve(scpla.cho_factor(target_sigma), np.eye(init_mean.shape[0]))

        self.alpha_function = alpha_function
        self.max_kl = max_kl

        self.iteration = 0

        # We generate initial samples with emcee that will be improved upon with each iteration
        self.act = None
        self.ensemble_size = 32
        self.min_mcmc_steps = int(np.ceil(samples_per_iter / self.ensemble_size))

        self.sample_trunc()
        self.update_internal_values()

    @staticmethod
    def lme(x):
        x_max = torch.max(x)
        log_n = np.log(np.prod(x.shape))
        return torch.log(torch.sum(torch.exp(x - x_max))) + x_max - log_n

    @staticmethod
    def _unnormalized_log_pdf(x, mu, sigma_inv):
        if isinstance(x, torch.Tensor):
            pred = torch
            x = x.float()
            mu = mu.float()
            sigma_inv = sigma_inv.float()
        else:
            pred = np

        diffs = x - mu
        return -0.5 * pred.einsum("...k,kl,...l->...", diffs, sigma_inv, diffs)

    def unnormalized_log_pdf(self, x):
        return self._unnormalized_log_pdf(x, self.mean, self.sigma_inv)

    def unnormalized_target_log_pdf(self, x):
        if self.target_mean is None or self.target_sigma_inv is None:
            return np.zeros_like(x[..., 0])
        else:
            return self._unnormalized_log_pdf(x, self.target_mean, self.target_sigma_inv)

    def unnormalized_pdf(self, x):
        return np.exp(self.unnormalized_log_pdf(x))

    def sample_trunc(self):
        sigma = scpla.cho_solve(scpla.cho_factor(self.sigma_inv), np.eye(self.mean.shape[0]))
        # sigma = np.diag(np.diag(sigma))
        rejection_sampling = False

        if self.use_rejection_sampling:
            cdf_u = multivariate_normal.cdf(self.bounds[1], self.mean, sigma, allow_singular=True)
            ### THIS IS NOT WORKING ###
            cdf_l = [multivariate_normal.cdf(self.bounds[0], self.mean, sigma, allow_singular=True)]

            # As long as there is a 5 percent chance that a sample from a regular normal is in the bounds, 200 rejection
            # sampling iterations have only a chance of 0.003% of failing
            if cdf_u - sum(cdf_l) >= 0.05:
                rejection_sampling = True

        if rejection_sampling:
            if LOG_LEVEL == "all":
                print("Rejection sampling")
            samples = self.sample_rejection(self.mean, sigma, self.bounds,
                                            self.min_mcmc_steps * self.ensemble_size,
                                            rejection_iters=200)

            self.act = None
            self.context_samples = np.reshape(samples, (self.min_mcmc_steps, self.ensemble_size, -1))
        else:
            if LOG_LEVEL == "all":
                print("Sampling with MCMC")
            burn_in = 500
            n_step = 100000
            if self.act is not None:
                burn_in = 0
                n_step = max(self.min_mcmc_steps + 1, 200 * self.act)

            chain, self.act = self.sample_mcmc(self.unnormalized_log_pdf, self.bounds, mcmc_burn_in=burn_in,
                                               mcmc_n_steps=n_step, ensemble_size=self.ensemble_size)

            self.context_samples = chain[min(-self.min_mcmc_steps, int(-2 * self.act)):, :, :]

        self.context_samples_torch = torch.from_numpy(self.context_samples)

        # Generate a random permutation for sampling
        self.idxs = np.random.permutation(np.arange(np.prod(self.context_samples.shape[0:2])))
        self.sample_count = 0

    def update_internal_values(self):
        # We also pre-compute the unnormalized log-pdfs and pdfs over q_0 for the current- and target distribution
        self.log_q0 = torch.from_numpy(self.unnormalized_log_pdf(self.context_samples))
        self.log_qmu = torch.from_numpy(self.unnormalized_target_log_pdf(self.context_samples))
        self.log_zmu_z0 = self.lme(self.log_qmu - self.log_q0)

        self.target_kl = self._target_context_kl(self.log_q0, torch.ones_like(self.log_q0), 0.)

    @staticmethod
    def sample_rejection(mu, sigma, bounds, n_samples=100, rejection_iters=20):
        # Initialize the MCMC sampler with a few iterations of rejection sampling
        count = 0
        rejection_success = False

        p0 = np.random.multivariate_normal(mu, sigma, size=n_samples)
        while not rejection_success and count < rejection_iters:
            violations = np.logical_or(np.any(p0 < bounds[0], axis=-1), np.any(p0 > bounds[1], axis=-1))
            rejection_success = not np.any(violations)
            if not rejection_success:
                p0[violations] = np.random.multivariate_normal(mu, sigma, size=np.sum(violations))
            count += 1

        if not rejection_success and LOG_LEVEL != "no":
            print("Warning! Rejection sampling not successful! {} samples clipped over {}!"
                  .format(np.sum(violations), n_samples))

        return np.clip(p0, bounds[0], bounds[1])

    @staticmethod
    def sample_mcmc(log_pdf_fn, bounds, ensemble_size=32, mcmc_burn_in=500, mcmc_n_steps=10000):
        # Now run the MCMC sampler
        def log_pdf(ps):
            violations = np.logical_or(np.any(ps < bounds[0], axis=-1), np.any(ps > bounds[1], axis=-1))
            log_pdfs = log_pdf_fn(ps)
            log_pdfs[violations] = -np.inf
            return log_pdfs

        def proposal_fun(x0, rng):
            return rng.uniform(bounds[0], bounds[1], x0.shape), np.zeros(x0.shape[0])

        sampler = emcee.EnsembleSampler(ensemble_size, ndim=bounds[0].shape[0], log_prob_fn=log_pdf, vectorize=True,
                                        moves=[emcee.moves.MHMove(proposal_fun)])
        p0 = proposal_fun(np.ones((ensemble_size, bounds[0].shape[0])), np.random)[0]
        if mcmc_burn_in > 0:
            state = sampler.run_mcmc(p0, mcmc_burn_in)
            sampler.reset()
        else:
            state = p0

        sampler.run_mcmc(state, mcmc_n_steps)
        info = (np.mean(sampler.acceptance_fraction), np.mean(sampler.get_autocorr_time()))
        if LOG_LEVEL == "all":
            print("MCMC sampling completed - mean acceptance fraction / autocorrelation time: %.3f / %.3f" % info)

        return sampler.get_chain(), info[1]

    def _target_context_kl(self, log_q1, weights, log_z1_z0):
        return torch.mean(weights * (log_q1 - self.log_qmu)) + self.log_zmu_z0 - log_z1_z0

    def target_context_kl(self, mu=None, chol_inv=None):
        if mu is None:
            mu = torch.from_numpy(self.mean)

        if chol_inv is None:
            sigma_inv = torch.from_numpy(self.sigma_inv)
        else:
            sigma_inv = torch.einsum("ij,kj->ik", chol_inv, chol_inv)
        log_q1 = self._unnormalized_log_pdf(self.context_samples_torch, mu, sigma_inv)

        # Numerically stable computation of torch.log(z1_z0), necessary to avoid a false estimate of KL being zero for
        # very dissimilar distributions
        log_z1_z0 = self.lme(log_q1 - self.log_q0)

        # weights = q1_q0 / z1_z0 = exp(log_q1 - log_q0 - log-z1_z0)
        weights = torch.exp(log_q1 - self.log_q0 - log_z1_z0)
        return self._target_context_kl(log_q1, weights, log_z1_z0)

    def kl(self, mean, chol_inv):
        sigma_inv = torch.einsum("ij,kj->ik", chol_inv, chol_inv)
        log_q1 = self._unnormalized_log_pdf(self.context_samples_torch, mean, sigma_inv)

        # Numerically stable computation of torch.log(z1_z0), necessary to avoid a false estimate of KL being zero for
        # very dissimilar distributions
        log_z1_z0 = self.lme(log_q1 - self.log_q0)

        # weights = q1_q0 / z1_z0 = exp(log_q1 - log_q0 - log-z1_z0)
        weights = torch.exp(log_q1 - self.log_q0 - log_z1_z0)
        return torch.mean(weights * (log_q1 - self.log_q0)) - log_z1_z0

    def save(self, path):
        weights = self.context_dist.get_weights()
        np.save(path, weights)

    def load(self, path):
        self.context_dist.set_weights(np.load(path))
        self.mean = self.context_dist.mean()
        self.sigma_inv = scpla.cho_solve(scpla.cho_factor(self.context_dist.covariance_matrix()),
                                         np.eye(self.mean.shape[0]))

    def _compute_objective(self, mean, chol_inv, cons, c_val, log_con_q0, alpha_cur):
        sigma_inv = torch.einsum("ij,kj->ik", chol_inv, chol_inv)
        log_con_q1 = self._unnormalized_log_pdf(cons, mean, sigma_inv)
        log_con_z1_z0 = self.lme(log_con_q1 - log_con_q0)
        weights = torch.exp(log_con_q1 - log_con_q0 - log_con_z1_z0)
        return torch.mean(weights * c_val) - alpha_cur * self.target_context_kl(mean, chol_inv)


class SelfPacedTeacher(AbstractTeacher, AbstractSelfPacedTeacher):

    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, update_frequency, update_offset, alpha_function,
                 initial_dist=None, target_dist=None, max_kl=0.1, std_lower_bound=None, kl_threshold=None,
                 samples_per_iter=300, discount_factor=0.99, max_context_buffer_size=1000, reset_contexts=True,
                 use_rejection_sampling=True):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)
        torch.manual_seed(self.seed)

        initial_mean, initial_variance = self.get_or_create_dist(initial_dist, mins, maxs,
                                                                 subspace=True)  # Random subspace of the task space if no intial dist
        # target_mean, target_variance = self.get_or_create_dist(target_dist, mins, maxs,
        #                                                        subspace=False)  # Full task space if no intial dist
        if target_dist is not None:
            target_mean, target_variance = target_dist["mean"], target_dist["variance"]
        else:
            target_mean, target_variance = None, None

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
        self.context_buffer = Buffer(2, max_context_buffer_size, reset_contexts)

        # The bounds that we show to the outside are limited to the interval [-1, 1], as this is typically better for
        # neural nets to deal with
        self.context_dim = initial_mean.shape[0]
        self.context_bounds = context_bounds

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
        if not isinstance(initial_variance, np.ndarray):
            initial_variance = initial_variance * np.eye(self.context_dim)

        # Create the target distribution
        if target_variance is not None and not isinstance(target_variance, np.ndarray):
            target_variance = target_variance * np.eye(self.context_dim)

        AbstractSelfPacedTeacher.__init__(self, initial_mean, initial_variance, context_bounds,
                                          alpha_function, max_kl,
                                          target_mean=target_mean,
                                          target_sigma=target_variance,
                                          samples_per_iter=samples_per_iter,
                                          use_rejection_sampling=use_rejection_sampling)
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

                log_con_q0 = torch.from_numpy(self.unnormalized_log_pdf(contexts))
                contexts_t = torch.from_numpy(contexts)
                c_val_t = torch.from_numpy(values)

                # Add the penalty term
                alpha_cur_t = self.alpha_function(self.iteration, avg_performance, self.target_kl)

                # Define the KL-Constraint
                def kl_con_fn(x):
                    mu = torch.from_numpy(x[0:self.context_dim])
                    chol_inv = torch.from_numpy(
                        GaussianTorchDistribution.to_tril_matrix(x[self.context_dim:], self.context_dim))
                    return self.kl(mu, chol_inv)

                def kl_con_grad_fn(x):
                    x_t = torch.from_numpy(x).requires_grad_(True)
                    mu = x_t[0:self.context_dim]
                    chol_inv = GaussianTorchDistribution.to_tril_matrix(x_t[self.context_dim:], self.context_dim)
                    cur_kl = self.kl(mu, chol_inv)
                    return torch.autograd.grad(cur_kl, [x_t])[0].detach().numpy()

                kl_constraint = NonlinearConstraint(kl_con_fn, -np.inf, self.max_kl, jac=kl_con_grad_fn, keep_feasible=True)
                constraints = [kl_constraint]

                x0 = np.concatenate([self.mean, GaussianTorchDistribution.flatten_matrix(self.sigma_inv, tril=False)])
                cones = np.ones_like(x0)
                lb = -np.inf * cones.copy()
                lb[0:self.context_dim] = self.bounds[0]
                ub = np.inf * cones.copy()
                ub[0:self.context_dim] = self.bounds[1]
                if self.kl_threshold is not None and self.target_kl > self.kl_threshold:
                    # Define the variance constraint as bounds
                    ub[self.context_dim: 2 * self.context_dim] = np.log(1 / self.std_lower_bound)

                bounds = Bounds(lb, ub, keep_feasible=True)
                x0 = np.clip(x0, lb, ub)

                # Define the objective plus Jacobian
                def objective(x):
                    x_t = torch.from_numpy(x).requires_grad_(True)
                    mu = x_t[0:self.context_dim]
                    chol_inv = GaussianTorchDistribution.to_tril_matrix(x_t[self.context_dim:], self.context_dim)

                    f = self._compute_objective(mu, chol_inv, contexts_t, c_val_t, log_con_q0, alpha_cur_t)
                    grad = torch.autograd.grad(f, [x_t])[0]

                    if torch.isnan(f) or torch.any(torch.isnan(grad)) or torch.isinf(f) or torch.any(torch.isinf(grad)):
                        raise RuntimeError("Nan/Inf detected")

                    return -f.detach().numpy() / avg_performance, -grad.detach().numpy() / avg_performance

                res = minimize(objective, x0.copy(), method="trust-constr", jac=True, bounds=bounds,
                               constraints=constraints, options={"gtol": 1e-4, "xtol": 1e-6})

                success = res.success
                if not success:
                    # If it was not a success, but the objective value was improved and the bounds are still valid, we still
                    # use the result
                    old_f = objective(x0)[0]
                    kl_ok = kl_con_fn(res.x) <= self.max_kl
                    std_ok = bounds is None or (np.all(bounds.lb <= res.x) and np.all(res.x <= bounds.ub))
                    if kl_ok and std_ok and res.fun < old_f:
                        success = True

                if success:
                    self.mean = res.x[0:self.context_dim]
                    chol_inv = GaussianTorchDistribution.to_tril_matrix(res.x[self.context_dim:], self.context_dim)
                    self.sigma_inv = np.dot(chol_inv, chol_inv.T)
                    self.context_dist = DummyGaussian(self.mean, self.sigma_inv)
                    if LOG_LEVEL == "all":
                        print("Successful optimisation !")
                else:
                    if LOG_LEVEL != "no":
                        print("Warning! Context optimisation unsuccessful - will keep old values. Message: %s" % res.message)

                self.sample_trunc()
                self.update_internal_values()
                self.bk["mean"].append(self.context_dist.mean())
                self.bk["covariance"].append(self.context_dist.covariance_matrix())
                self.bk["steps"].append(self.step_counter)
                self.bk["algo_iterations"].append(self.algorithm_iterations)
                self.bk["kl"].append(self.target_context_kl())
            else:
                if LOG_LEVEL != "no":
                    print("Warning! Skipping iteration at step {} because buffers are empty.".format(self.step_counter))

    def sample_task(self):
        if self.sample_count > len(self.idxs) - 1:
            self.sample_trunc()
            self.update_internal_values()

        step_idx = int(self.idxs[self.sample_count] / self.context_samples.shape[1])
        ensemble_idx = self.idxs[self.sample_count] % self.context_samples.shape[1]
        sample = self.context_samples[step_idx, ensemble_idx, :].copy()
        self.sample_count += 1

        return np.clip(sample, self.context_bounds[0], self.context_bounds[1], dtype=np.float32)

    def non_exploratory_task_sampling(self):
        return {"task": self.sample_task(),
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": None}
                }