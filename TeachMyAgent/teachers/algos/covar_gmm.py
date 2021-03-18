# Taken from https://arxiv.org/abs/1910.07224
# Modified by Clément Romac, copy of the license at TeachMyAgent/teachers/LICENSES/ALP-GMM

from sklearn.mixture import GaussianMixture as GMM
import numpy as np
from gym.spaces import Box
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher

def proportional_choice(v, random_state, eps=0.):
    if np.sum(v) == 0 or random_state.rand() < eps:
        return random_state.randint(np.size(v))
    else:
        probas = np.array(v) / np.sum(v)
        return np.where(random_state.multinomial(1, probas) == 1)[0][0]

# Implementation of IGMM (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3893575/) + minor improvements
class CovarGMM(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, absolute_lp=False, fit_rate=250,
                 potential_ks=np.arange(2, 11, 1), random_task_ratio=0.2, nb_bootstrap=None, initial_dist=None):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        # Range of number of Gaussians to try when fitting the GMM
        self.potential_ks = potential_ks
        # Ratio of randomly sampled tasks VS tasks sampling using GMM
        self.random_task_ratio = random_task_ratio
        self.random_task_generator = Box(self.mins, self.maxs, dtype=np.float32)
        self.random_task_generator.seed(self.seed)

        # Number of episodes between two fit of the GMM
        self.fit_rate = fit_rate
        self.nb_bootstrap = nb_bootstrap if nb_bootstrap is not None else fit_rate  # Number of bootstrapping episodes, must be >= to fit_rate
        self.initial_dist = initial_dist  # Initial Gaussian distribution. If None, bootstrap with random tasks

        # Original version does not use Absolute LP, only LP.
        self.absolute_lp = absolute_lp

        self.tasks = []
        self.tasks_times_rewards = []
        self.all_times = np.arange(0, 1, 1/self.fit_rate)
        self.gmm = None

        # boring book-keeping
        self.bk = {'weights': [], 'covariances': [], 'means': [], 'tasks_lps': [], 'episodes': [], 'tasks_origin': []}

    def episodic_update(self, task, reward, is_success):
        # Compute time of task, relative to position in current batch of tasks
        current_time = self.all_times[len(self.tasks) % self.fit_rate]

        self.tasks.append(task)

        # Concatenate task with its corresponding time and reward
        self.tasks_times_rewards.append(np.array(task.tolist() + [current_time] + [reward]))

        if len(self.tasks) >= self.nb_bootstrap:  # If initial bootstrapping is done
            if (len(self.tasks) % self.fit_rate) == 0:  # Time to fit
                # 1 - Retrieve last <fit_rate> (task, time, reward) triplets
                cur_tasks_times_rewards = np.array(self.tasks_times_rewards[-self.fit_rate:])

                # 2 - Fit batch of GMMs with varying number of Gaussians
                potential_gmms = [GMM(n_components=k, covariance_type='full', random_state=self.seed) for k in self.potential_ks]
                potential_gmms = [g.fit(cur_tasks_times_rewards) for g in potential_gmms]

                # 3 - Compute fitness and keep best GMM
                aics = [m.aic(cur_tasks_times_rewards) for m in potential_gmms]
                self.gmm = potential_gmms[np.argmin(aics)]

                # book-keeping
                self.bk['weights'].append(self.gmm.weights_.copy())
                self.bk['covariances'].append(self.gmm.covariances_.copy())
                self.bk['means'].append(self.gmm.means_.copy())
                self.bk['tasks_lps'] = self.tasks_times_rewards
                self.bk['episodes'].append(len(self.tasks))

    def sample_task(self):
        task_origin = None
        if len(self.tasks) < self.nb_bootstrap or self.random_state.random() < self.random_task_ratio or self.gmm is None:
            if self.initial_dist and len(self.tasks) < self.nb_bootstrap:  # bootstrap in initial dist
                # Expert bootstrap Gaussian task sampling
                new_task = self.random_state.multivariate_normal(self.initial_dist['mean'],
                                                                 self.initial_dist['variance'])
                new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
                task_origin = -2  # -2 = task originates from initial bootstrap gaussian sampling
            else:
                # Random task sampling
                new_task = self.random_task_generator.sample()
                task_origin = -1  # -1 = task originates from random sampling
        else:
            # Task sampling based on positive time-reward covariance
            # 1 - Retrieve positive time-reward covariance for each Gaussian
            self.times_rewards_covars = []
            for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
                if self.absolute_lp:
                    self.times_rewards_covars.append(np.abs(covar[-2, -1]))
                else:
                    self.times_rewards_covars.append(max(0, covar[-2, -1]))

            # 2 - Sample Gaussian according to its Learning Progress, defined as positive time-reward covariance
            idx = proportional_choice(self.times_rewards_covars, self.random_state, eps=0.0)
            task_origin = idx

            # 3 - Sample task in Gaussian, without forgetting to remove time and reward dimension
            new_task = self.random_state.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-2]
            new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)

        # boring book-keeping
        self.bk['tasks_origin'].append(task_origin)
        return new_task

    def is_non_exploratory_task_sampling_available(self):
        return self.gmm is not None

    def non_exploratory_task_sampling(self):
        # 1 - Retrieve positive time-reward covariance for each Gaussian
        self.times_rewards_covars = []
        for pos, covar, w in zip(self.gmm.means_, self.gmm.covariances_, self.gmm.weights_):
            if self.absolute_lp:
                self.times_rewards_covars.append(np.abs(covar[-2, -1]))
            else:
                self.times_rewards_covars.append(max(0, covar[-2, -1]))

        # 2 - Sample Gaussian according to its Learning Progress, defined as positive time-reward covariance
        idx = proportional_choice(self.times_rewards_covars, self.random_state, eps=0.0)

        # 3 - Sample task in Gaussian, without forgetting to remove time and reward dimension
        new_task = self.random_state.multivariate_normal(self.gmm.means_[idx], self.gmm.covariances_[idx])[:-2]
        new_task = np.clip(new_task, self.mins, self.maxs).astype(np.float32)
        return {"task": new_task,
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": idx}
                }