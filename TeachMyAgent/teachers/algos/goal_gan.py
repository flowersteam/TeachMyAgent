# Taken from https://github.com/psclklnk/spdl and wrapped in our architecture
# Modified by Clément Romac, copy of the license at TeachMyAgent/teachers/LICENSES/SPDL

import os
import numpy as np

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from queue import Queue
from TeachMyAgent.teachers.algos.AbstractTeacher import AbstractTeacher
from TeachMyAgent.teachers.utils.gan.generator import StateGAN, StateCollection


class GoalGAN(AbstractTeacher):
    def __init__(self, mins, maxs, seed, env_reward_lb, env_reward_ub, state_noise_level, success_distance_threshold,
                 update_size,  n_rollouts=2, goid_lb=0.25, goid_ub=0.75, p_old=0.2, use_pretrained_samples=False,
                 initial_dist=None):
        AbstractTeacher.__init__(self, mins, maxs, env_reward_lb, env_reward_ub, seed)

        np.random.seed(self.seed) # To seed the GAN (sufficient ?)
        tf.set_random_seed(
            seed
        )

        tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=1,
            intra_op_parallelism_threads=1)
        # Prevent tensorflow from taking all the gpu memory
        tf_config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(config=tf_config)
        self.gan = StateGAN(
            state_size=len(mins),
            evaluater_size=1,
            state_range=0.5 * (self.maxs - self.mins) + 1e-6, # avoid normalization issues for dimensions where min==max
            state_center=mins + 0.5 * (self.maxs - self.mins),
            state_noise_level=(state_noise_level * (self.maxs - self.mins))[None, :],
            generator_layers=[256, 256],
            discriminator_layers=[128, 128],
            noise_size=self.mins.shape[0],
            tf_session=self.tf_session,
            configs={"supress_all_logging": True}
        )
        self.tf_session.run(tf.initialize_local_variables())
        self.replay_noise = state_noise_level * (self.maxs - self.mins)
        self.success_buffer = StateCollection(1, success_distance_threshold * np.linalg.norm(self.maxs - self.mins))

        self.update_size = update_size
        self.contexts = []
        self.labels = []

        self.p_old = p_old
        self.n_rollouts = n_rollouts
        self.goid_lb = goid_lb
        self.goid_ub = goid_ub

        self.pending_contexts = {}
        self.context_queue = Queue()
        self.episode_counter = 0

        if use_pretrained_samples:
            print("Pretraining GAN...")
            initial_mean, initial_variance = self.get_or_create_dist(initial_dist, mins, maxs, subspace=True)
            pretrain_samples = self.random_state.multivariate_normal(initial_mean, initial_variance, size=1000)
            pretrain_samples = np.clip(pretrain_samples, mins, maxs, dtype=np.float32)
            self.gan.pretrain(pretrain_samples)

        self.bk = {'dis_log_loss': [],
                   'gen_log_loss': [],
                   'episodes': []}

    def sample_task(self):
        if self.context_queue.empty():
            if self.random_state.random() > self.p_old or self.success_buffer.size == 0:
                context = np.float32(self.gan.sample_states_with_noise(1)[0][0, :])
                context = np.clip(context, self.mins, self.maxs, dtype=np.float32) # added because the generators uses mins - 1e-6 and maxs - 1-6
                context_key = context.tobytes()

                # Either extend or create the list - note that this is a nested list to support the case when an
                # identical context is sampled twice (Happens e.g. due to clipping)
                if context_key in self.pending_contexts:
                    self.pending_contexts[context_key].append([])
                else:
                    self.pending_contexts[context_key] = [[]]

                # Store the contexts in the buffer for being sampled
                for i in range(0, self.n_rollouts - 1):
                    self.context_queue.put(context.copy())
            else:
                context = self.success_buffer.sample(size=1, replay_noise=self.replay_noise)[0, :]
                context = np.clip(context, self.mins, self.maxs, dtype=np.float32)
        else:
            context = self.context_queue.get()

        return context

    def non_exploratory_task_sampling(self):
        return {"task": np.float32(self.gan.sample_states(1)[0][0, :]),
                "infos": {
                    "bk_index": len(self.bk[list(self.bk.keys())[0]]) - 1,
                    "task_infos": None}
                }

    def episodic_update(self, task, reward, is_success):
        self.episode_counter += 1
        context_key = task.tobytes()
        if context_key in self.pending_contexts:
            # Always append to the first list
            self.pending_contexts[context_key][0].append(is_success)

            if len(self.pending_contexts[context_key][0]) >= self.n_rollouts:
                mean_success = np.mean(self.pending_contexts[context_key][0])
                self.labels.append(self.goid_lb <= mean_success <= self.goid_ub)
                self.contexts.append(task.copy())

                if mean_success > self.goid_ub:
                    self.success_buffer.append(task.copy()[None, :])

                # Delete the first entry of the nested list
                del self.pending_contexts[context_key][0]

                # If the list is now empty, we can remove the whole entry in the map
                if len(self.pending_contexts[context_key]) == 0:
                    del self.pending_contexts[context_key]

        if len(self.contexts) >= self.update_size:
            labels = np.array(self.labels, dtype=np.float)[:, None]
            if np.any(labels):
                print("Training GoalGAN with " + str(len(self.contexts)) + " contexts")
                dis_log_loss, gen_log_loss = self.gan.train(np.array(self.contexts), labels, 250)
                self.bk["dis_log_loss"].append(dis_log_loss)
                self.bk["gen_log_loss"].append(gen_log_loss)
                self.bk["episodes"].append(self.episode_counter)
            else:
                print("No positive samples in " + str(len(self.contexts)) + " contexts - skipping GoalGAN training")

            self.contexts = []
            self.labels = []