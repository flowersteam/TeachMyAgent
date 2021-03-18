from TeachMyAgent.run_utils.abstract_args_handler import AbstractArgsHandler
from TeachMyAgent.teachers.teacher_controller import TeacherController
from TeachMyAgent.teachers.utils.alpha_functions import PercentageAlphaFunction

import numpy as np

class TeacherArgsHandler(AbstractArgsHandler):
    @staticmethod
    def set_parser_arguments(parser):
        parser.add_argument('--teacher', type=str, default="ALP-GMM")  # Random, ADR, ALP-GMM, Covar-GMM, RIAC, GoalGAN, Self-Paced, Truncated-Self-Paced, Setter-Solver
        parser.add_argument('--test_set', type=str, default=None)
        parser.add_argument('--allow_expert_knowledge', type=str, default="original") # original (white paper's version), no, low, high
        parser.add_argument('--keep_periodical_task_samples', type=int, default=None) # in steps /!\
        parser.add_argument('--shuffle_dimensions', action='store_true')
        parser.add_argument('--scale_reward', action='store_true')

        # ALP-GMM (Absolute Learning Progress - Gaussian Mixture Model) related arguments
        parser.add_argument('--random_percentage', '-rnd', type=float, default=5)
        parser.add_argument('--gmm_fitness_func', '-fit', type=str, default='aic')
        parser.add_argument('--nb_em_init', type=int, default=1)
        parser.add_argument('--min_k', type=int, default=2)
        parser.add_argument('--max_k', type=int, default=10)
        parser.add_argument('--fit_rate', type=int, default=150)
        parser.add_argument('--weighted_gmm', '-wgmm', action='store_true')
        parser.add_argument('--alp_max_size', type=float,
                            default=None)  # alp-computer window, expressed in Millions of env steps

        # Covar-GMM related arguments
        parser.add_argument('--absolute_lp', '-alp', action='store_true')

        # RIAC related arguments
        parser.add_argument('--max_region_size', type=int, default=150)
        parser.add_argument('--alp_window_size', type=int, default=None)
        parser.add_argument('--nb_split_attempts', type=int, default=75)
        parser.add_argument('--min_region_size', type=int, default=None)
        parser.add_argument('--min_dims_range_ratio', type=float, default=0.1)

        # ADR related arguments
        parser.add_argument('--boundary_sampling_p', '-bsp', type=float, default=0.7)
        parser.add_argument('--step_size', '-ss', type=float, default=0.1)
        parser.add_argument('--min_reward_thr', '-minrt', type=float, default=0)
        parser.add_argument('--max_reward_thr', '-maxrt', type=float, default=180)
        parser.add_argument('--queue_len', '-ql', type=int, default=10)

        # (Truncated) Self-Paced related arguments
        parser.add_argument('--sp_update_frequency', type=int, default=100000)
        parser.add_argument('--sp_update_offset', type=int, default=200000)
        parser.add_argument('--alpha_offset', type=int, default=0)
        parser.add_argument('--zeta', type=float, default=0.05)
        parser.add_argument('--max_kl', type=float, default=0.8)
        parser.add_argument('--std_lower_bound', type=float, default=None)
        parser.add_argument('--kl_threshold', type=float, default=None)
        parser.add_argument('--cg_parameters', type=float, default=None)
        parser.add_argument('--use_avg_performance', action='store_true')
        parser.add_argument('--use_rejection_sampling', action='store_true')

        # GoalGAN related arguments
        parser.add_argument('--state_noise_level', type=float, default=0.01)
        parser.add_argument('--success_distance_threshold', type=float, default=0.01)
        parser.add_argument('--gg_update_size', type=int, default=100)
        parser.add_argument('--n_rollouts', type=int, default=2)
        parser.add_argument('--goid_lb', type=float, default=0.25)
        parser.add_argument('--goid_ub', type=float, default=0.75)
        parser.add_argument('--p_old', type=float, default=0.2)
        parser.add_argument('--use_pretrained_samples', action='store_true')

        # Setter-Solver related arguments
        parser.add_argument('--ss_update_frequency', type=int, default=100)
        parser.add_argument('--setter_loss_noise_ub', type=float, default=0.05)
        parser.add_argument('--setter_hidden_size', type=int, default=128)

    @staticmethod
    def get_object_from_arguments(args, param_env_bounds, initial_dist=None, target_dist=None):
        params = {}
        # Reward bounds are necessary if you want a normalized reward for your teachers (not used as default).
        params["env_reward_lb"] = args.env_reward_lb
        params["env_reward_ub"] = args.env_reward_ub

        # Shuffle task space for the teacher ? (i.e. Rugged difficulty experiment)
        if args.shuffle_dimensions:
            shuffle_dims = True
        else:
            shuffle_dims = False

        # Normalize reward ?
        if args.scale_reward:
            scale_reward = True
        else:
            scale_reward = False

        if args.teacher == 'ALP-GMM':
            params['gmm_fitness_func'] = args.gmm_fitness_func
            params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
            if args.weighted_gmm is True:
                params['weighted_gmm'] = args.weighted_gmm
            params['nb_em_init'] = args.nb_em_init
            params['fit_rate'] = args.fit_rate
            if args.alp_max_size is not None:
                params['alp_max_size'] = int(args.alp_max_size * 1e6)
            if args.random_percentage is not None:
                params["random_task_ratio"] = args.random_percentage / 100.0
            if initial_dist is not None and args.allow_expert_knowledge == "high":
                params['initial_dist'] = initial_dist

        elif args.teacher == 'Covar-GMM':
            if args.absolute_lp is True:
                params['absolute_lp'] = args.absolute_lp
            params['fit_rate'] = args.fit_rate
            params['potential_ks'] = np.arange(args.min_k, args.max_k, 1)
            if args.random_percentage is not None:
                params["random_task_ratio"] = args.random_percentage / 100.0
            if initial_dist is not None and args.allow_expert_knowledge == "high":
                params['initial_dist'] = initial_dist

        elif args.teacher == "RIAC":
            params['max_region_size'] = args.max_region_size
            params['nb_split_attempts'] = args.nb_split_attempts
            if args.min_region_size is not None:
                params['min_region_size'] = args.min_region_size
            params['min_dims_range_ratio'] = args.min_dims_range_ratio
            if args.alp_window_size is not None:
                params['alp_window_size'] = args.alp_window_size

        elif args.teacher == "Oracle":
            if 'stump_height' in param_env_bounds and 'obstacle_spacing' in param_env_bounds:
                params['window_step_vector'] = [0.1, -0.2]  # order must match param_env_bounds construction
            elif 'poly_shape' in param_env_bounds:
                params['window_step_vector'] = [0.1] * 12
                print('hih')
            elif 'stump_seq' in param_env_bounds:
                params['window_step_vector'] = [0.1] * 10
            else:
                print('Oracle not defined for this parameter space')
                exit(1)

        elif args.teacher == "ADR":
            params['step_size'] = args.step_size
            params['boundary_sampling_p'] = args.boundary_sampling_p
            if args.allow_expert_knowledge in ["original", "low", "high"]:
                params['min_reward_thr'] = args.min_reward_thr
                params['max_reward_thr'] = args.max_reward_thr
            else:
                raise Exception("Unable to run ADR without any expert knowledge (needs at least reward thresholds).")
            params['queue_len'] = args.queue_len
            if initial_dist is not None and args.allow_expert_knowledge in ["original", "high"]:
                params['initial_dist'] = initial_dist

        elif args.teacher == "Self-Paced" or args.teacher == "Truncated-Self-Paced":
            params["update_frequency"] = args.sp_update_frequency
            params["update_offset"] = args.sp_update_offset
            params["alpha_function"] = PercentageAlphaFunction(args.alpha_offset, args.zeta)
            params["max_kl"] = args.max_kl
            params["std_lower_bound"] = args.std_lower_bound
            params["kl_threshold"] = args.kl_threshold
            if "gamma" in args:
                params["discount_factor"] = args.gamma
            else:
                params["discount_factor"] = 0.99

            if args.teacher == "Self-Paced":
                params["cg_parameters"] = args.cg_parameters
                if args.use_avg_performance:
                    params["use_avg_performance"] = True
                else:
                    params["use_avg_performance"] = False
            else:
                params["use_rejection_sampling"] = args.use_rejection_sampling

            if initial_dist is not None and args.allow_expert_knowledge in ["original", "maximal"]:
                params['initial_dist'] = initial_dist
            if target_dist is not None and args.allow_expert_knowledge in ["original", "maximal"]:
                params['target_dist'] = target_dist

        elif args.teacher == "GoalGAN":
            if args.allow_expert_knowledge == "no":
                raise Exception("Unable to run GoalGAN without any expert knowledge (needs at least a success definition).")
            params["state_noise_level"] = args.state_noise_level
            params["success_distance_threshold"] = args.success_distance_threshold
            params["update_size"] = args.gg_update_size
            params["n_rollouts"] = args.n_rollouts
            params["goid_lb"] = args.goid_lb
            params["goid_ub"] = args.goid_ub
            params["p_old"] = args.p_old

            if args.use_pretrained_samples or args.allow_expert_knowledge in ["original", "high"]:
                params["use_pretrained_samples"] = True
            else:
                params["use_pretrained_samples"] = False

            if initial_dist is not None and args.allow_expert_knowledge in ["original", "high"]:
                params['initial_dist'] = initial_dist

        elif args.teacher == "Setter-Solver":
            if args.allow_expert_knowledge == "no":
                raise Exception("Unable to run Setter-Solver without any expert knowledge (needs at least a success definition).")
            params["update_frequency"] = args.ss_update_frequency
            params["setter_loss_noise_ub"] = args.setter_loss_noise_ub
            params["setter_hidden_size"] = args.setter_hidden_size

        # Initialize teacher
        teacher = TeacherController(args.teacher, args.nb_test_episodes, param_env_bounds, test_set=args.test_set,
                                    seed=args.seed, keep_periodical_task_samples=args.keep_periodical_task_samples,
                                    shuffle_dimensions=shuffle_dims, scale_reward=scale_reward, **params)
        if args.test_set is not None:
            args.nb_test_episodes = teacher.nb_test_episodes
            print("Changing the number of test episodes to {0} with the test set {1}".format(args.nb_test_episodes,
                                                                                             args.test_set))
        return teacher
