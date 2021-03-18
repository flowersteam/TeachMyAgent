from TeachMyAgent.run_utils.abstract_args_handler import AbstractArgsHandler
from TeachMyAgent.students.spinup.utils.run_utils import setup_logger_kwargs

class StudentArgsHandler(AbstractArgsHandler):
    @staticmethod
    def set_parser_arguments(parser):
        parser.add_argument('--student', type=str, default='sac_v0.1.1')  # Currently : sac_v0.1.1, sac_v0.2, ppo
        parser.add_argument('--backend', type=str, default='tf1')  # Currently : tf1 / pytorch

        parser.add_argument('--network', type=str, default='custom_mlp')
        parser.add_argument('--hidden_sizes', type=str, default="400/300")  # layers (with nb of neurons) separated by '/'

        # For all students
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--nb_env_steps', type=float,
                            default=20.0)  # Training time, expressed in Millions of env steps
        parser.add_argument('--gpu_id', type=int)  # default is no GPU
        parser.add_argument('--max_ep_len', type=int, default=2000)
        parser.add_argument('--nb_test_episodes', type=int, default=100)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--batch_size', type=int, default=1000)
        parser.add_argument('--half_save', '-hs', action='store_true')
        parser.add_argument('--no_save', action='store_true')
        parser.add_argument('--steps_per_ep', type=int,
                            default=500000)  # nb env steps/epoch (stay above max_ep_len and nb_env_steps)
        parser.add_argument('--reset_frequency', type=int, default=None)

        # SAC specific
        parser.add_argument('--train_freq', type=int, default=10)
        parser.add_argument('--buf_size', type=int, default=2000000)
        parser.add_argument('--alpha', type=float, default=0.005)

        # PPO specific
        parser.add_argument('--sample_size', type=int,
                            default=20000)  # size of trajectories sample between two updates
        parser.add_argument('--epochs_per_update', type=int,
                            default=5)  # how many times the model is trained on the full trajectories sample
        parser.add_argument('--vf_coef', type=float, default=0.5)
        parser.add_argument('--ent_coef', type=float, default=0.0)
        parser.add_argument('--clip_range', type=float, default=0.2)
        parser.add_argument('--max_grad_norm_coef', type=float, default=0.5)
        parser.add_argument('--lambda_factor', type=float, default=0.95)
        parser.add_argument('--value_network', type=str, default='shared') # shared OR copy

    @staticmethod
    def get_object_from_arguments(args, env_f, teacher):
        logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)
        ac_kwargs = dict()

        # Set up Student's DeepNN architecture if provided
        ac_kwargs['hidden_sizes'] = [int(layer) for layer in args.hidden_sizes.split("/")]
        if args.network == 'lstm':
            print("Using an lstm state size of {}".format(ac_kwargs['hidden_sizes'][0]))
            ac_kwargs['hidden_sizes'] = ac_kwargs['hidden_sizes'][0]
            if args.student == "ppo" and args.batch_size != args.sample_size:
                print("Setting batch size to the same value as sample size ({}) for ppo with lstm..."
                      .format(args.sample_size))
                args.batch_size = args.sample_size

        pretrained_model = None
        start_steps = 10000

        if "sac" in args.student:
            n_epochs = int((args.nb_env_steps * 1e6) // args.steps_per_ep)
            half_save = False
            if args.no_save is True:
                save_freq = n_epochs + 1  # no save
            else:
                save_freq = 1
                if args.half_save is True:
                    half_save = True

            if args.student == "sac_v0.1.1":
                if args.backend == "tf1":
                    from TeachMyAgent.students.spinup import sac_011_tf1 as sac
                elif args.backend == "pytorch":
                    raise("Old Spinup version of SAC only uses TF1. Please use the new version to use PyTorch.")

                return lambda: sac(env_f, ac_kwargs=ac_kwargs, gamma=args.gamma, seed=args.seed,
                                   epochs=n_epochs, start_steps=start_steps,
                                   logger_kwargs=logger_kwargs, alpha=args.alpha, max_ep_len=args.max_ep_len,
                                   steps_per_epoch=args.steps_per_ep, replay_size=args.buf_size,
                                   nb_test_episodes=args.nb_test_episodes, lr=args.lr, train_freq=args.train_freq,
                                   batch_size=args.batch_size, Teacher=teacher, half_save=half_save,
                                   pretrained_model=pretrained_model, save_freq=save_freq, reset_frequency=args.reset_frequency)
            elif args.student == "sac_v0.2":
                if args.backend == "tf1":
                    from TeachMyAgent.students.spinup import sac_02_tf1 as sac
                elif args.backend == "pytorch":
                    from TeachMyAgent.students.spinup import sac_02_pytorch as sac
                    if args.reset_frequency is not None:
                        raise NotImplementedError(
                            "Resetting the student is currently not implemented with the Pytorch backend of SAC.")

                return lambda: sac(env_f, ac_kwargs=ac_kwargs, gamma=args.gamma, seed=args.seed,
                                   epochs=int((args.nb_env_steps * 1e6) // args.steps_per_ep), start_steps=start_steps,
                                   logger_kwargs=logger_kwargs, alpha=args.alpha, max_ep_len=args.max_ep_len,
                                   steps_per_epoch=args.steps_per_ep, replay_size=args.buf_size,
                                   num_test_episodes=args.nb_test_episodes, lr=args.lr, update_every=args.train_freq,
                                   batch_size=args.batch_size, Teacher=teacher, half_save=half_save,
                                   pretrained_model=pretrained_model, save_freq=save_freq, reset_frequency=args.reset_frequency)
        elif args.student == "ppo":
            if args.backend == "pytorch":
                raise ("Currently only implemented OpenAI Baselines' version, which is TF1.")
            from TeachMyAgent.students.openai_baselines.ppo2.ppo2 import learn
            from TeachMyAgent.students.ppo_utils import create_custom_vec_normalized_envs

            assert args.sample_size % args.batch_size == 0
            assert args.steps_per_ep % args.sample_size == 0
            if args.reset_frequency is not None:
                assert args.reset_frequency % args.sample_size == 0

            log_interval = int(args.steps_per_ep // args.sample_size)  # Log every "epoch"
            if args.no_save is True:
                save_interval = None
            else:
                if args.half_save is True:
                    save_interval = ((args.nb_env_steps * 1e6) // args.sample_size) / 2
                else:
                    save_interval = log_interval
                    print("Warning: not using half_save with PPO will produce a lot of disk usage.")

            env, eval_env = create_custom_vec_normalized_envs(env_f)

            return lambda: learn(network=args.network, env=env, eval_env=eval_env, total_timesteps=args.nb_env_steps * 1e6,
                                 nsteps=args.sample_size, nminibatches=int(args.sample_size//args.batch_size), lr=args.lr,
                                 noptepochs=args.epochs_per_update, log_interval=log_interval, gamma=args.gamma, seed=args.seed,
                                 ent_coef=args.ent_coef, vf_coef=args.vf_coef, lam=args.lambda_factor, cliprange=args.clip_range,
                                 max_grad_norm=args.max_grad_norm_coef, save_interval=save_interval, Teacher=teacher,
                                 max_ep_len=args.max_ep_len, nb_test_episodes=args.nb_test_episodes,
                                 hidden_sizes=ac_kwargs['hidden_sizes'], logger_dir=logger_kwargs['output_dir'],
                                 reset_frequency=args.reset_frequency, value_network=args.value_network)



