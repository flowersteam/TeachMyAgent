from collections import OrderedDict
import numpy as np
import gym
import TeachMyAgent.environments
from TeachMyAgent.run_utils.abstract_args_handler import AbstractArgsHandler
from TeachMyAgent.environments.envs.bodies.BodyTypesEnum import BodyTypesEnum
from TeachMyAgent.environments.envs.bodies.BodiesEnum import BodiesEnum


class EnvironmentArgsHandler(AbstractArgsHandler):
    @staticmethod
    def get_body_wargs(args):
        body_args = {}
        if args.motors_torque is not None:
            body_args["motors_torque"] = args.motors_torque

        if args.embodiment == "millepede" and args.nb_of_bodies is not None:
            body_args["nb_of_bodies"] = args.nb_of_bodies
        elif args.embodiment == "spider" and args.nb_pairs_of_legs is not None:
            body_args["nb_pairs_of_legs"] = args.nb_pairs_of_legs
        elif args.embodiment == "wheel" and args.body_scale is not None:
            body_args["body_scale"] = args.body_scale

        body_type = BodiesEnum.get_body_type(args.embodiment)
        if body_type == BodyTypesEnum.SWIMMER and args.outside_water is not None:
            body_args["nb_steps_outside_water"] = args.outside_water
        elif (body_type == BodyTypesEnum.WALKER or body_type == BodyTypesEnum.CLIMBER) and args.under_water is not None:
            body_args["nb_steps_under_water"] = args.under_water

        return body_args

    @staticmethod
    def set_parser_arguments(parser):
        parser.add_argument('--env', type=str, default="parametric-continuous-stump-tracks-v0")

        # Choose student's embodiment
        parser.add_argument('--embodiment', type=str, default="old_classic_bipedal")  # choose morphology
        parser.add_argument('--motors_torque', type=float, default=None)
        parser.add_argument('--nb_of_bodies', type=float, default=None) # Millipede
        parser.add_argument('--nb_pairs_of_legs', type=float, default=None) # Spider
        parser.add_argument('--body_scale', type=float, default=None)  # Wheel
        parser.add_argument('--outside_water', type=float, default=None)  # Fish
        parser.add_argument('--under_water', type=float, default=None)

        ##### Stump Tracks Env #####
        parser.add_argument('--min_stump_h', type=float, default=0.0)
        parser.add_argument('--max_stump_h', type=float, default=3.0)
        parser.add_argument('--max_stump_w', type=float, default=None)
        parser.add_argument('--max_stump_r', type=float, default=None)
        parser.add_argument('--roughness', type=float, default=None)
        parser.add_argument('--max_obstacle_spacing', type=float, default=6.0)
        parser.add_argument('--max_gap_w', type=float, default=None)
        parser.add_argument('--step_h', type=float, default=None)
        parser.add_argument('--step_nb', type=float, default=None)
        parser.add_argument('--hexa_shape', '-hexa', action='store_true')
        parser.add_argument('--stump_seq', '-seq', action='store_true')

        ##### Parkour Env #####
        # Selection of parameter space
        parser.add_argument('--CPPN_weights_path', type=str, default=None)
        parser.add_argument('--input_CPPN_dim', type=int, default=3)
        parser.add_argument('--ceiling_offset', type=float, default=200)
        parser.add_argument('--ceiling_clip_offset', type=float, default=0)
        parser.add_argument('--terrain_cppn_scale', type=float, default=10)
        parser.add_argument('--water_clip_push', type=float, default=20)
        parser.add_argument('--CPPN_input_space', type=str, default='medium')
        parser.add_argument('--min_input_CPPN', type=float, default=-1.0)
        parser.add_argument('--max_input_CPPN', type=float, default=1.0)
        parser.add_argument('--min_water_level', type=float, default=0.0)
        parser.add_argument('--max_water_level', type=float, default=1.0)
        parser.add_argument('--min_creepers_w', type=float, default=0.25)
        parser.add_argument('--max_creepers_w', type=float, default=0.25)
        parser.add_argument('--max_creepers_h', type=float, default=4)
        parser.add_argument('--max_creepers_spacing', type=float, default=5)
        parser.add_argument('--lidars_type',  type=str, default=None)
        parser.add_argument('--movable_creepers', action='store_true')


    @classmethod
    def get_object_from_arguments(cls, args):
        param_env_bounds = OrderedDict()
        # For teachers using an initial distribution of easy tasks
        initial_dist = None
        # For teachers using a target distribution of tasks
        target_dist = None

        if args.env == "parametric-continuous-stump-tracks-v0":
            args.env_reward_lb = -150
            args.env_reward_ub = 350
            if args.hexa_shape:
                print('hexa env')
                # adjust default parameters
                args.steps_per_ep = 2000000
                args.nb_test_episodes = 400

            initial_dist = {
                "mean": [],
                "variance": []
            }

            if args.max_stump_h is not None:
                param_env_bounds['stump_height'] = [args.min_stump_h, args.max_stump_h]
                initial_dist["mean"].append(0)
                initial_dist["variance"].append((abs(args.max_stump_h - args.min_stump_h) * 0.1)**2) # std = 10% of dimension
            if args.max_stump_w is not None:
                param_env_bounds['stump_width'] = [0, args.max_stump_w]
                initial_dist["mean"].append(0)
                initial_dist["variance"].append((args.max_stump_w * 0.1)**2) # std = 10% of dimension
            if args.max_stump_r is not None:
                param_env_bounds['stump_rot'] = [0, args.max_stump_r]
                initial_dist["mean"].append(0)
                initial_dist["variance"].append((args.max_stump_r * 0.1)**2) # std = 10% of dimension
            if args.max_obstacle_spacing is not None:
                param_env_bounds['obstacle_spacing'] = [0, args.max_obstacle_spacing]
                initial_dist["mean"].append(args.max_obstacle_spacing)
                initial_dist["variance"].append((args.max_obstacle_spacing * 0.1)**2) # std = 10% of dimension
            if args.hexa_shape:
                param_env_bounds['poly_shape'] = [0, 4.0, 12]
                initial_dist["mean"].extend([0 for _ in range(12)])
                initial_dist["variance"].extend([0.01 for _ in range(12)])
            if args.stump_seq:
                param_env_bounds['stump_seq'] = [0, 6.0, 10]
                initial_dist["mean"].extend([0 for _ in range(10)])
                initial_dist["variance"].extend([0.01 for _ in range(10)])

            initial_dist["mean"] = np.array(initial_dist["mean"])
            initial_dist["variance"] = np.diag(initial_dist["variance"])

            env_f = lambda: gym.make(args.env, walker_type=args.embodiment, **cls.get_body_wargs(args))

        elif args.env == "parametric-continuous-parkour-v0":
            args.env_reward_lb = -150
            args.env_reward_ub = 360
            if args.CPPN_input_space == "easy":
                param_env_bounds["input_vector"] = [[-0.25, -0.05],
                                                    [0.8, 1.0],
                                                    [0.0, 0.2]]
            elif args.CPPN_input_space == "medium":
                param_env_bounds["input_vector"] = [[-0.35, 0.05],
                                                    [0.6, 1.0],
                                                    [-0.1, 0.3]]
            else:
                param_env_bounds["input_vector"] = [args.min_input_CPPN,
                                                    args.max_input_CPPN,
                                                    args.input_CPPN_dim]

            param_env_bounds["water_level"] = [args.min_water_level, args.max_water_level]
            param_env_bounds['creepers_width'] = [args.min_creepers_w, args.max_creepers_w]
            param_env_bounds['creepers_height'] = [0, args.max_creepers_h]
            param_env_bounds['creepers_spacing'] = [0, args.max_creepers_spacing]
            param_env_bounds['terrain_cppn_scale'] = [args.terrain_cppn_scale, args.terrain_cppn_scale]
            if args.movable_creepers:
                movable_creepers = True
            else:
                movable_creepers = False

            body_type = BodiesEnum.get_body_type(args.embodiment)
            if body_type == BodyTypesEnum.WALKER:
                if args.lidars_type is None:
                    args.lidars_type = "down"
            elif body_type == BodyTypesEnum.CLIMBER:
                if args.lidars_type is None:
                    args.lidars_type = "up"
            if body_type == BodyTypesEnum.SWIMMER:
                if args.lidars_type is None:
                    args.lidars_type = "full"

            env_f = lambda: gym.make(args.env,
                                     agent_body_type=args.embodiment,
                                     CPPN_weights_path=args.CPPN_weights_path,
                                     input_CPPN_dim=args.input_CPPN_dim,
                                     ceiling_offset=args.ceiling_offset,
                                     ceiling_clip_offset=args.ceiling_clip_offset,
                                     terrain_cppn_scale=args.terrain_cppn_scale,
                                     lidars_type=args.lidars_type,
                                     water_clip=args.water_clip_push,
                                     movable_creepers=movable_creepers,
                                     **cls.get_body_wargs(args))

        else:
            print("Using an unknown env with no parameters...")
            args.env_reward_lb = -100
            args.env_reward_ub = 100
            env_f = lambda: gym.make(args.env)
            # raise Exception("No such an environment !")

        return env_f, param_env_bounds, initial_dist, target_dist
