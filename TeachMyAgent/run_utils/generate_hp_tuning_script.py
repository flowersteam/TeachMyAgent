import sys
import itertools

if __name__ == '__main__':
    '''
    Generate scripts to perform grid search on teachers' hyperparameters.
    '''

    tuning_dict = {
        "ALP-GMM": {
            "fit": [50, 150, 250, 350],
            "max_k": [5, 10, 15, 20],
            "random_percentage": [5, 10, 20, 30]
        },
        "Covar-GMM": {
            "fit": [50, 150, 250, 350],
            "max_k": [5, 10, 15, 20],
            "random_percentage": [5, 10, 20, 30]
        },
        "ADR": {
            "min_reward_thr": [0, 50],
            "max_reward_thr": [180, 230, 280],
            "boundary_sampling_p": [0.3, 0.5, 0.7],
            "queue_len": [10, 20],
            "step_size": [0.05, 0.1]
        },
        "RIAC": {
            "max_region_size": [50, 150, 250, 350],
            "nb_split_attempts": [25, 50, 75, 100],
            "min_dims_range_ratio": [0.0667, 0.1, 0.1667, 0.2],
        },
        "Self-Paced": {
            "sp_update_offset": [100000, 200000],
            "sp_update_frequency": [50000, 100000],
            "alpha_offset": [0, 5, 10],
            "zeta": [0.05, 0.25, 0.5],
            "max_kl": [0.1, 0.8],
            "use_avg_performance": [None]
        },
        "Truncated-Self-Paced": {
            "sp_update_offset": [100000, 200000],
            "sp_update_frequency": [50000, 100000],
            "alpha_offset": [0, 10],
            "zeta": [0.5, 1.0, 2.0, 4.0],
            "max_kl": [0.2, 0.6],
        },
        "Setter-Solver": {
            "ss_update_frequency": [50, 100, 200, 300],
            "setter_loss_noise_ub": [0.005, 0.01, 0.05, 0.1],
            "setter_hidden_size": [64, 128, 256, 512],
        },
        "GoalGAN": {
            "state_noise_level": [0.01, 0.05, 0.1],
            "gg_update_size": [100, 200, 300],
            "p_old": [0.1, 0.2, 0.3],
            "n_rollouts": [2, 5, 10],
            "use_pretrained_samples": [None]
        },

    }

    with open("hp_tuning_teachers.txt", 'w') as f:
        for teacher in tuning_dict:
            f.write('## {}\n'.format(teacher))
            current_teacher_parameters = list(tuning_dict[teacher].keys())
            current_teacher_hyperparams = tuning_dict[teacher].values()
            for point in itertools.product(*current_teacher_hyperparams):
                current_arguments = '--*teacher ' + teacher
                for i in range(len(current_teacher_parameters)):
                    current_arguments += ' --*' + current_teacher_parameters[i]
                    current_arguments += ' ' + str(point[i]) if point[i] is not None else ''

                f.write(
                    '--slurm_conf jeanzay_medium --nb_seeds 16 --exp_name teachers_hp_tuning --allow_expert_knowledge original '
                    '--test_set parametric_stumps_test_set --env parametric-continuous-walker-v0 --max_stump_h 3.0 '
                    '--max_obstacle_spacing 6.0 --walker_type old_classic_bipedal --student sac_v0.1.1 --backend tf1 '
                    '--steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 7 {} --keep_periodical_task_samples 250000\n'
                    .format(current_arguments)
                )