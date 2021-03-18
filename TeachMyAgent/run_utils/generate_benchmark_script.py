import sys

if __name__ == '__main__':
    experiment_name = sys.argv[1]
    experiment_arguments = ' '.join(sys.argv[2:])
    nb_seeds = 16  # {}

    with open("benchmark_scripts/full_benchmark_" + experiment_name + ".txt", 'w') as f:
        f.write('#### PROFILING STUMPS\n')
        for ek in ["no", "low", "high"]:
            f.write('## {} Expert Knowledge\n'.format(ek))
            f.write('# Mostly unfeasible task space\n')
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_1 '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 9.0 --max_obstacle_spacing 6.0 --walker_type old_classic_bipedal --*allow_expert_knowledge {} '
                '--student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20 {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
            f.write('# Mostly feasible task space\n')
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_2 '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 3.0 --min_stump_h -3.0 --max_obstacle_spacing 6.0 --walker_type old_classic_bipedal '
                '--*allow_expert_knowledge {} --student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 '
                '--nb_test_episode 100 --nb_env_steps 20 {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
                )
            f.write('# Ability to handle a student that can forget\n')
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_3 '
                '--test_set parametric_stumps_test_set --reset_frequency 7000000 --keep_periodical_task_samples 250000 '
                '--env parametric-continuous-walker-v0 --max_stump_h 3.0 --max_obstacle_spacing 6.0 --walker_type old_classic_bipedal '
                '--*allow_expert_knowledge {} --student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 '
                '--nb_test_episode 100 --nb_env_steps 20 {}\n'
                .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
            f.write('# Handle discontinuous difficulty over task space\n')
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_4 --shuffle_dimensions '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 3.0 --max_obstacle_spacing 6.0 --walker_type old_classic_bipedal --*allow_expert_knowledge {} '
                '--student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20 {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
            f.write('# Robustness over a variety of students\n')
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_5 '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 3.0 --max_obstacle_spacing 6.0 --*walker_type spider --*allow_expert_knowledge {} '
                '--*student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20 {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_5 '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 3.0 --max_obstacle_spacing 6.0 --*walker_type small_bipedal --*allow_expert_knowledge {} '
                '--*student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20 {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_5 '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 3.0 --max_obstacle_spacing 6.0 --*walker_type spider --*allow_expert_knowledge {} '
                '--*student ppo --lr 0.0003 --backend tf1 --steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20  -hs {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
            f.write(
                '--slurm_conf curta_inria_long --nb_seeds {} --exp_name profiling_benchmark_stumps_{}_criteria_5 '
                '--test_set parametric_stumps_test_set --keep_periodical_task_samples 250000 --env parametric-continuous-walker-v0 '
                '--max_stump_h 3.0 --max_obstacle_spacing 6.0 --*walker_type small_bipedal --*allow_expert_knowledge {} '
                '--*student ppo --lr 0.0003 --backend tf1 --steps_per_ep 500000 --nb_test_episode 100 --nb_env_steps 20  -hs {}\n'
                    .format(nb_seeds, experiment_name, ek, experiment_arguments)
            )
        f.write('#### PARKOUR\n')
        f.write(
            '--slurm_conf curta_inria_long --nb_seeds 16 --exp_name benchmark_parkour_{} --test_set walking_test_set_v1 '
            '--keep_periodical_task_samples 250000 --env parametric-continuous-parkour-v0 --*walker_type old_classic_bipedal '
            '--allow_expert_knowledge minimal --student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 '
            '--nb_test_episode 100 --nb_env_steps 20 {}\n'
                .format(nb_seeds, experiment_name, ek, experiment_arguments)
        )
        f.write(
            '--slurm_conf curta_inria_long --nb_seeds 16+16 --exp_name benchmark_parkour_{} --test_set climbing_test_set_v1 '
            '--keep_periodical_task_samples 250000 --env parametric-continuous-parkour-v0 --*walker_type climbing_profile_chimpanzee '
            '--allow_expert_knowledge minimal --student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 '
            '--nb_test_episode 100 --nb_env_steps 20 {}\n'
                .format(nb_seeds, experiment_name, ek, experiment_arguments)
        )
        f.write(
            '--slurm_conf curta_inria_long --nb_seeds 16+32 --exp_name benchmark_parkour_{} --test_set swimming_test_set_v1 '
            '--keep_periodical_task_samples 250000 --env parametric-continuous-parkour-v0 --*walker_type fish '
            '--allow_expert_knowledge minimal --student sac_v0.1.1 --backend tf1 --steps_per_ep 500000 '
            '--nb_test_episode 100 --nb_env_steps 20 {}\n'
                .format(nb_seeds, experiment_name, ek, experiment_arguments)
        )
