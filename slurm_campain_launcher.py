import sys
from pathlib import Path
from datetime import date
import subprocess
import shutil
import os
import stat

##### USE THIS FILE TO LAUNCH EXPERIMENT CAMPAIGNS ON A CLUSTER #####
##### A CAMPAIGN IS COMPOSE OF MULTIPLE EXPERIMENTS AND MULTIPLE SEEDS PER EXPERIMENT #####

def process_arg_string(expe_args):  # function to extract flagged (with a *) arguments as details for experience name
    details_string = ''
    processed_arg_string = expe_args.replace('*', '')  # keep a version of args cleaned from exp name related flags
    arg_chunks = [arg_chunk for arg_chunk in expe_args.split(' --')]
    args_list = []
    for arg in arg_chunks:
        if ' -' in arg and arg.split(' -')[1].isalpha():
            args_list.extend(arg.split(' -'))
        else:
            args_list.append(arg)
    for arg in args_list:
        if arg == '':
            continue
        if arg[0] == '*':
            if arg[-1] == ' ':
                arg = arg[:-1]
            details_string += '_' + arg[1:].replace(' ', '_').replace('/', '-')
    return details_string, processed_arg_string


slurm_confs = {'curta_inria_extra_long': "#SBATCH -p XXX\n"
                                         "#SBATCH -t 119:00:00\n",
               'curta_inria_long': "#SBATCH -p XXX\n"
                                   "#SBATCH -t 72:00:00\n",
               'curta_inria_medium': "#SBATCH -p XXX\n"
                                     "#SBATCH -t 48:00:00\n",
               'curta_inria_short': "#SBATCH -p XXX\n"
                                    "#SBATCH -t 24:00:00\n",
               'jeanzay_short': '#SBATCH -A XXX\n'
                                '#SBATCH --gres=gpu:1\n'
                                "#SBATCH -t 19:59:00\n"
                                "#SBATCH --qos=qos_gpu-t3\n",
               'jeanzay_medium': '#SBATCH -A XXX\n'
                                 '#SBATCH --gres=gpu:1\n'
                                 "#SBATCH -t 48:00:00\n"
                                 "#SBATCH --qos=qos_gpu-t4\n",
               'jeanzay_long': '#SBATCH -A XXX\n'
                               '#SBATCH --gres=gpu:1\n'
                               "#SBATCH -t 72:00:00\n"
                               "#SBATCH --qos=qos_gpu-t4\n",
               'jeanzay_new_short': '#SBATCH -A XXX\n'
                                '#SBATCH --gres=gpu:1\n'
                                "#SBATCH -t 19:59:00\n"
                                "#SBATCH --qos=qos_gpu-t3\n",
               'jeanzay_new_medium': '#SBATCH -A XXX\n' 
                                 '#SBATCH --gres=gpu:1\n'
                                 "#SBATCH -t 48:00:00\n"
                                 "#SBATCH --qos=qos_gpu-t4\n",
               'jeanzay_new_long': '#SBATCH -A XXX\n'
                               '#SBATCH --gres=gpu:1\n'
                               "#SBATCH -t 72:00:00\n"
                               "#SBATCH --qos=qos_gpu-t4\n",
               'plafrim_cpu_medium': "#SBATCH -t 48:00:00\n",
               'plafrim_cpu_long': "#SBATCH -t 72:00:00\n",
               'plafrim_gpu_medium': '#SBATCH -p XXX\n'
                                     "#SBATCH -t 48:00:00\n"
                                     '#SBATCH --gres=gpu:1\n'
               }

cur_path = str(Path.cwd())
date = date.today().strftime("%d-%m")
# create campain log dir if not already done
Path(cur_path + "/campain_logs/jobouts/").mkdir(parents=True, exist_ok=True)
Path(cur_path + "/campain_logs/scripts/").mkdir(parents=True, exist_ok=True)
# Load txt file containing experiments to run (give it as argument to this script)
filename = 'to_run.txt'
if len(sys.argv) >= 2:
    filename = sys.argv[1]
launch = True
# Save a copy of txt file
shutil.copyfile(cur_path + "/" + filename, cur_path + '/campain_logs/scripts/' + date + '_' + filename)

global_seed_offset = 0
incremental = False
if len(sys.argv) >= 3:
    if sys.argv[2] == 'nolaunch':
        launch = False
    if sys.argv[2] == 'seed_offset':
        global_seed_offset = int(sys.argv[3])
    if sys.argv[2] == 'incremental_seed_offset':
        global_seed_offset = int(sys.argv[3])
        incremental = True
if launch:
    print('Creating and Launching slurm scripts given arguments from {}'.format(filename))
    # time.sleep(1.0)
expe_list = []
with open(filename, 'r') as f:
    expe_list = [line.rstrip() for line in f]

for expe_args in expe_list:
    seed_offset_to_use = global_seed_offset
    if expe_args[0] == '#':
        continue
    print('creating slurm script with: {}'.format(expe_args))
    exp_config = expe_args.split('--')[1:4]
    slurm_conf_name, nb_seeds, exp_name = [arg.split(' ')[1] for arg in exp_config]
    if 'curta' in slurm_conf_name:
        gpu = ''
        PYTHON_INTERP = '/gpfs/home/XXX/bin/python'
        n_cpus = 1
    elif 'plafrim' in slurm_conf_name:
        gpu = ''
        PYTHON_INTERP = '/home/XXX/bin/python'
        n_cpus = 1
    elif 'jeanzay_new' in slurm_conf_name:
        PYTHON_INTERP = '/gpfsdswork/projects/rech/XXX/bin/python'
        gpu = ''  # '--gpu_id 0'
        n_cpus = 2
    elif 'jeanzay' in slurm_conf_name:
        PYTHON_INTERP = '/gpfswork/rech/XXX/bin/python'
        gpu = ''
        n_cpus = 2
    else:
        raise Exception("Unrecognized conf name.")

    # parse possible seed offset
    if '+' in nb_seeds:
        nb_seeds, offset = nb_seeds.split('+')
        seed_offset_to_use += int(offset)
    assert ((int(nb_seeds) % 8) == 0), 'number of seeds should be divisible by 8'
    run_args = expe_args.split(exp_name, 1)[
        1]  # WARNING: assumes that exp_name comes after slurm_conf and nb_seeds in txt
    # prepare experiment name formatting (use --* or -* instead of -- or - to use argument in experiment name
    # print(expe_args.split(exp_name))
    exp_details, run_args = process_arg_string(run_args)
    exp_name = date + '_' + exp_name + exp_details

    slurm_script_fullname = cur_path + "/campain_logs/scripts/{}".format(exp_name) + ".sh"
    # create corresponding slurm script
    with open(slurm_script_fullname, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('#SBATCH --ntasks=1\n')
        f.write('#SBATCH --cpus-per-task=8\n')
        if "jeanzay" in slurm_conf_name:
            f.write('#SBATCH --hint=nomultithread\n')
        f.write(slurm_confs[slurm_conf_name])
        f.write('#SBATCH --open-mode=append\n')  # append logs in logs files instead of truncating
        f.write('#SBATCH -o campain_logs/jobouts/{}.sh.out\n'
                '#SBATCH -e campain_logs/jobouts/{}.sh.err\n'.format(exp_name, exp_name))
        f.write("export EXP_INTERP='{}' ;\n".format(PYTHON_INTERP))
        f.write('mkdir TeachMyAgent/data/{}\n'.format(exp_name))
        f.write('# Copy itself in experimental dir\n')
        f.write('cp $SLURM_JOB_NAME TeachMyAgent/data/{}/{}.sh\n'.format(exp_name, exp_name))
        f.write('# Add info about git status to exp dir\n')
        f.write('current_commit="$(git log -n 1)"\n')
        f.write('echo "${{current_commit}}" > TeachMyAgent/data/{}/git_status.txt\n'.format(exp_name))
        f.write('# Launch !\n')
        f.write(
            'cpu_list=$(taskset -pc $$ | sed -E "s/(.*): (.*)/\\2/g" | tr "," "\\n" | sed -E "s/^[0-9]*$/&-&/g" | sed -E "s/-/ /g" | xargs -l seq | tr "\\n" " ")\n')
        f.write('COUNT=${1:-0}\n')
        f.write('i=0\n')
        f.write('cpus=""\n')
        f.write('for cpu in $cpu_list; do\n')
        f.write('cpus="$cpus$cpu"\n')
        f.write('i=$(($i+1))\n')
        f.write('if [ "$i" = "{}" ]; then\n'.format(n_cpus))
        f.write('taskset -c $cpus $EXP_INTERP run.py --exp_name {} {} --seed $COUNT'.format(exp_name,
                                                                                            gpu) + run_args + ' &\n')
        f.write('echo "Using cpus $cpus for seed $COUNT"\n')
        f.write('COUNT=$(( $COUNT + 1 ))\n')
        f.write('cpus=""\n')
        f.write('i=0\n')
        f.write('else\n')
        f.write('cpus="$cpus,"\n')
        f.write('fi\n')
        f.write('done\n')
        f.write('wait\n')
        f.close()

    st = os.stat(slurm_script_fullname)
    os.chmod(slurm_script_fullname, st.st_mode | stat.S_IEXEC)
    # launch scripts (1 launch per 8 seeds)
    if launch:
        for i in range(int(nb_seeds) // 8):
            print('starting from seed {}'.format((i * 8) + global_seed_offset))
            subprocess.check_call(
                ['sbatch', 'campain_logs/scripts/{}.sh'.format(exp_name), str((i * 8) + seed_offset_to_use)])
    if incremental:
        global_seed_offset += int(nb_seeds)
