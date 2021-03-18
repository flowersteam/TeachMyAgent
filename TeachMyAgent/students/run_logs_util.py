import json
import pickle
import os
import pandas as pd

def get_spinup_run_logs(root, exp_idx, units, condition=None):
    exp_name = None
    try:
        config_path = open(os.path.join(root, 'config.json'))
        config = json.load(config_path)
        if 'exp_name' in config:
            exp_name = config['exp_name']

    except:
        print('No file named config.json')
    condition1 = condition or exp_name or 'exp'
    condition2 = condition1 + '-' + str(exp_idx)
    if condition1 not in units:
        units[condition1] = 0
    unit = units[condition1]
    units[condition1] += 1

    try:
        exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
    except Exception as err:
        print(err)
        print('no progress data, aborting')
        return None

    exp_data.insert(len(exp_data.columns), 'Unit', unit)
    exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
    exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
    exp_data.insert(len(exp_data.columns), 'training return', exp_data['AverageEpRet'])
    if 'AverageTestEpRet' in exp_data:
        exp_data.insert(len(exp_data.columns), 'evaluation return', exp_data['AverageTestEpRet'])

    data_dict = exp_data.to_dict("list")
    data_dict['total timesteps'] = []
    for e in data_dict['Epoch']:
        data_dict['total timesteps'].append(e * config['steps_per_epoch'])
    data_dict['config'] = config

    return data_dict

def get_baselines_run_logs(root, exp_idx, units, condition=None):
    condition1 = condition or 'exp'
    condition2 = condition1 + '-' + str(exp_idx)
    if condition1 not in units:
        units[condition1] = 0
    unit = units[condition1]
    units[condition1] += 1

    try:
        exp_data = pd.read_csv(os.path.join(root, 'progress.csv'))
    except Exception as err:
        print(err)
        print('no progress data, aborting')
        return None

    exp_data.insert(len(exp_data.columns), 'Unit', unit)
    exp_data.insert(len(exp_data.columns), 'Condition1', condition1)
    exp_data.insert(len(exp_data.columns), 'Condition2', condition2)
    exp_data.insert(len(exp_data.columns), 'training return', exp_data['eprewmean'])
    if 'eval_eprewmean' in exp_data:
        exp_data.insert(len(exp_data.columns), 'evaluation return', exp_data['eval_eprewmean'])

    data_dict = exp_data.to_dict("list")
    data_dict['total timesteps'] = exp_data["misc/total_timesteps"]
    splitted_name = root.split("_")
    seed = splitted_name[len(splitted_name) - 1].replace('s', '')
    data_dict['config'] = {
        "seed": seed
    }

    return data_dict

def get_run_logs(logdir, min_len=4):
    """
        Recursively look through logdir for output files produced by
        Assumes that any file "progress.txt/csv" is a valid hit.
    """
    exp_idx = 0
    units = dict()
    datasets = []
    for root, _, files in os.walk(logdir):
        data_dict = None
        if 'progress.txt' in files: # Spinup
            print("Reading spinup run...")
            data_dict = get_spinup_run_logs(root, exp_idx, units)
        elif 'progress.csv' in files: # OpenAI Baselines
            print("Reading baselines run...")
            data_dict = get_baselines_run_logs(root, exp_idx, units)

        exp_idx += 1
        if data_dict is None:
            continue

        run_name = root[13:]

        nb_epochs = len(data_dict['total timesteps'])
        print('{} -> {}'.format(run_name, nb_epochs))
        if nb_epochs >= min_len:
            if 'env_params_save.pkl' in files:
                try:
                    env_params_dict = pickle.load(open(os.path.join(root, 'env_params_save.pkl'), "rb"))
                    for k, v in env_params_dict.items():
                        data_dict[k] = v
                except EOFError as err:
                    print(err)
                    print('Corrupted save, ignoring {}'.format(root[-1])) #data_dict['config']['seed']))
                except pickle.UnpicklingError as err:
                    print(err)
                    print('Corrupted save, ignoring {}'.format(root[-1:]))
                    #continue
            datasets.append(data_dict)
    return datasets

