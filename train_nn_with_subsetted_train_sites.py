# imports
from experiment_utils import read_config_file, get_site_splits
from generate_nn_predictions import find_best_hp_run, run_experiment_models_through_one_split
from utils import get_project_dir
import numpy as np
import os
import random
import subprocess
import sys
import torch
import yaml

project_dir = get_project_dir()
split_seed = 10
n_trials = 10
train_site_counts = [3, 6, 9, 12]
config_name = sys.argv[1]
config_file = read_config_file(config_yaml=f'experiment_configs/{config_name}')
variable = sys.argv[2] # channels, layers_tuned, or freeze_backbone
variable_value = sys.argv[3] # int or Bool

if variable != 'freeze_backbone':
    setting_dir = f"{config_file['exp_name']}/{config_file['version_id_base']}/{variable_value}_{variable}"
else:
    setting_dir = f"{config_file['exp_name']}/{config_file['version_id_base']}/{variable}_{variable_value}"
print(setting_dir)

for split in range(4):
    print(f'Split {split}')

    best_run = find_best_hp_run(setting_dir=setting_dir, split=split)
    print(best_run)     

    for seed in range(n_trials):
        print(f'Seed {seed}')

        for train_site_count in train_site_counts:
            print(f'{train_site_count} train sites')

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # train model for this split's best hyperparameters
            subprocess.run(f"sbatch job.sh {config_name} {best_run.split('_')[1]} {best_run.split('_')[3]} --{variable}={variable_value} --train_sites={train_site_count} --split={split} --seed={seed}", shell=True)
