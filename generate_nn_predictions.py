# imports
from datamodules.chm_datamodule import get_default_layers_and_transforms
from experiment_utils import read_config_file, get_site_splits
from model_to_tif import load_ckpt_weights_to_model, predict_site_with_model
from utils import get_project_dir
import numpy as np
import os
import pandas as pd
import sys

project_dir = get_project_dir()

def find_best_hp_run(setting_dir, split):
    combinations = os.listdir(f'{project_dir}/experiment_results/{setting_dir}/split_{split}/logs')
    results = []

    for combination in combinations:
        results += [np.load(f'{project_dir}/experiment_results/{setting_dir}/split_{split}/logs/{combination}/val_loss.npy').tolist()]
    
    where_min = np.argwhere(results == np.min(results))[0]
    hp_combination = combinations[where_min[0]]
    
    return hp_combination

def run_experiment_models_through_one_split(split_dir,
                                            best_run,
                                            test_sites,
                                            output_dir):

    hparams_path = f'{split_dir}/logs/{best_run}/hparams.yaml'
    hparams = read_config_file(hparams_path)
    del hparams['ignore']
    print(hparams)
    num_image_channels = hparams['in_channels']
    img_layers, transforms_pred_model = get_default_layers_and_transforms(num_image_channels)
    patch_size = 64
    padding = hparams['pad_pixels']
    stride = patch_size - 2*padding

    pred_args = {'patch_size': patch_size, 
                 'padding': padding,
                 'batch_size': 1,
                 'num_workers': 1,
                 'stride': stride}

    for test_site in test_sites:
        print(test_site)
        output_fp = f'{output_dir}/preds_{test_site}.tif'

        model, _ = load_ckpt_weights_to_model(task_conditions_dict=hparams, checkpoint_dir=f'{split_dir}/models/{best_run}')

        predict_site_with_model(site_id=test_site,
                                model=model, 
                                output_fp=output_fp,
                                img_layers=img_layers,
                                pred_args=pred_args, 
                                transforms=transforms_pred_model,
                                nodata_value=-9999.0,
                                img_nodata_value=-9999.0,
                                nodata_pad=padding,
                                device='cuda')

def generate_nn_predictions(setting_dir, subsetted_train_sites=False):    
    if not subsetted_train_sites:
        output_dir = f'{project_dir}/model_output/{setting_dir}'
        os.makedirs(output_dir, exist_ok=True)

        for split in range(4):
            print(f'Split {split}')

            best_run = find_best_hp_run(setting_dir=setting_dir, split=split)
            test_sites = get_site_splits(random_seed=10)[split]['test_sites']

            run_experiment_models_through_one_split(split_dir=f'{project_dir}/experiment_results/{setting_dir}/split_{split}',
                                                    best_run=best_run,
                                                    test_sites=test_sites,
                                                    output_dir=output_dir)
    else:
        for split in range(4):
            print(f'Split {split}')

            test_sites = get_site_splits(random_seed=10)[split]['test_sites']

            for seed in range(10):
                for train_site_count in [12]:
                # for train_site_count in [3, 6, 9, 12]:
                    best_run = os.listdir(f'{project_dir}/experiment_results/{setting_dir}/{train_site_count}_train_sites/seed_{seed}/split_{split}/models')[0] # only one hp combo since we already picked the best one
                    output_dir = f'{project_dir}/model_output/{setting_dir}/{train_site_count}_train_sites/seed_{seed}'
                    os.makedirs(output_dir, exist_ok=True)

                    run_experiment_models_through_one_split(split_dir=f'{project_dir}/experiment_results/{setting_dir}/{train_site_count}_train_sites/seed_{seed}/split_{split}',
                                                            best_run=best_run,
                                                            test_sites=test_sites,
                                                            output_dir=output_dir)

if __name__ == '__main__':
    setting_dir = sys.argv[1] # e.g., "local_only_models/128_filters/3_channels"
    subsetted_train_sites = True if 'subset' in setting_dir else False
    generate_nn_predictions(setting_dir=setting_dir, subsetted_train_sites=subsetted_train_sites)
