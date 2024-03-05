from hp_search_random_forest import make_tabular_dataset, train_rf
from experiment_utils import read_config_file, get_site_splits
from model_to_tif import predict_site_with_model

from datamodules.chm_datamodule import get_default_layers_and_transforms

import pandas as pd
import numpy as np
import os
import sklearn
import sys


def find_best_hp_run(hp_runs, split, criterion, minimize):
    runs_this = hp_runs[hp_runs['split'] == split]
    if minimize: best_run = runs_this.iloc[np.argmin(runs_this[criterion])]
    else: best_run = runs_this.iloc[np.argmax(runs_this[criterion])]
    
    return best_run

def read_best_run_rf(run_row):
    return {
        'max_depth': run_row['max_depth'],
        'n_estimators': run_row['n_estimators'],
    }
            
def predict_test_sites_with_rf(test_sites, model, cfg_data, save_name, save_dir='data/model_output'):
    
    num_image_channels = cfg_data['num_image_channels']
    img_layers, transforms_pred_model = get_default_layers_and_transforms(num_image_channels)
        
    pred_args = {'patch_size': 64, 
                 'padding': 5,
                 'batch_size': 1,
                 'num_workers': 1,
                 'stride': 54}
    
    for site_id in test_sites:
        print(site_id)
        output_fp = os.path.join(save_dir, f'{save_name}/{site_id}_{save_name}.tif')
        if not os.path.exists(os.path.join(save_dir)): os.mkdir(save_dir)
        if not os.path.exists(os.path.join(save_dir,save_name)): os.mkdir(os.path.join(save_dir,save_name))
        
        predict_site_with_model(site_id, 
                            model,
                            output_fp,
                            img_layers=img_layers,
                            transforms=transforms_pred_model,
                            pred_args = pred_args,
                            nodata_value = -9999.,
                            img_nodata_value = -9999.,
                            nodata_pad= 0,
                            model_is_random_forest=True,
                            device='cpu')
        
        
        
if __name__ == '__main__':
    config_fp = sys.argv[1]
    cfg_hp_search = read_config_file(config_fp)
    
    # cfg_hp_search = read_config_file('experiment_configs/tune_rf_4_channel.yaml')
    hp_results_dir = 'tune_results'
    hp_summary_file = os.path.join(hp_results_dir, f'{cfg_hp_search["exp_name"]}_{cfg_hp_search["version_id_base"]}.csv')
    hp_runs = pd.read_csv(hp_summary_file)
    
    cfg_data = cfg_hp_search['data']
    save_name = f'rf_{cfg_hp_search["version_id_base"]}'

    splits_to_do = cfg_hp_search['splits_to_do']
    split_seed = cfg_data['split_seed']
    
    for this_split_number in splits_to_do:
        print(f'split {this_split_number}')
        # get HPs for this split
        best_run = find_best_hp_run(hp_runs, this_split_number, 'mse', minimize=True)
        best_hps = read_best_run_rf(best_run)

        model_kwargs = {'model_random_seed': cfg_hp_search['task']['random_seed'],
                    'criterion': cfg_hp_search['task']['criterion'],
                    'return_model': True,
                   }
        model_kwargs.update(best_hps)

        # get train dataset for this split
        x_train, y_train, x_val, y_val = make_tabular_dataset(cfg_data, this_split_number)

        # train model for this split's best hps
        model, perf = train_rf(x_train,y_train,x_val,y_val, **model_kwargs)

        # run model forward and save tif in test sites
        test_sites = get_site_splits(split_seed)[this_split_number]['test_sites']
        predict_test_sites_with_rf(test_sites, model, cfg_data, save_name)