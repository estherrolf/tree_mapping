from hp_search_random_forest import train_rf
from experiment_utils import read_config_file, get_site_splits
from train_models import setup_chm_datamodule
from dataloader_to_static_datasets import chm_from_config, datset_to_xy
from model_to_tif import predict_site_with_model

from datamodules.chm_datamodule import get_default_layers_and_transforms

import pandas as pd
import numpy as np
import os
import sklearn
import sys
import time


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

def make_tabular_dataset_subset_train(data_config, split_number, return_test=False, patch_size=32, num_train_sites=6, train_sample_seed=0):
    chm, sampled_train_sites = chm_from_config_subset_train(data_config, split_number, num_train_sites=num_train_sites, train_sample_seed=train_sample_seed)
    chm.setup('fit')
        
    x_val, y_val = datset_to_xy(chm.val_dataset, patch_size=32)
    x_train, y_train= datset_to_xy(chm.train_dataset, patch_size=32)
    
    if return_test:
        x_test, y_test = datset_to_xy(chm.test_dataset, patch_size=32)
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    return x_train, y_train, x_val, y_val, sampled_train_sites

def chm_from_config_subset_train(data_config, split_number, num_train_sites, train_sample_seed=0):
    
    
    # get this data split  
    split_seed = data_config['split_seed']
    splits = get_site_splits(split_seed)
    sites_per_split = splits[split_number]
    
    # this will take random samples such that for the same train_sample_seed value,
    # a sample of 3 will be contained within the sample of 6, which will be contained
    # in the sample of 9, etc.
    rs_train_sample = np.random.RandomState(train_sample_seed)
    train_site_sample = rs_train_sample.choice(len(sites_per_split['train_sites']), 
                                                   num_train_sites, 
                                                   replace=False)
    
    sampled_train_sites = [sites_per_split['train_sites'][x] for x in train_site_sample]
    sites_per_split['train_sites'] = sampled_train_sites
    # make the data module
    chm = setup_chm_datamodule(sites_per_split, data_config)
    
    return chm, sampled_train_sites

            
def predict_test_sites_with_rf(test_sites, model, cfg_data, save_name, save_dir='model_output'):
    
    num_image_channels = cfg_data['num_image_channels']
    img_layers, transforms_pred_model = get_default_layers_and_transforms(num_image_channels)
    
    patch_size = cfg_data['datamodule']['patch_size']
    pred_padding = cfg_data['datamodule']['eval_pad']
        
    pred_args = {'patch_size': patch_size, 
                 'padding': pred_padding,
                 'batch_size': 1,
                 'num_workers': 1,
                 'stride': patch_size - 2*pred_padding}
    
    for site_id in test_sites:
        print(site_id)
        output_fp = os.path.join(save_dir, f'{save_name}/preds_{site_id}.tif')
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
    # cfg_hp_search will determine the number of layers as well
    
    n_trials = 10
    num_train_sites = [3,6,9,12]
    
    # cfg_hp_search = read_config_file('experiment_configs/tune_rf_4_channel.yaml')
    hp_results_dir = 'tune_results'
    hp_summary_file = os.path.join(hp_results_dir, f'{cfg_hp_search["exp_name"]}_{cfg_hp_search["version_id_base"]}.csv')
    hp_runs = pd.read_csv(hp_summary_file)
    
    data_cfg = cfg_hp_search['data']
    save_name = f'rf_{cfg_hp_search["version_id_base"]}'

    splits_to_do = cfg_hp_search['splits_to_do']
    split_seed = data_cfg['split_seed']
    
    for this_split_number in splits_to_do:
        print(f'split {this_split_number}')
        # get HPs for this split (from the 12 training site conditions)
        best_run = find_best_hp_run(hp_runs, this_split_number, 'mse', minimize=True)
        best_hps = read_best_run_rf(best_run)

        model_kwargs = {'model_random_seed': cfg_hp_search['task']['random_seed'],
                    'criterion': cfg_hp_search['task']['criterion'],
                    'return_model': True,
                   }
        model_kwargs.update(best_hps)
        
        t1 = time.time()
        for i in range(n_trials):
            for n_train_sites in num_train_sites:
                print(n_train_sites, end = ' ')
                
                x_train, y_train, x_val, y_val, sampled_train_sites = make_tabular_dataset_subset_train(data_cfg, 
                                                                                                        this_split_number, 
                                                                                                        return_test=False, 
                                                                                                        num_train_sites=n_train_sites, 
                                                                                                        train_sample_seed=i)
                # set the model random seed to get variation even when using 12 train sites
                model_kwargs['model_random_seed'] = i
                
                # train model for this split's best hps
                model, perf = train_rf(x_train,y_train,x_val,y_val, **model_kwargs)

                # run model forward and save tif in test sites
                test_sites = get_site_splits(split_seed)[this_split_number]['test_sites']
                
                # save directory formatting
                save_dir=f'model_output/subsetted_training_sets/{n_train_sites}_train_sites/seed_{i}'
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                
                # run through the test set to make tifs of this models output
                predict_test_sites_with_rf(test_sites, model, data_cfg, save_name,save_dir=save_dir)
                
            print()
        print(f'split {this_split_number} took {(time.time()-t1)/60:.2f} minutes')