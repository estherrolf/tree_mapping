from dataloader_to_static_datasets import make_tabular_dataset
from train_models import read_config_file

import numpy as np
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import sklearn.metrics
import sys
import time


def train_rf(x_train,
             y_train,
             x_val,
             y_val,
             criterion,
             n_estimators,
             max_depth,
             model_random_seed,
             return_model=False):
    
    model = RandomForestRegressor(n_estimators = n_estimators,
                              criterion = criterion, 
                              max_depth = max_depth,
                              random_state= model_random_seed,
                             )
    t1 = time.time()
    model.fit(x_train, y_train)
    t2 = time.time()
    
    yhat_val = model.predict(x_val)
    yhat_train = model.predict(x_train)
    
    perf = {'mse': sklearn.metrics.mean_squared_error(y_val, yhat_val),
            'mae': sklearn.metrics.mean_absolute_error(y_val, yhat_val),
            'rmse': np.sqrt(sklearn.metrics.mean_squared_error(y_val, yhat_val)),
            'r2': sklearn.metrics.r2_score(y_val, yhat_val),
            'mse_train': sklearn.metrics.mean_squared_error(y_train, yhat_train),
            'mae_train': sklearn.metrics.mean_absolute_error(y_train, yhat_train),
            'rmse_train': np.sqrt(sklearn.metrics.mean_squared_error(y_train, yhat_train)),
            'r2_train': sklearn.metrics.r2_score(y_train, yhat_train),
            'train_time': t2-t1
           }
    
    if return_model:
        return model, perf
    else:
        return perf
    
    

if __name__ == "__main__":
    config_fp = sys.argv[1]
    cfg = read_config_file(config_fp)
    
    hp_results_dir = 'tune_results'
    if not os.path.exists(hp_results_dir): os.mkdir(hp_results_dir)
    outfile = os.path.join(hp_results_dir, f'{cfg["exp_name"]}_{cfg["version_id_base"]}.csv')
    

    split_seed = cfg['data']['split_seed']
    splits_to_do = cfg['splits_to_do']
    # for now do one, should iterate
    
    max_depths_hp_search = [2,4,8,16]
    n_estimators_hp_search = [50,100,200]
    
    rows = []
    for this_split_number in splits_to_do:

        x_train, y_train, x_val, y_val = make_tabular_dataset(cfg['data'], this_split_number)
        
        for max_depth in max_depths_hp_search:
            for n_estimators in n_estimators_hp_search:
                
                row = {'split': this_split_number,
                       'split_seed': split_seed,
                       'num_image_channels': cfg['data']['num_image_channels']
                      }
                
                hparams = {'model_random_seed': cfg['task']['random_seed'],
                           'max_depth': max_depth, 
                           'n_estimators': n_estimators,
                           'criterion': cfg['task']['criterion'],
                          }
                row.update(hparams)
                
                # train and add results
                results_row = train_rf(x_train, y_train, x_val, y_val, **hparams)                
                row.update(results_row)

                # save most recent                
                rows.append(row)
                pd.DataFrame(rows).to_csv(outfile)