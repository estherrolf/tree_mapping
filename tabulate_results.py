# imports
from eval_utils import compare_aligned_data
from experiment_utils import get_site_splits
from utils import get_project_dir
import json
import numpy as np
import os
import rasterio

project_dir = get_project_dir()
outputs_dir = '../../../tambe_lab/Everyone/model_output'
resolution = 10
split_seed = 10
pred_padding = 40 # defined by how we cropped the satellite imagery to the Karingani sites
height_interval_bounds = [0, 3, 6, 10, 30]
num_height_intervals = len(height_interval_bounds) - 1
lidar_coarsened_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_{resolution}m'
sites = sorted([x for x in os.listdir(lidar_coarsened_dir) if not x.startswith('.')])
reference_maps = ['ETH', 'GLAD']

# models
models_rf = [f'local_only_models/rf/{x}' for x in ['rf_3_channel', 'rf_4_channel', 'rf_12_channel', 'rf_15_channel']] 
models_fcn = [f'local_only_models/128_filters/{x}' for x in ['3_channels', '4_channels', '12_channels', '15_channels']]
models_xception = [f'finetune_xceptionS2/{x}/{y}' for y in ['1_layers_tuned', '2_layers_tuned', '3_layers_tuned'] for x in ['pretrained', 'randominit_nolatlon', 'randominit_latlon']]
models_unet = ['unet/randominit/12_channels'] + [f'unet/pretrained/{x}' for x in ['freeze_backbone_True', 'freeze_backbone_False']]
# models = models_rf + models_fcn + models_xception + models_unet
models = models_fcn + models_xception + models_unet

# aggregate predictions and calculate performance metrics
results_aggregated = {x: {} for x in reference_maps + models}
results_by_split = {x: {} for x in reference_maps + models}
results_by_site = {x: {} for x in reference_maps + models}
results_by_height = {x: {} for x in reference_maps + models}

# get results
for reference_map in reference_maps:
    labels = []
    preds = []
    reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'

    for split_number in range(4):
        labels_by_split = []
        preds_by_split = []
        test_sites = get_site_splits(split_seed)[split_number]['test_sites']
        
        for site in test_sites:
            with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
                site_labels = file.read().ravel()
                labels = np.append(labels, site_labels)
                labels_by_split = np.append(labels_by_split, site_labels)

            with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
                site_preds = file.read().ravel()
                preds = np.append(preds, site_preds)
                preds_by_split = np.append(preds_by_split, site_preds)

            results_by_site[reference_map][site] = compare_aligned_data(site_labels, site_preds)
        results_by_split[reference_map][split_number] = compare_aligned_data(labels_by_split, preds_by_split)
    results_aggregated[reference_map] = compare_aligned_data(labels, preds)

    for i in range(num_height_intervals):
        results_by_height[reference_map][f'{height_interval_bounds[i]}-{height_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, interval=[height_interval_bounds[i], height_interval_bounds[i+1]])['errors']

for model in models:
    labels = []
    preds = []
    model_output_dir = f'{outputs_dir}/{model}'
    
    for split_number in range(4):
        labels_by_split = []
        preds_by_split = []
        test_sites = get_site_splits(split_seed)[split_number]['test_sites']
        
        for site in test_sites:
            with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
                site_labels = file.read().ravel()
                labels = np.append(labels, site_labels)
                labels_by_split = np.append(labels_by_split, site_labels)

            with rasterio.open(f'{model_output_dir}/preds_{site}.tif') as file: # opens prediction tiff
                site_preds = file.read()[:, pred_padding:-pred_padding, pred_padding:-pred_padding].ravel()
                preds = np.append(preds, site_preds)
                preds_by_split = np.append(preds_by_split, site_preds)

            results_by_site[model][site] = compare_aligned_data(site_labels, site_preds)
        results_by_split[model][split_number] = compare_aligned_data(labels_by_split, preds_by_split)
    results_aggregated[model] = compare_aligned_data(labels, preds)

    for i in range(num_height_intervals):
        results_by_height[model][f'{height_interval_bounds[i]}-{height_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, interval=[height_interval_bounds[i], height_interval_bounds[i+1]])['errors']

# save results as JSON
results = {}

for setting in reference_maps + models:
    results[setting] = {}

    for stratifier in ['total', 'split', 'site', 'height']:
        results[setting][stratifier] = {}

        if stratifier == 'total':
            for metric in ['r2', 'mae', 'mse', 'rmse']:
                results[setting][stratifier][metric] = float(results_aggregated[setting][metric])
        elif stratifier == 'split':
            for split in results_by_split[setting]:
                results[setting][stratifier][split] = {}
                
                for metric in ['r2', 'mae', 'mse', 'rmse']:
                    results[setting][stratifier][split][metric] = float(results_by_split[setting][split][metric])
        elif stratifier == 'site':
            for site in results_by_site[setting]:
                results[setting][stratifier][site] = {}
                
                for metric in ['r2', 'mae', 'mse', 'rmse']:
                    results[setting][stratifier][site][metric] = float(results_by_site[setting][site][metric])
        elif stratifier == 'height':
            for interval in results_by_height[setting]:
                results[setting][stratifier][interval] = {'errors': results_by_height[setting][interval].tolist()}

with open('results.json', 'w') as file:
    json.dump(results, file, indent=4)
