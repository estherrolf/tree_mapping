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
river_interval_bounds = [0, 100, 300, 600, 1000, 2000, 3000]
geologies = ['Igneous', 'Unconsolidated', 'Sedimentary', 'Dambo colluvium']
num_height_intervals = len(height_interval_bounds) - 1
num_river_intervals = len(river_interval_bounds) - 1
num_geologies = len(geologies)
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
models_subset_train_sites = ['local_only_models/128_filters/12_channels', 'finetune_xceptionS2/pretrained/3_layers_tuned']
models_subset_training = ['local_only_models/128_filters/3_channels',
                          'local_only_models/128_filters/4_channels',
                          'local_only_models/128_filters/12_channels',
                          'local_only_models/128_filters/15_channels',
                          'finetune_xceptionS2/pretrained/1_layers_tuned',
                          'finetune_xceptionS2/pretrained/2_layers_tuned',
                          'finetune_xceptionS2/pretrained/3_layers_tuned',
                          'finetune_xceptionS2/randominit_nolatlon/1_layers_tuned',
                          'finetune_xceptionS2/randominit_nolatlon/2_layers_tuned',
                          'finetune_xceptionS2/randominit_nolatlon/3_layers_tuned',
                          'finetune_xceptionS2/randominit_latlon/1_layers_tuned',
                          'finetune_xceptionS2/randominit_latlon/2_layers_tuned',
                          'finetune_xceptionS2/randominit_latlon/3_layers_tuned',
                          'unet/randominit/12_channels',
                          'unet/pretrained/freeze_backbone_True',
                          'unet/pretrained/freeze_backbone_False'
                          ]

def save_subset_train_results_as_JSON():
    results_train_subset = {}

    # reference maps
    for model in reference_maps:
        results_train_subset[model] = {'total': {}, 'split': {}, 'site': {}}

        results_aggregated = []
        results_by_split = {}
        results_by_site = {}
        results_by_height = {}
        results_by_river = {}
        results_by_geology = {}
            
        labels = []
        preds = []
        reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{model.lower()}_maps_per_site_{resolution}m'

        for split_number in range(4):
            labels_by_split = []
            preds_by_split = []
            test_sites = get_site_splits(split_seed)[split_number]['test_sites']

            for site in test_sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
                    site_labels = file.read().ravel()
                    labels = np.append(labels, site_labels)
                    labels_by_split = np.append(labels_by_split, site_labels)

                with rasterio.open(f'{reference_map_by_site_dir}/{model}_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
                    site_preds = file.read().ravel()
                    preds = np.append(preds, site_preds)
                    preds_by_split = np.append(preds_by_split, site_preds)

                results_by_site[site] = compare_aligned_data(site_labels, site_preds)
            results_by_split[split_number] = compare_aligned_data(labels_by_split, preds_by_split)
        results_aggregated = compare_aligned_data(labels, preds)

        # stratify by height
        for i in range(num_height_intervals):
            results_by_height[f'{height_interval_bounds[i]}-{height_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, interval=[height_interval_bounds[i], height_interval_bounds[i+1]])

        # stratify by river distance
        for i in range(num_river_intervals):
            labels = []
            preds = []

            for site in sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                    site_labels = file.read().ravel()

                with rasterio.open(f'{reference_map_by_site_dir}/{model}_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
                    site_preds = file.read().ravel()

                site_river_distances = np.load(f'../../../tambe_lab/Everyone/features/river/distances_to_river_{resolution}m/{site}_distances_to_river_{resolution}m.npy').ravel()
                mask = (site_river_distances >= river_interval_bounds[i]) & (site_river_distances <= river_interval_bounds[i+1])
                assert len(site_labels) == len(mask)
                labels = np.append(labels, site_labels[mask])
                preds = np.append(preds, site_preds[mask])
            
            results_by_river[f'{river_interval_bounds[i]}-{river_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds)

        # stratify by geology
        for i in range(num_geologies):
            labels = []
            preds = []

            for site in sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                    site_labels = file.read().ravel()

                with rasterio.open(f'{reference_map_by_site_dir}/{model}_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
                    site_preds = file.read().ravel()

                geology_array = np.load(f'../../../tambe_lab/Everyone/features/geology/geology_{resolution}m/{site}_geology_{resolution}m.npy').ravel()
                mask = geology_array == i+1
                assert len(site_labels) == len(mask)
                labels = np.append(labels, site_labels[mask])
                preds = np.append(preds, site_preds[mask])

            results_by_geology[f'{geologies[i]}'] = compare_aligned_data(labels, preds)

        results_train_subset[model]['total'] = results_aggregated
        results_train_subset[model]['split'] = results_by_split
        results_train_subset[model]['site'] = results_by_site
        results_train_subset[model]['height'] = results_by_height
        results_train_subset[model]['river'] = results_by_river
        results_train_subset[model]['geology'] = results_by_geology

    # models
    for model in models_subset_training:
        results_train_subset[model] = {}

        train_site_counts = [3, 6, 9, 12] if model in models_subset_train_sites else [12]

        for n_train_sites in train_site_counts:
            results_train_subset[model][n_train_sites] = {}

            for seed in range(10):
                results_train_subset[model][n_train_sites][seed] = {'total': {}, 'split': {}, 'site': {}}

                results_aggregated = []
                results_by_split = {}
                results_by_site = {}
                results_by_height = {}
                results_by_river = {}
                results_by_geology = {}
                    
                labels = []
                preds = []
                model_output_dir = f'{outputs_dir}/subsetted_train_sites/{model}/{n_train_sites}_train_sites/seed_{seed}'

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

                        results_by_site[site] = compare_aligned_data(site_labels, site_preds)
                    results_by_split[split_number] = compare_aligned_data(labels_by_split, preds_by_split)
                results_aggregated = compare_aligned_data(labels, preds)

                # stratify by height
                for i in range(num_height_intervals):
                    results_by_height[f'{height_interval_bounds[i]}-{height_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, interval=[height_interval_bounds[i], height_interval_bounds[i+1]])

                # stratify by river distance
                for i in range(num_river_intervals):
                    labels = []
                    preds = []

                    for site in sites:
                        with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                            site_labels = file.read().ravel()

                        with rasterio.open(f'{model_output_dir}/preds_{site}.tif') as file: # opens prediction tiff
                            site_preds = file.read()[:, pred_padding:-pred_padding, pred_padding:-pred_padding].ravel()

                        site_river_distances = np.load(f'../../../tambe_lab/Everyone/features/river/distances_to_river_{resolution}m/{site}_distances_to_river_{resolution}m.npy').ravel()
                        mask = (site_river_distances >= river_interval_bounds[i]) & (site_river_distances <= river_interval_bounds[i+1])
                        assert len(site_labels) == len(mask)
                        labels = np.append(labels, site_labels[mask])
                        preds = np.append(preds, site_preds[mask])
                    
                    results_by_river[f'{river_interval_bounds[i]}-{river_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds)

                # stratify by geology
                for i in range(num_geologies):
                    labels = []
                    preds = []

                    for site in sites:
                        with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                            site_labels = file.read().ravel()

                        with rasterio.open(f'{model_output_dir}/preds_{site}.tif') as file: # opens prediction tiff
                            site_preds = file.read()[:, pred_padding:-pred_padding, pred_padding:-pred_padding].ravel()

                        geology_array = np.load(f'../../../tambe_lab/Everyone/features/geology/geology_{resolution}m/{site}_geology_{resolution}m.npy').ravel()
                        mask = geology_array == i+1
                        assert len(site_labels) == len(mask)
                        labels = np.append(labels, site_labels[mask])
                        preds = np.append(preds, site_preds[mask])

                    results_by_geology[f'{geologies[i]}'] = compare_aligned_data(labels, preds)

                results_train_subset[model][n_train_sites][seed]['total'] = results_aggregated
                results_train_subset[model][n_train_sites][seed]['split'] = results_by_split
                results_train_subset[model][n_train_sites][seed]['site'] = results_by_site
                results_train_subset[model][n_train_sites][seed]['height'] = results_by_height
                results_train_subset[model][n_train_sites][seed]['river'] = results_by_river
                results_train_subset[model][n_train_sites][seed]['geology'] = results_by_geology

    # save results as JSON
    results_subset_train_dict = {}

    for setting in reference_maps + models_subset_training:
        results_subset_train_dict[setting] = {}

        train_site_counts = [3, 6, 9, 12] if model in models_subset_train_sites else [12]

        for n_train_sites in train_site_counts:
            results_subset_train_dict[setting][f'{n_train_sites} train sites'] = {}

            for seed in range(10):
                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'] = {}

                stratifiers = ['total', 'split', 'site', 'height', 'river', 'geology']

                for stratifier in stratifiers:
                    results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier] = {}

                    if stratifier == 'total':
                        for metric in ['r2', 'mae', 'mse', 'rmse']:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][metric] = float(results_train_subset[setting][n_train_sites][seed][stratifier][metric])
                    elif stratifier == 'split':
                        for split in results_train_subset[setting][n_train_sites][seed][stratifier]:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][split] = {}
                            
                            for metric in ['r2', 'mae', 'mse', 'rmse']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][split][metric] = float(results_train_subset[setting][n_train_sites][seed][stratifier][split][metric])
                    elif stratifier == 'site':
                        for site in results_train_subset[setting][n_train_sites][seed][stratifier]:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][site] = {}
                            
                            for metric in ['r2', 'mae', 'mse', 'rmse']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][site][metric] = float(results_train_subset[setting][n_train_sites][seed][stratifier][site][metric])
                    elif stratifier == 'height':
                        for interval in results_train_subset[setting][n_train_sites][seed][stratifier]:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][interval] = {}

                            for metric in ['perc_10', 'q1', 'median', 'q3', 'perc_90']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][interval][metric] = float(results_train_subset[setting][n_train_sites][seed][stratifier][interval][metric])
                    elif stratifier == 'river':
                        for interval in results_train_subset[setting][n_train_sites][seed][stratifier]:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][interval] = {}

                            for metric in ['perc_10', 'q1', 'median', 'q3', 'perc_90']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][interval][metric] = float(results_train_subset[setting][n_train_sites][seed][stratifier][interval][metric])
                    elif stratifier == 'geology':
                        for category in results_train_subset[setting][n_train_sites][seed][stratifier]:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][category] = {}

                            for metric in ['perc_10', 'q1', 'median', 'q3', 'perc_90']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][category][metric] = float(results_train_subset[setting][n_train_sites][seed][stratifier][category][metric])

    with open('results_subset_train.json', 'w') as file:
        json.dump(results_subset_train_dict, file, indent=4)

if __name__ == '__main__':
    save_subset_train_results_as_JSON()
