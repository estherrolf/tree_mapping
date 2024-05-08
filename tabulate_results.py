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
geologies = ['Igneous', 'Unconsolidated', 'Sedimentary', 'Dambo colluvium', 'Undifferentiated']
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
                          'unet/pretrained/freeze_backbone_False']

def get_results():
    results_aggregated = {x: {} for x in reference_maps + models}
    results_by_split = {x: {} for x in reference_maps + models}
    results_by_site = {x: {} for x in reference_maps + models}
    results_by_height = {x: {} for x in reference_maps + models}

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

                results_by_site[reference_map][site] = compare_aligned_data(site_labels, site_preds, reference_map=True)
            results_by_split[reference_map][split_number] = compare_aligned_data(labels_by_split, preds_by_split, reference_map=True)
        results_aggregated[reference_map] = compare_aligned_data(labels, preds, reference_map=True)

        for i in range(num_height_intervals):
            results_by_height[reference_map][f'{height_interval_bounds[i]}-{height_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, interval=[height_interval_bounds[i], height_interval_bounds[i+1]], reference_map=True)

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
            results_by_height[model][f'{height_interval_bounds[i]}-{height_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, interval=[height_interval_bounds[i], height_interval_bounds[i+1]])

    return results_aggregated, results_by_split, results_by_site, results_by_height

def get_results_by_river():
    results_by_river = {x: {} for x in reference_maps + models}

    for reference_map in reference_maps:
        reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'
        
        for i in range(num_river_intervals):
            labels = []
            preds = []

            for site in sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                    site_labels = file.read().ravel()

                with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file:
                    site_preds = file.read().ravel()

                site_river_distances = np.load(f'../../../tambe_lab/Everyone/features/river/distances_to_river_{resolution}m/{site}_distances_to_river_{resolution}m.npy').ravel()
                mask = (site_river_distances >= river_interval_bounds[i]) & (site_river_distances <= river_interval_bounds[i+1])
                assert len(site_labels) == len(mask)
                labels = np.append(labels, site_labels[mask])
                preds = np.append(preds, site_preds[mask])
            
            results_by_river[reference_map][f'{river_interval_bounds[i]}-{river_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds, reference_map=True)

    for model in models:
        model_output_dir = f'{outputs_dir}/{model}'
        
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
            
            results_by_river[model][f'{river_interval_bounds[i]}-{river_interval_bounds[i+1]}'] = compare_aligned_data(labels, preds)
    
    return results_by_river

def get_results_by_geology():
    results_by_geology = {x: {} for x in reference_maps + models}

    for reference_map in reference_maps:
        reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'
        
        for i in range(num_geologies):
            labels = []
            preds = []

            for site in sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                    site_labels = file.read().ravel()

                with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file:
                    site_preds = file.read().ravel()
                # print(len(site_labels))
                # print(len(site_preds))
                geology_array = np.load(f'../../../tambe_lab/Everyone/features/geology/geology_{resolution}m/{site}_geology_{resolution}m.npy').ravel()
                # print(len(geology_array))
                mask = geology_array == i+1
                # print(len(mask))
                assert len(site_labels) == len(mask)
                labels = np.append(labels, site_labels[mask])
                preds = np.append(preds, site_preds[mask])
            
            results_by_geology[reference_map][f'{geologies[i]}'] = compare_aligned_data(labels, preds, reference_map=True)

    for model in models:
        model_output_dir = f'{outputs_dir}/{model}'
        
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
            
            results_by_geology[reference_map][f'{geologies[i]}'] = compare_aligned_data(labels, preds)

    return results_by_geology

def save_results_as_JSON(results_aggregated, results_by_split, results_by_site, results_by_height, results_by_river):
    results = {}

    for setting in reference_maps + models:
        results[setting] = {}

        for stratifier in ['total', 'split', 'site', 'height', 'river']:
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
                    results[setting][stratifier][interval] = {}

                    for metric in ['perc_10', 'q1', 'median', 'q3', 'perc_90']:
                        results[setting][stratifier][interval][metric] = float(results_by_height[setting][interval][metric])
            elif stratifier == 'river':
                for interval in results_by_river[setting]:
                    results[setting][stratifier][interval] = {}

                    for metric in ['perc_10', 'q1', 'median', 'q3', 'perc_90']:
                        results[setting][stratifier][interval][metric] = float(results_by_river[setting][interval][metric])
            elif stratifier == 'geology':
                for category in results_by_geology[setting]:
                    results[setting][stratifier][category] = {}

                    for metric in ['perc_10', 'q1', 'median', 'q3', 'perc_90']:
                        results[setting][stratifier][category][metric] = float(results_by_geology[setting][category][metric])

    with open('results.json', 'w') as file:
        json.dump(results, file, indent=4)

def get_reference_map_results():
    results = {}

    for reference_map in reference_maps:
        results[reference_map] = {}

        results_aggregated = []
        results_by_split = {}
        results_by_site = {}
        results_by_height = {}
        results_by_river = {}
        results_by_geology = {}
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

                with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file:
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
                print(site)
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                    print(file.read().shape)
                    site_labels = file.read().ravel()

                with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file:
                    site_preds = file.read().ravel()
                # print(len(site_labels))
                # print(len(site_preds))
                # geology_array = np.load(f'../../../tambe_lab/Everyone/features/geology/geology_{resolution}m/{site}_geology_{resolution}m.npy').ravel()
                geology_array = np.load(f'../../../tambe_lab/Users/luciagordon/tree_mapping/data/features/geology/geology_{resolution}m/{site}_geology_{resolution}m.npy').ravel()
                print(np.load(f'../../../tambe_lab/Users/luciagordon/tree_mapping/data/features/geology/geology_{resolution}m/{site}_geology_{resolution}m.npy').shape)
                # print(len(geology_array))
                mask = geology_array == i+1
                # print(len(mask))
                assert len(site_labels) == len(mask)
                labels = np.append(labels, site_labels[mask])
                preds = np.append(preds, site_preds[mask])
            
            results_by_geology[f'{geologies[i]}'] = compare_aligned_data(labels, preds)

        results[reference_map]['total'] = results_aggregated
        results[reference_map]['split'] = results_by_split
        results[reference_map]['site'] = results_by_site
        results[reference_map]['height'] = results_by_height
        results[reference_map]['river'] = results_by_river
        results[reference_map]['geology'] = results_by_geology

    return results

def save_subset_train_results_as_JSON():
    results_train_subset = {}

    for model in models_subset_training:
        results_train_subset[model] = {}

        if model in models_subset_train_sites:
            train_site_counts = [3, 6]
        else:
            train_site_counts = [12]

        for n_train_sites in train_site_counts:
            results_train_subset[model][n_train_sites] = {}

            for seed in range(10):
                results_train_subset[model][n_train_sites][seed] = {'total': {}, 'split': {}, 'site': {}}

                results_aggregated = []
                results_by_split = {}
                results_by_site = {}
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

                results_train_subset[model][n_train_sites][seed]['total'] = results_aggregated
                results_train_subset[model][n_train_sites][seed]['split'] = results_by_split
                results_train_subset[model][n_train_sites][seed]['site'] = results_by_site

    # save results as JSON
    results_subset_train_dict = {}

    for setting in models_subset_training:
        results_subset_train_dict[setting] = {}

        if setting in models_subset_train_sites:
            train_site_counts = [3, 6]
        else:
            train_site_counts = [12]

        for n_train_sites in train_site_counts:
            results_subset_train_dict[setting][f'{n_train_sites} train sites'] = {}

            for seed in range(10):
                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'] = {}

                for stratifier in ['total', 'split', 'site']:
                    results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier] = {}

                    if stratifier == 'total':
                        for metric in ['r2', 'mae', 'mse', 'rmse']:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][metric] = float(results_train_subset[setting][n_train_sites][seed]['total'][metric])
                    elif stratifier == 'split':
                        for split in results_train_subset[setting][n_train_sites][seed]['split']:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][split] = {}
                            
                            for metric in ['r2', 'mae', 'mse', 'rmse']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][split][metric] = float(results_train_subset[setting][n_train_sites][seed]['split'][split][metric])
                    elif stratifier == 'site':
                        for site in results_train_subset[setting][n_train_sites][seed]['site']:
                            results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][site] = {}
                            
                            for metric in ['r2', 'mae', 'mse', 'rmse']:
                                results_subset_train_dict[setting][f'{n_train_sites} train sites'][f'seed {seed}'][stratifier][site][metric] = float(results_train_subset[setting][n_train_sites][seed]['site'][site][metric])

    with open('results_subset_train.json', 'w') as file:
        json.dump(results_subset_train_dict, file, indent=4)

if __name__ == '__main__':
    # results_aggregated, results_by_split, results_by_site, results_by_height = get_results()
    # results_by_river = get_results_by_river()
    # results_by_geology = get_results_by_geology()
    # save_results_as_JSON(results_aggregated, results_by_split, results_by_site, results_by_height, results_by_river)
    save_subset_train_results_as_JSON()
    # get_reference_map_results()
