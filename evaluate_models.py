# imports
from eval_utils import compare_aligned_data
from experiment_utils import read_config_file, get_site_splits
from utils import get_project_dir
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio
import sklearn.metrics
import seaborn as sns

project_dir = get_project_dir()
os.makedirs('figures', exist_ok=True)
reference_maps = ['ETH', 'GLAD']
eval_metrics = {'r2': 'R\u00b2', 'mae': 'Mean Absolute Error', 'mse': 'Mean Squared Error', 'rmse': 'RMSE', 'me': 'Mean Error'}
pred_padding = 40
split_seed = 10

def evaluate_models(models, resolution, eval_metric):
    lidar_coarsened_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    sites = sorted(os.listdir(lidar_coarsened_dir))
    interval_bounds = [0, 3, 6, 10, 30]
    num_intervals = len(interval_bounds) - 1

    results_by_site = {x: {} for x in reference_maps + models}
    results_by_site_plot = []
    results_aggregated = {x: {} for x in reference_maps + models}
    results_by_split = {x: {} for x in reference_maps + models}
    results_by_interval = {x: {} for x in reference_maps + models}
    results_by_interval_plot = []

    # get results
    for reference_map in reference_maps:
        labels = []
        preds = []
        reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'

        for split_number in np.arange(4):
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

        results_by_site_plot += [[results_by_site[reference_map][site][eval_metric] for site in sites]]
        results_aggregated[reference_map] = compare_aligned_data(labels, preds)

        for i in range(num_intervals):
            results_by_interval[reference_map][f'interval_{i}'] = compare_aligned_data(labels, preds, interval=[interval_bounds[i], interval_bounds[i+1]])['errors']
        
        results_by_interval_plot += [[results_by_interval[reference_map][f'interval_{i}'] for i in range(num_intervals)]]

    for model in models:
        labels = []
        preds = []
        model_output_dir = f'{project_dir}/model_output/{model}'

        for split_number in np.arange(4):
            labels_by_split = []
            preds_by_split = []
            test_sites = get_site_splits(split_seed)[split_number]['test_sites']

            for site in test_sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
                    site_labels = file.read().ravel()
                    labels = np.append(labels, site_labels)
                    labels_by_split = np.append(labels_by_split, site_labels)

                with rasterio.open(f'{model_output_dir}/preds_{site}.tif') as file: # opens prediction tiff
                    site_preds = file.read()[:,pred_padding:-pred_padding,pred_padding:-pred_padding].ravel()
                    preds = np.append(preds, site_preds)
                    preds_by_split = np.append(preds_by_split, site_preds)

                results_by_site[model][site] = compare_aligned_data(site_labels, site_preds)
        
            results_by_split[model][split_number] = compare_aligned_data(labels_by_split, preds_by_split)
        
        results_by_site_plot += [[results_by_site[model][site][eval_metric] for site in sites]]
        results_aggregated[model] = compare_aligned_data(labels, preds)
        
        for i in range(num_intervals):
            results_by_interval[model][f'interval_{i}'] = compare_aligned_data(labels, preds, interval=[interval_bounds[i], interval_bounds[i+1]])['errors']
        
        results_by_interval_plot += [[results_by_interval[model][f'interval_{i}'] for i in range(num_intervals)]]

    print('Got results')

    # plot results by interval
    aE = [np.mean(list(itertools.chain.from_iterable(map_data))) for map_data in results_by_interval_plot] # average error

    fig, ax = plt.subplots(layout='constrained', dpi=300)
    x = np.arange(num_intervals)
    width = 0.1
    tick_positions = sorted(list(np.arange(num_intervals+1) - 7/6*width) + list(x + width/2))
    boxplots = []
    lines = []

    for i in range(len(results_by_interval_plot)):
        pos = x + width*i + width/10*((-1)**(i+1))
        boxplots.append(ax.boxplot(results_by_interval_plot[i], sym='', positions=pos, widths=width, patch_artist=True, boxprops=dict(facecolor=f'C{i}'), medianprops=dict(color='black')))
        # lines.append(ax.plot(np.arange(num_intervals+1) - 7/6*width, (num_intervals+1)*[aE[i]], linestyle='dashed', label=f'{reference_maps[i]} aE'))

    ax.set_xticks(tick_positions, labels=['' if i % 2 == 0 else f'{interval_bounds[int(i/2)]}-{interval_bounds[int(i/2)+1]}' for i in range(2*len(interval_bounds)-1)])
    ax.set_xlabel('LiDAR-Derived Height (m)')
    ax.set_ylabel('Mean Absolute Error (m)')
    ax.set_title('Maps Evaluated by Height Interval')
    ax.legend([boxplots[i]["boxes"][0] for i in range(len(boxplots))], reference_maps + models, ncols=2, bbox_to_anchor=(1,1))
    
    for i in range(len(ax.xaxis.get_major_ticks())):
        if i % 2 != 0:
            ax.xaxis.get_major_ticks()[i].tick1line.set_visible(False)

    plt.savefig(f'figures/maps-evaluated-by-interval-{resolution}m.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Plotted results by height interval')

    # coalesce into pandas dataframes for plotting
    results_keys = ['r2', 'mae', 'mse', 'rmse']
    rows = []
    for map_type, results_this_map_type in results_aggregated.items():
        results_this_type = {k: results_this_map_type[k] for k in results_keys}
        # results_this_type.update({'map_type': map_type.replace('/', '/\n')})
        results_this_type.update({'map_type': map_type})
        rows.append(results_this_type)
    results_agg_df = pd.DataFrame(rows)

    rows = []
    for map_type, results_this_map_type in results_by_split.items():
        for split, results_this_split in results_this_map_type.items():
            results_this_type = {k: results_this_split[k] for k in results_keys}
            # results_this_type.update({'map_type': map_type.replace('/', '/\n'), 'test_split': split})
            results_this_type.update({'map_type': map_type, 'test_split': split})
            rows.append(results_this_type)
    results_by_split_df = pd.DataFrame(rows)

    rows = []
    for map_type, results_this_map_type in results_by_site.items():
        for site, results_this_site in results_this_map_type.items():
            results_this_type = {k: results_this_site[k] for k in results_keys}
            # results_this_type.update({'map_type': map_type.replace('/', '/\n'), 'test_site': site})
            results_this_type.update({'map_type': map_type, 'test_site': site})
            rows.append(results_this_type)
    results_by_site_df = pd.DataFrame(rows)
    print('Made dataframes')

    # scatterplot
    y_key = 'rmse'
    metric_labels = {'mae': "Mean Absolute Error (m)", 'rmse': "Root Mean Squared Error (m)"}
    fig, ax = plt.subplots(1, figsize=(10,5), sharey=True, dpi=300)
    ax = [ax]

    sns.scatterplot(data=results_by_split_df, x='map_type', y=y_key, 
                    s=100,
                    color='lightgrey', label='each of 4 test splits',
                ax=ax[0])
    sns.scatterplot(data=results_by_split_df.groupby('map_type')[[y_key]].mean(), x='map_type', y=y_key, 
                    s=100,
                    color='forestgreen', label='average across test splits',
                ax=ax[0])

    ax[0].set_ylabel(metric_labels[y_key])

    for i in range(len(ax)):
        ax[i].set_xlabel(None)

    sns.despine()
    plt.savefig('figures/rmse_scatterplot.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Made RMSE scatterplot')

    y_key = 'mae'
    metric_labels = {'mae': "Mean Absolute Error (m)"}
    fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True, dpi=300)

    sns.scatterplot(data=results_by_split_df, x='map_type',y=y_key, 
                    s=100,
                    color='lightgrey', label='each of 4 test splits',
                    ax=ax[0])
    sns.scatterplot(data=results_by_split_df.groupby('map_type')[[y_key]].mean(), x='map_type', y=y_key, 
                    s=100,
                    color='forestgreen', label='average across test splits',
                ax=ax[0])
    sns.scatterplot(data=results_by_site_df, x='map_type',y=y_key, 
                    s=100,
                    color='lightgrey', label='each of 24 test sites',
                ax=ax[1])
    sns.scatterplot(data=results_by_site_df.groupby('map_type')[[y_key]].mean(), x='map_type', y=y_key, 
                    s=100,
                    color='forestgreen', label='average across test sites',
                ax=ax[1])

    ax[0].set_ylabel(metric_labels[y_key])

    for i in range(len(ax)):
        ax[i].set_xlabel(None)
    
    for axis in ax:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=90)

    sns.despine()
    plt.savefig('figures/mae_scatterplots.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Made MAE scatterplots')

    y_key = 'mae'
    metric_labels = {'mae': "Mean Absolute Error (m)"}
    fig, ax = plt.subplots(1, figsize=(10,5), sharey=True, dpi=300)
    ax = [ax]
    sns.scatterplot(data=results_by_site_df, x='test_site', y=y_key, 
                    s=100,
                    hue='map_type',
                # color='lightgrey', label='each of 24 test sites',
                ax=ax[0])

    ax[0].set_ylabel(metric_labels[y_key])

    for i in range(len(ax)):
        ax[i].set_xlabel(None)

    sns.despine()

    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=90);
    ax[0].legend(bbox_to_anchor=(1,1))
    plt.savefig('figures/mae_scatterplot_by_site.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Made MAE scatterplot by site')

def visualize_predictions(model, resolution):
    model_output_dir = f'{project_dir}/model_output/{model}'
    existing_maps_dir = f'{project_dir}/data/existing_reference_data'
    lidar_coarsened_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    plot_sites = ['KaringaniMassingirDevNode']#, 'KaringaniSite03']

    for site in plot_sites:
        with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
            site_labels = file.read()[0]

        with rasterio.open(f'{model_output_dir}/preds_{site}.tif') as file: # opens prediction tiff
            site_preds = file.read()[0,pred_padding:-pred_padding,pred_padding:-pred_padding]
            
        with rasterio.open(f'{existing_maps_dir}/glad_maps_per_site_{resolution}m/GLAD_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
            glad_preds = file.read()[0]#,pred_padding:-pred_padding,pred_padding:-pred_padding]
            
        with rasterio.open(f'{existing_maps_dir}/eth_maps_per_site_{resolution}m/ETH_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
            eth_preds = file.read()[0]#,pred_padding:-pred_padding,pred_padding:-pred_padding]
            
        map_labels = [f'lidar-derived labels ({resolution}m)',
                      'local predictions (ours)',
                      'GLAD CHM map',
                      'ETH CHM map']

        maps_to_print = [x.copy() for x in [site_labels, site_preds, glad_preds, eth_preds]]
        mask = site_labels < 0

        for x in maps_to_print:
            x[mask] = np.nan

        fig, ax = plt.subplots(1,len(maps_to_print), figsize=(len(maps_to_print)*5, 5), dpi=300)

        for i, map_to_print in enumerate(maps_to_print):
            ax[i].imshow(map_to_print, vmin=0, vmax=15, cmap='Greens', interpolation='none')
            ax[i].set_title(map_labels[i])

        for axis in ax:
            axis.axis('off')
        
        fig.suptitle(site)
        plt.savefig(f'figures/{model}/{site}_visualization.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f'Visualized {site}')

    plot_sites = ['KaringaniSite01', 'KaringaniSite03', 'KaringaniMassingirDevNode']

    x1, x2, y1, y2 = 200, 300, 100, 200

    for site in plot_sites:
        with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
            site_labels = file.read()[0][y1:y2, x1:x2]

        with rasterio.open(f'{model_output_dir}/preds_{site}.tif') as file: # opens prediction tiff
            site_preds = file.read()[0, pred_padding:-pred_padding, pred_padding:-pred_padding][y1:y2, x1:x2]
            
        with rasterio.open(f'{existing_maps_dir}/glad_maps_per_site_{resolution}m/GLAD_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
            glad_preds = file.read()[0][y1:y2, x1:x2]
            
        with rasterio.open(f'{existing_maps_dir}/eth_maps_per_site_{resolution}m/ETH_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
            eth_preds = file.read()[0][y1:y2, x1:x2]
                    
        maps_to_print = [x.copy() for x in [site_labels, site_preds, glad_preds, eth_preds]]
        mask = site_labels < 0

        for x in maps_to_print:
            x[mask] = np.nan

        fig, ax = plt.subplots(1, len(maps_to_print), figsize=(len(maps_to_print)*5, 5), dpi=300)

        for i, map_to_print in enumerate(maps_to_print):
            ax[i].imshow(map_to_print, vmin=0, vmax=15, cmap='Greens', interpolation='none')
            ax[i].set_title(map_labels[i])

        for axis in ax:
            axis.axis('off')
        
        fig.suptitle(site)
        plt.savefig(f'figures/{model}/{site}_visualization_zoomed_in.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()
        print(f'Visualized {site} zoomed in')

if __name__ == '__main__':
    evaluate_models(models=['local_only_models/128_filters/3_channels', 'local_only_models/128_filters/4_channels', 'local_only_models/128_filters/12_channels', 'local_only_models/128_filters/15_channels', 'finetune_xceptionS2/pretrained/1_layers_tuned', 'finetune_xceptionS2/pretrained/2_layers_tuned', 'finetune_xceptionS2/pretrained/3_layers_tuned', 'finetune_xceptionS2/randominit_nolatlon/1_layers_tuned', 'finetune_xceptionS2/randominit_nolatlon/2_layers_tuned', 'finetune_xceptionS2/randominit_nolatlon/3_layers_tuned', 'finetune_xceptionS2/randominit_latlon/1_layers_tuned', 'finetune_xceptionS2/randominit_latlon/2_layers_tuned', 'finetune_xceptionS2/randominit_latlon/3_layers_tuned', 'unet/randominit/12_channels', 'unet/pretrained/freeze_backbone_True', 'unet/pretrained/freeze_backbone_False'], resolution=10, eval_metric='mae')
    visualize_predictions(model='local_only_models/128_filters/3_channels', resolution=10)
    visualize_predictions(model='local_only_models/128_filters/4_channels', resolution=10)
    visualize_predictions(model='local_only_models/128_filters/12_channels', resolution=10)
    visualize_predictions(model='local_only_models/128_filters/15_channels', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/pretrained/1_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/pretrained/2_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/pretrained/3_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/randominit_nolatlon/1_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/randominit_nolatlon/2_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/randominit_nolatlon/3_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/randominit_latlon/1_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/randominit_latlon/2_layers_tuned', resolution=10)
    visualize_predictions(model='finetune_xceptionS2/randominit_latlon/3_layers_tuned', resolution=10)
    visualize_predictions(model='unet/randominit/12_channels', resolution=10)
    visualize_predictions(model='unet/pretrained/freeze_backbone_True', resolution=10)
    visualize_predictions(model='unet/pretrained/freeze_backbone_False', resolution=10)
