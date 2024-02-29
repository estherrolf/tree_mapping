from eval_utils import compare_aligned_data
from utils import get_project_dir
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import sklearn.metrics

data_dir = f'{get_project_dir()}/data'
reference_maps = ['ETH', 'GLAD']

def eval_reference_maps_feature(feature, resolution):
    lidar_coarsened_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_{resolution}m'
    sites = sorted(os.listdir(lidar_coarsened_dir))
    results_feature = {reference_maps[0]: {}, reference_maps[1]: {}}
    results_plot = []
    interval_bounds = [0, 100, 300, 600, 1000, 2000, 3000] # 2854 m is the max
    num_intervals = len(interval_bounds) - 1

    for reference_map in reference_maps:
        reference_map_by_site_dir = f'{data_dir}/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'
        
        for i in range(num_intervals):
            labels = []
            preds = []

            for site in sites:
                with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                    site_labels = file.read().ravel()

                with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file:
                    site_preds = file.read().ravel()
                
                site_feature_distances = np.load(f'{data_dir}/features/{feature}/distances_to_{feature}_{resolution}m/{site}_distances_to_{feature}_{resolution}m.npy').ravel()
                mask = (site_feature_distances >= interval_bounds[i]) & (site_feature_distances <= interval_bounds[i+1])
                assert len(site_labels) == len(mask)
                labels = np.append(labels, site_labels[mask])
                preds = np.append(preds, site_preds[mask])
            
            results_feature[reference_map][f'interval_{i}'] = compare_aligned_data(labels, preds)['errors']

        results_plot += [[results_feature[reference_map][f'interval_{i}'] for i in range(num_intervals)]]

    # plot results by interval
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    x = np.arange(num_intervals)
    width = 0.3
    tick_positions = sorted(list(np.arange(num_intervals+1) - 7/6*width) + list(x + width/2))
    boxplots = []
    lines = []
    aE = [np.mean(list(itertools.chain.from_iterable(reference_map_data))) for reference_map_data in results_plot]

    for i in range(len(results_plot)):
        pos = x + width*i + width/10*((-1)**(i+1))
        boxplots.append(ax.boxplot(results_plot[i], sym='', positions=pos, widths=width, patch_artist=True, boxprops=dict(facecolor=f'C{i}'), medianprops=dict(color='black')))
        lines.append(ax.plot(np.arange(num_intervals+1) - 7/6*width, (num_intervals+1)*[aE[i]], linestyle='dashed', label=f'{reference_maps[i]} aE'))
    
    ax.set_xticks(tick_positions, labels=['' if i % 2 == 0 else f'{interval_bounds[int(i/2)]}-{interval_bounds[int(i/2)+1]}' for i in range(2*len(interval_bounds)-1)])
    ax.set_xlabel(f'Distance to {feature.capitalize()} (m)')
    ax.set_ylabel('Error (m)')
    ax.set_title(f'ETH and GLAD Maps Evaluated by Distance to {feature.capitalize()}')
    ax.legend([boxplots[0]['boxes'][0], boxplots[1]['boxes'][0], lines[0][0], lines[1][0]], [reference_maps[0], reference_maps[1], f'{reference_maps[0]} aE', f'{reference_maps[1]} aE'], ncols=2)
    
    for i in range(len(ax.xaxis.get_major_ticks())):
        if i % 2 != 0:
            ax.xaxis.get_major_ticks()[i].tick1line.set_visible(False)
    
    plt.savefig(f'figures/ETH-and-GLAD-maps-evaluated-by-{feature}-distance-{resolution}m.png', bbox_inches='tight', pad_inches=0.1)
    print(f'Plotted results by {feature} distance')

if __name__ == '__main__':
    eval_reference_maps_feature(feature='river', resolution=10)
    eval_reference_maps_feature(feature='river', resolution=30)
