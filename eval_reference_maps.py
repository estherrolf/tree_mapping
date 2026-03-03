# imports
import itertools
import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from eval_utils import compare_aligned_data
from utils import get_project_dir

project_dir = get_project_dir()
reference_maps = ['ETH', 'GLAD', 'PAULS']
eval_metrics = {'r2': 'R\u00b2', 'mae': 'Mean Absolute Error', 'mse': 'Mean Squared Error', 'rmse': 'RMSE', 'me': 'Mean Error'}
os.makedirs('figures', exist_ok=True)

def eval_reference_maps(resolution, eval_metric):
    lidar_coarsened_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    sites = sorted(os.listdir(lidar_coarsened_dir))
    results_by_site = {reference_maps[i]: {} for i in range(len(reference_maps))}
    results_by_site_plot = []
    labels = []
    preds = []
    results = {reference_maps[i]: {} for i in range(len(reference_maps))}
    results_plot = []
    interval_bounds = [0, 3, 6, 10, 30]
    num_intervals = len(interval_bounds) - 1

    # get results
    for reference_map in reference_maps:
        reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'

        for site in sites:
            with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file: # opens label tiff
                site_labels = file.read().ravel()
                labels = np.append(labels, site_labels)

            with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file: # opens prediction tiff
                site_preds = file.read().ravel()
                preds = np.append(preds, site_preds)

            results_by_site[reference_map][site] = compare_aligned_data(site_labels, site_preds)

        results_by_site_plot += [[results_by_site[reference_map][site][eval_metric] for site in sites]]

        for i in range(num_intervals):
            results[reference_map][f'interval_{i}'] = compare_aligned_data(labels, preds, interval=[interval_bounds[i], interval_bounds[i+1]])['errors']

        results_plot += [[results[reference_map][f'interval_{i}'] for i in range(num_intervals)]]

    print(f'Resolution = {resolution}')
    for i in range(len(results_plot)):
        print(reference_maps[i])
        for j in range(len(results_plot[i])):
            print(f'Number of pixels in interval {j} = {len(results_plot[i][j])}')

    aE = [np.mean(list(itertools.chain.from_iterable(reference_map_data))) for reference_map_data in results_plot] # average error

    # plot results by site
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    x = np.arange(len(sites))
    width = 0.3
    multiplier = 0

    for i in range(len(results_by_site_plot)):
        offset = width * multiplier
        ax.bar(x+offset, results_by_site_plot[i], width, label=reference_maps[i])
        print(results_by_site_plot[i])
        multiplier += 1
    
    # ax.plot(np.arange(len(sites)+1 - 7/6*width), (len(sites)+1)*[aE[0]], linestyle='dashed', label=f'{reference_maps[i]} aE')
    # ax.plot(np.arange(len(sites)+1 - 7/6*width), (len(sites)+1)*[aE[1]], linestyle='dashed', label=f'{reference_maps[i]} aE')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(sites, rotation=90)
    ax.set_ylabel(f'{eval_metrics[eval_metric]} (m)')
    ax.set_title('ETH and GLAD Maps Evaluated by Site')
    ax.legend(ncols=2)
    plt.savefig(f'figures/ETH-and-GLAD-maps-evaluated-by-site-{resolution}m.png', bbox_inches='tight', pad_inches=0.1)
    print('Plotted results by site')

    # plot results by interval
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    x = np.arange(num_intervals)
    width = 0.3
    tick_positions = sorted(list(np.arange(num_intervals+1) - 7/6*width) + list(x + width/2))
    boxplots = []
    lines = []

    for i in range(len(results_plot)):
        pos = x + width*i + width/10*((-1)**(i+1))
        boxplots.append(ax.boxplot(results_plot[i], sym='', positions=pos, widths=width, patch_artist=True, boxprops=dict(facecolor=f'C{i}'), medianprops=dict(color='black')))
        lines.append(ax.plot(np.arange(num_intervals+1) - 7/6*width, (num_intervals+1)*[aE[i]], linestyle='dashed', label=f'{reference_maps[i]} aE'))

    ax.set_xticks(tick_positions, labels=['' if i % 2 == 0 else f'{interval_bounds[int(i/2)]}-{interval_bounds[int(i/2)+1]}' for i in range(2*len(interval_bounds)-1)])
    ax.set_xlabel('LiDAR-Derived Height (m)')
    ax.set_ylabel('Error (m)')
    ax.set_title('ETH and GLAD Maps Evaluated by Height Interval')
    ax.legend([boxplots[0]["boxes"][0], boxplots[1]["boxes"][0], lines[0][0], lines[1][0]], [reference_maps[0], reference_maps[1], f'{reference_maps[0]} aE', f'{reference_maps[1]} aE'], ncols=2)
    
    for i in range(len(ax.xaxis.get_major_ticks())):
        if i % 2 != 0:
            ax.xaxis.get_major_ticks()[i].tick1line.set_visible(False)

    plt.savefig(f'figures/ETH-and-GLAD-maps-evaluated-by-interval-{resolution}m.png', bbox_inches='tight', pad_inches=0.1)
    print('Plotted results by height interval')

if __name__ == '__main__':
    eval_reference_maps(resolution=10, eval_metric='rmse')
    eval_reference_maps(resolution=30, eval_metric='rmse')
