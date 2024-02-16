from eval_utils import compare_aligned_data
from utils import get_project_dir
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import sklearn.metrics

project_dir = get_project_dir()
lidar_10m_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_10m'
sites = sorted(os.listdir(lidar_10m_dir))
reference_maps = ['ETH', 'GLAD']
eval_metric = 'errors'
results_river = {reference_maps[0]: {}, reference_maps[1]: {}}
results_plot = []
interval_bounds = [0, 500, 1000, 1500, 2000, 2500, 3000] # 2854 m is the max
num_intervals = len(interval_bounds) - 1

for reference_map in reference_maps:
    reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_10m'
    
    for i in range(num_intervals):
        labels = []
        preds = []

        for site in sites:
            with rasterio.open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif') as file:
                site_labels = file.read().ravel()

            with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_10m.tif') as file:
                site_preds = file.read().ravel()
            
            site_river_distances = np.load(f'{project_dir}/distances-to-river/{site}_distances_to_river.npy').ravel()
            mask = (site_river_distances >= interval_bounds[i]) & (site_river_distances <= interval_bounds[i+1])
            assert len(site_labels) == len(mask)
            labels = np.append(labels, site_labels[mask])
            preds = np.append(preds, site_preds[mask])
        
        results_river[reference_map][f'interval_{i}'] = list(compare_aligned_data(labels, preds, return_vals=True)[eval_metric])

    results_plot += [[results_river[reference_map][f'interval_{i}'] for i in range(num_intervals)]]

# plot results by interval
fig, ax = plt.subplots(layout='constrained', dpi=300)
x = np.arange(num_intervals)
width = 0.3
boxplots = []
lines = []
aE = [np.mean(list(itertools.chain.from_iterable(reference_map_data))) for reference_map_data in results_plot]

for i in range(len(results_plot)):
    pos = x + width*i + width/10*((-1)**(i+1))
    boxplots.append(ax.boxplot(results_plot[i], sym='', positions=pos, patch_artist=True, boxprops=dict(facecolor=f'C{i}'), medianprops=dict(color='black')))
    lines.append(ax.plot(np.arange(num_intervals+1) - 7/6*width, (num_intervals+1)*[aE[i]], linestyle='dashed', label=f'{reference_maps[i]} aE'))

ax.set_xticks(np.arange(num_intervals+1) - 7/6*width, labels=interval_bounds)
ax.set_xlabel('Distance to River (m)')
ax.set_ylabel('Error (m)')
ax.set_title('ETH and GLAD Maps Evaluated by Distance to River')
ax.legend([boxplots[0]['boxes'][0], boxplots[1]['boxes'][0], lines[0][0], lines[1][0]], [reference_maps[0], reference_maps[1], f'{reference_maps[0]} aE', f'{reference_maps[1]} aE'], loc='upper right')
# plt.savefig(f'{project_dir}/ETH-and-GLAD-maps-evaluated-by-interval.png', bbox_inches='tight', pad_inches=0.1)
plt.savefig(f'ETH-and-GLAD-maps-evaluated-by-river-distance.png', bbox_inches='tight', pad_inches=0.1)





# # plot results by interval
# fig, ax = plt.subplots(layout='constrained', dpi=300)
# x = np.arange(num_intervals)
# width = 0.3
# multiplier = 0

# for i in range(len(results_plot)):
#     offset = width * multiplier
#     ax.bar(x+offset, results_plot[i], width, label=reference_maps[i])
#     multiplier += 1
#     ax.plot(np.arange(num_intervals+1) - 7/6*width, (num_intervals+1)*[np.mean(results_plot[i])], linestyle='dashed', label=f'{reference_maps[i]} aME')

# ax.set_xticks(np.arange(num_intervals+1) - 7/6*width, labels=interval_bounds)
# ax.set_xlabel('Distance to River (m)')
# ax.set_ylabel('Mean Error (m)')
# ax.set_title('ETH and GLAD Maps Evaluated by Distance to River')
# ax.legend(ncols=2)
# # plt.savefig(f'{project_dir}/ETH-and-GLAD-maps-evaluated-by-interval.png', bbox_inches='tight', pad_inches=0.1)
# plt.savefig(f'ETH-and-GLAD-maps-evaluated-by-river-distance.png', bbox_inches='tight', pad_inches=0.1)
