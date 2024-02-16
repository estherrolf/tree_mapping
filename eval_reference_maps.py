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
reference_maps = ['ETH', 'GLAD']

def eval_reference_maps(resolution=30, eval_metric='me'):
    lidar_coarsened_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    sites = sorted(os.listdir(lidar_coarsened_dir))
    results_by_site = {reference_maps[0]: {}, reference_maps[1]: {}}
    results_by_site_plot = []
    labels = []
    preds = []
    results = {reference_maps[0]: {}, reference_maps[1]: {}}
    results_plot = []
    interval_bounds = [0, 10, 20, 30]
    num_intervals = len(interval_bounds) - 1

    # get results
    for reference_map in reference_maps:
        reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_{resolution}m'

        for site in sites:
            with rasterio.open(f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif') as file:
                site_labels = file.read().ravel()
                labels = np.append(labels, site_labels)

            with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_{resolution}m.tif') as file:
                site_preds = file.read().ravel()
                preds = np.append(preds, site_preds)

            results_by_site[reference_map][site] = compare_aligned_data(site_labels, site_preds, return_vals=True)

        results_by_site_plot += [[results_by_site[reference_map][site][eval_metric] for site in sites]]

        for i in range(num_intervals):
        #     results[reference_map][f'interval_{i}'] = compare_aligned_data(labels, preds, interval=[interval_bounds[i], interval_bounds[i+1]], return_vals=True)

        # results_plot += [[results[reference_map][f'interval_{i}'][eval_metric] for i in range(num_intervals)]]
            results[reference_map][f'interval_{i}'] = compare_aligned_data(labels, preds, interval=[interval_bounds[i], interval_bounds[i+1]], return_vals=True)['errors']

        results_plot += [[results[reference_map][f'interval_{i}'] for i in range(num_intervals)]]

    for i in range(len(results_plot)):
        print(len(results_plot[i]))
        for j in range(len(results_plot[i])):
            print(len(results_plot[i][j]))

    # print(f'results plot shape = {np.array(results_plot).shape}')

    # plot results by site
    fig, ax = plt.subplots(layout='constrained', dpi=300)
    x = np.arange(len(sites))
    width = 0.3
    multiplier = 0

    for i in range(len(results_by_site_plot)):
        offset = width * multiplier
        rectangles = ax.bar(x+offset, results_by_site_plot[i], width, label=reference_maps[i])
        print(results_by_site_plot[i])
        multiplier += 1

    ax.set_xticks(x + width/2)
    ax.set_xticklabels(sites, rotation=90)
    ax.set_ylabel('Mean Error (m)')
    ax.set_title('ETH and GLAD Maps Evaluated by Site')
    ax.legend(ncols=2)
    # plt.savefig(f'{project_dir}/ETH-and-GLAD-maps-evaluated-by-site.png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'ETH-and-GLAD-maps-evaluated-by-site-{resolution}.png', bbox_inches='tight', pad_inches=0.1)

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
    ax.set_xlabel('LiDAR-Derived Height (m)')
    ax.set_ylabel('Error (m)')
    ax.set_title('ETH and GLAD Maps Evaluated by Height Interval')
    ax.legend([boxplots[0]["boxes"][0], boxplots[1]["boxes"][0], lines[0][0], lines[1][0]], [reference_maps[0], reference_maps[1], f'{reference_maps[0]} aE', f'{reference_maps[1]} aE'], loc='upper right')
    # plt.savefig(f'{project_dir}/ETH-and-GLAD-maps-evaluated-by-interval.png', bbox_inches='tight', pad_inches=0.1)
    plt.savefig(f'ETH-and-GLAD-maps-evaluated-by-interval.png', bbox_inches='tight', pad_inches=0.1)

eval_reference_maps(resolution=10)
eval_reference_maps(resolution=30)

# # scatter plot one for sanity check
# site = sites[4]
# nodata_val = -9999.
# labels = results_by_site['ETH'][site]['labels'][0]
# preds = results_by_site['ETH'][site]['preds'][0]
# # mask out NAN labels
# mask = results_by_site['ETH'][site]['mask'][0]
# preds_ = preds[mask]
# labels_ = labels[mask]
# preds_valid = preds_[preds_ != nodata_val].ravel()
# labels_valid = labels_[preds_ != nodata_val].ravel()
# plt.scatter(preds_valid, labels_valid)
# plt.savefig(f'{project_dir}/scatter.png')

# print(sklearn.metrics.mean_absolute_error(labels_valid, preds_valid))
# print(results_by_site_plot)

# # plot aligned data
# vis_site_ids = ['KaringaniSite12', 'KaringaniSite16']

# def plot_aligned_data(labels, preds, vis, title, mask_nodata=False, nodata_val=-9999.):
#     if mask_nodata: 
#         preds[preds == nodata_val] = None
#         labels[labels == nodata_val] = None

#     fig, ax = plt.subplots(1,3, figsize=(24,8))
#     ax[0].imshow(labels, vmin=0, vmax=15, interpolation=None, cmap='Greens')
#     ax[1].imshow(preds, vmin=0, vmax=15, interpolation=None, cmap='Greens')
#     ax[2].imshow(vis, interpolation=None)
    
#     ax[0].set_title('CHM labels (10m)')
#     ax[1].set_title('reference/predicted values')
#     ax[2].set_title('S2 image')

#     for axis in ax:
#         axis.axis('off')

#     print(preds.min())
#     plt.savefig(f'{project_dir}/{title}')

# for reference_map in reference_maps:
#     for eval_site_id in vis_site_ids:
#         # load tif
#         labels = results_by_site[reference_map][eval_site_id]['labels'][0]
#         preds = results_by_site[reference_map][eval_site_id]['preds'][0]
        
#         eval_input_dir = f'{project_dir}/data/int/sentinel/sentinel_by_site_32736_10m/{eval_site_id}'
#         example_tif = [x for x in os.listdir(eval_input_dir) if x.endswith("TCI_10m.tif")][0]
#         with rasterio.open(f'{eval_input_dir}/{example_tif}') as f:
#             vis = f.read().transpose(1,2,0)
        
#         plot_aligned_data(labels, preds, vis=vis, title=f'{reference_map} map {eval_site_id}', mask_nodata=True)
