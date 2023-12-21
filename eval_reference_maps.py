# imports
import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from eval_utils import compare_aligned_data
from utils import get_project_dir

project_dir = get_project_dir()
lidar_10m_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_10m'
sites = sorted(os.listdir(lidar_10m_dir))
reference_maps = ['ETH', 'GLAD']
results_plot = []
results_by_site = {reference_maps[0]: {}, reference_maps[1]: {}}

for reference_map in reference_maps:
    reference_map_by_site_dir = f'{project_dir}/data/existing_reference_data/{reference_map.lower()}_maps_per_site_10m'

    for site in sites:
        with rasterio.open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif') as file:
            labels = file.read()
            
        with rasterio.open(f'{reference_map_by_site_dir}/{reference_map}_MAP_{site}_10m.tif') as file:
            preds = file.read()
            
        results_by_site[reference_map][site] = compare_aligned_data(labels, preds, return_vals=True)
        eval_metric = 'mae'
    
    results_plot += [[results_by_site[reference_map][site][eval_metric] for site in sites]]

site = sites[4]
nodata_val = -9999.
# scatter plot one for sanity check
labels = results_by_site['ETH'][site]['labels'][0]
preds = results_by_site['ETH'][site]['preds'][0]
# mask out NAN labels
mask = results_by_site['ETH'][site]['mask'][0]
preds_ = preds[mask]
labels_ = labels[mask]
preds_valid = preds_[preds_ != nodata_val].ravel()
labels_valid = labels_[preds_ != nodata_val].ravel()
plt.scatter(preds_valid, labels_valid)
plt.savefig(f'{project_dir}/scatter.png')

print(sklearn.metrics.mean_absolute_error(labels_valid, preds_valid))
print(results_plot)

fig, ax = plt.subplots(layout='constrained', dpi=300)
x = np.arange(len(sites))
width = 0.3
multiplier = 0

for i in range(len(results_plot)):
    offset = width * multiplier
    rectangles = ax.bar(x+offset, results_plot[i], width, label=reference_maps[i])
    multiplier += 1

ax.set_xticks(x + width/2, sites)
ax.set_xticklabels(sites, rotation=90)
ax.set_ylabel('Mean Average Error (m)')
ax.set_title('ETH and GLAD Maps Evaluated per Site')
ax.legend(ncols=2)
plt.savefig(f'{project_dir}/ETH-and-GLAD-maps-evaluated.png', bbox_inches='tight', pad_inches=0.1)

vis_site_ids = ['KaringaniSite12', 'KaringaniSite16']

def plot_aligned_data(labels, preds, vis, title, mask_nodata=False, nodata_val=-9999.):
    if mask_nodata: 
        preds[preds == nodata_val] = None
        labels[labels == nodata_val] = None

    fig, ax = plt.subplots(1,3, figsize=(24,8))
    ax[0].imshow(labels, vmin=0, vmax=15, interpolation=None, cmap='Greens')
    ax[1].imshow(preds, vmin=0, vmax=15, interpolation=None, cmap='Greens')
    ax[2].imshow(vis, interpolation=None)
    
    ax[0].set_title('CHM labels (10m)')
    ax[1].set_title('reference/predicted values')
    ax[2].set_title('S2 image')

    for axis in ax:
        axis.axis('off')

    print(preds.min())
    plt.savefig(f'{project_dir}/{title}')

for reference_map in reference_maps:
    for eval_site_id in vis_site_ids:
        # load tif
        labels = results_by_site[reference_map][eval_site_id]['labels'][0]
        preds = results_by_site[reference_map][eval_site_id]['preds'][0]
        
        eval_input_dir = f'{project_dir}/data/int/sentinel/sentinel_by_site_32736_10m/{eval_site_id}'
        example_tif = [x for x in os.listdir(eval_input_dir) if x.endswith("TCI_10m.tif")][0]
        with rasterio.open(f'{eval_input_dir}/{example_tif}') as f:
            vis = f.read().transpose(1,2,0)
        
        plot_aligned_data(labels, preds, vis=vis, title=f'{reference_map} map {eval_site_id}', mask_nodata=True)
