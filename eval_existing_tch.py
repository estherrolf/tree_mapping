import numpy as np
import os
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
from eval_utils import compare_aligned_data

data_dir = 'data'
# data_dir = '../../../tambe_lab/Users/luciagordon/tree_mapping_lucia_branch/data'   
label_dir = os.path.join(data_dir, 'int/lidar/lidar_by_site_32736_10m')
site_names = sorted(os.listdir(label_dir))
REFERENCES = ['ETH', 'GLAD']
results_plot = []
res_by_site_name = {REFERENCES[0]: {}, REFERENCES[1]: {}}

for reference in REFERENCES:
    eval_dir = f'{data_dir}/existing_reference_data/{reference.lower()}_maps_per_site_10m'
    fns_eval = os.listdir(eval_dir)

    for site_name in site_names:
        tif_fns = os.listdir(os.path.join(label_dir, site_name))
        assert len(tif_fns) == 1
        tif_fn = tif_fns[0]
        label_fp = os.path.join(label_dir, site_name, tif_fn)
        
        with rasterio.open(label_fp) as f:
            labels = f.read()
            
        this_pred_fns = [x for x in fns_eval if site_name in x]
        assert len(this_pred_fns) == 1
        this_pred_fn = this_pred_fns[0]
        with rasterio.open(os.path.join(eval_dir,this_pred_fn)) as f:
            preds = f.read()
            
        res_by_site_name[reference][site_name] = compare_aligned_data(labels, preds, return_vals=True)
        eval_metric = 'mae'
    
    results_plot += [[res_by_site_name[reference][x][eval_metric] for x in site_names]]

site_name = site_names[4]
nodata_val = -9999.
# scatter plot one for sanity check
labels = res_by_site_name['ETH'][site_name]['labels'][0]
preds = res_by_site_name['ETH'][site_name]['preds'][0]
# mask out NAN labels
mask = res_by_site_name['ETH'][site_name]['mask'][0]
preds_ = preds[mask]
labels_ = labels[mask]
preds_valid = preds_[preds_ != nodata_val].ravel()
labels_valid = labels_[preds_ != nodata_val].ravel()
plt.scatter(preds_valid, labels_valid)
plt.savefig('scatter.png')

print(sklearn.metrics.mean_absolute_error(labels_valid, preds_valid))
print(results_plot)

fig, ax = plt.subplots(layout = 'constrained', dpi = 300)
x = np.arange(len(site_names))
width = 0.3
multiplier = 0

for i in range(len(results_plot)):
    offset = width * multiplier
    rectangles = ax.bar(x + offset, results_plot[i], width, label = REFERENCES[i])
    multiplier += 1

ax.set_xticks(x + width/2, site_names)
ax.set_xticklabels(site_names, rotation = 90)
ax.set_ylabel('Mean Average Error')
ax.set_title('ETH and GLAD Maps Evaluated per Site')
ax.legend(ncols = 2)
plt.savefig('ETH-and-GLAD-maps-evaluated.png', bbox_inches = 'tight', pad_inches = 0.1)

vis_site_ids = ['KaringaniSite12', 'KaringaniSite16']

def plot_aligned_data(labels, preds, vis, title, mask_nodata=False, nodata_val=-9999.):
    if mask_nodata: 
        preds[preds == nodata_val] = None
        labels[labels == nodata_val] = None
    fig, ax = plt.subplots(1,3, figsize=(24,8))
    ax[0].imshow(labels, vmin=0, vmax=15, interpolation=None, cmap='Greens')
    ax[1].imshow(preds, vmin=0, vmax=15, interpolation=None, cmap='Greens')
    ax[2].imshow(vis,  interpolation=None)
    
    ax[0].set_title('CHM labels (10m)')
    ax[1].set_title('reference/predicted values')
    ax[2].set_title('S2 image')
    for axis in ax:
        axis.axis('off')
    print(preds.min())
    plt.savefig(title)

for reference in REFERENCES:
    for eval_site_id in vis_site_ids:
        # load tif
        labels = res_by_site_name[reference][eval_site_id]['labels'][0]
        preds = res_by_site_name[reference][eval_site_id]['preds'][0]
        
        eval_input_dir = os.path.join(data_dir,f'int/sentinel/sentinel_by_site_32736_10m/{eval_site_id}')
        example_tif = [x for x in os.listdir(eval_input_dir) if x.endswith("TCI_10m.tif")][0]
        with rasterio.open(os.path.join(eval_input_dir, example_tif)) as f:
            vis = f.read().transpose(1,2,0)
        
        plot_aligned_data(labels, preds, vis = vis, title=f'{reference} map {eval_site_id}', mask_nodata=True)
