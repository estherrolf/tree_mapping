import numpy as np
import os
import rasterio
import sklearn.metrics
import utils

def get_site_lidar_tif_fn(eval_site_id,data_dir):
    site_dir = data_dir + f'/lidar/Karingani_merged_crs_10/{eval_site_id}'
    tifs_this_site = [x for x in os.listdir(site_dir) if x.endswith('.tif')]
    
    assert len(tifs_this_site) == 1
    
    return os.path.join(site_dir,tifs_this_site[0])

def match_map_to_labels(eval_site_id, 
                        compare_identifier='eth',
                        data_dir = "/n/home10/erolf/tree_mapping/data",
                        model_output_dir = "/n/home10/erolf/tree_mapping/data/output/model_output"):
    
    compare_identifier = compare_identifier#.lower()

    # eth
    if compare_identifier == 'eth':
        pred_map_fn = data_dir + '/raw/global_tch_maps/ETH_GlobalCanopyHeight_10m_2020_S27E030_Map.tif'
    else:
        pred_map_fn = os.path.join(model_output_dir, f"{compare_identifier}/{eval_site_id}_{compare_identifier}.tif")

    # else print(f"compare identifier {compare_identifier} not understood")

    out_dir = model_output_dir.replace('model_output', 'matched_model_output') + f'/{compare_identifier}_map'
    out_fn = os.path.join(out_dir, f'{eval_site_id}_{compare_identifier}_map.tif')
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    
    target_fn = get_site_lidar_tif_fn(eval_site_id, data_dir)
    
    utils.match_input_to_target_tif(input_fn = pred_map_fn, 
                                    output_fn = out_fn, 
                                    target_fn = target_fn,
                                    verbose=False)
                     
    return out_fn                
    
    
def compare_aligned_data(labels, preds, nodata_value = -9999):
    
    mask = labels != nodata_value
    labels_ = labels[mask].ravel()
    preds_ = preds[mask].ravel()
    
    r2 = sklearn.metrics.r2_score(labels_,preds_)
    mae = sklearn.metrics.mean_absolute_error(labels_,preds_)
    mse = sklearn.metrics.mean_squared_error(labels_,preds_)

    return {'r2':r2, 'mae':mae, 'mse':mse}

def plot_aligned_data(labels, preds, vis=None,
                      title='title me!'):
    
    if vis is None:
        fig, ax = plt.subplots(1,2, figsize=(12*4,6*4))
    else:
        fig, ax = plt.subplots(1,3, figsize=(18*4,6*4))
    
    ymax = labels.max()
    ax[0].imshow(labels[0], vmin=0, vmax=ymax, cmap='Greens')
    ax[0].set_title('CHM data', fontsize=48)
    
    preds_show = preds.copy()
    mask = labels > 0
    preds_show[~mask] = 0
    ax[1].imshow(preds_show[0], vmin=0, vmax=ymax, cmap='Greens')
    ax[1].set_title('reference data / predictions', fontsize=48)
    
    if vis is not None:
        ax[2].imshow(vis.transpose(1,2,0).astype(int))
        ax[2].set_title('visual image', fontsize=48)
    plt.suptitle(title, fontsize=48 )
    
def eval_tif_at_cite(eval_site_id, compare_identifier, 
                     data_dir ="/n/home10/erolf/tree_mapping/data", 
                     model_output_dir="/n/home10/erolf/tree_mapping/data/output/model_output",
                     plot=True, return_data=False):
    
    cropped_map_fn_this = match_map_to_labels(eval_site_id = eval_site_id, 
                                              compare_identifier = compare_identifier,
                                              data_dir=data_dir, 
                                              model_output_dir=model_output_dir)

    print(cropped_map_fn_this)
    with rasterio.open(cropped_map_fn_this) as f:
        cropped_map = f.read()
        nodata_val = f.nodata
        
    if (cropped_map == nodata_val).all():
        print('reference data all nans - did you check that this matches the extent of the labels?')
        return {'r2':np.nan, 'mae':np.nan, 'mse':np.nan}
    elif (cropped_map == nodata_val).any():
        print('reference data contains nans - inputing as 0')
        cropped_map[cropped_map == nodata_val] = 0
        #return {'r2':np.nan, 'mae':np.nan, 'mse':np.nan}
    
    labels_fn = get_site_lidar_tif_fn(eval_site_id,data_dir)

    with rasterio.open(labels_fn) as f:
        labels = f.read()

    assert labels.shape == cropped_map.shape
    
    if plot:
        plot_aligned_data(labels, cropped_map, vis=None,
                          title=f'site {eval_site_id}')

    if return_data:
        return compare_aligned_data(labels, cropped_map), labels, cropped_map
    
    else: 
        return compare_aligned_data(labels, cropped_map)

def baseline_predict_mean_per_cite(eval_site_id, data_dir ='data'):
    labels_fn = get_site_lidar_tif_fn(eval_site_id,data_dir)
    with rasterio.open(labels_fn) as f:
        labels = f.read()
    mask = labels >= 0 
    return compare_aligned_data(labels, np.ones_like(labels) * labels[mask].mean())