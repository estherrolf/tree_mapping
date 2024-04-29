# imports
from torchgeo.trainers.utils import extract_backbone
import numpy as np
import os
import rasterio
import sklearn.metrics
import utils

def get_lowest_val_checkpoint(checkpoint_dir):
    # find the checkpoint in checkpoint_dir with the lowest val loss
    checkpoint_fps_lowest_val = [x for x in os.listdir(checkpoint_dir) if x.startswith('epoch=')]
    assert len(checkpoint_fps_lowest_val) == 1
    checkpoint_fp = os.path.join(checkpoint_dir, checkpoint_fps_lowest_val[0])
    return checkpoint_fp

def init_model_from_checkpoint(model, checkpoint_fp):
    # instantiate model weights with backbone from ckechpoint_fp
    _, state_dict = extract_backbone(checkpoint_fp)
    model.load_state_dict(state_dict)

def get_lowest_val_checkpoint(checkpoint_dir):
    # find the checkpoint in checkpoint_dir with the lowest val loss
    checkpoint_fps_lowest_val = [x for x in os.listdir(checkpoint_dir) if x.startswith('epoch=')]
    assert len(checkpoint_fps_lowest_val) == 1
    checkpoint_fp = os.path.join(checkpoint_dir, checkpoint_fps_lowest_val[0])
    return checkpoint_fp

def init_model_from_checkpoint(model, checkpoint_fp):
    # instantiate model weights with backbone from ckechpoint_fp
    _, state_dict = extract_backbone(checkpoint_fp)
    model.load_state_dict(state_dict)

def get_site_lidar_tif_fn(eval_site_id, data_dir):
    site_dir = f'{data_dir}/lidar/Karingani_merged_crs_10/{eval_site_id}'
    tifs_this_site = [x for x in os.listdir(site_dir) if x.endswith('.tif')]
    
    assert len(tifs_this_site) == 1
    
    return os.path.join(site_dir,tifs_this_site[0])

def match_map_to_labels(eval_site_id, 
                        compare_identifier,
                        data_dir = "/n/home10/erolf/tree_mapping/data",
                        model_output_dir = "/n/home10/erolf/tree_mapping/data/output/model_output",
                        appender=''):
    
    compare_identifier = compare_identifier#.lower()

    pred_map_fn = os.path.join(model_output_dir, f"{compare_identifier}/{eval_site_id}_{compare_identifier}{appender}.tif")

    # else print(f"compare identifier {compare_identifier} not understood")

    matched_output_dir = model_output_dir.replace('model_output', 'matched_model_output')
    out_dir = matched_output_dir + f'/{compare_identifier}_map'
    out_fn = os.path.join(out_dir, f'{eval_site_id}_{compare_identifier}_map{appender}.tif')
    
    for d in [matched_output_dir, out_dir]:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

    target_fn = get_site_lidar_tif_fn(eval_site_id, data_dir)
    
    utils.match_input_to_target_tif(input_fn=pred_map_fn, 
                                    output_fn=out_fn, 
                                    target_fn=target_fn,
                                    verbose=False)
                     
    return out_fn                

def compare_aligned_data(labels, 
                         preds, 
                         interval=[0, 30],
                         nodata_value=-9999.0,
                         reference_map=False,
                         return_vals=False):

    mask = (labels != nodata_value) & (preds != nodata_value) & (labels >= interval[0]) & (labels <= interval[1]) # excludes pixels for which we have no labels or predictions or whose labels are out of range in the current analysis
    masked_labels = labels[mask]
    masked_preds = preds[mask]

    if not reference_map:
        masked_preds[masked_preds < 0] = 0 # set negative predicted values to 0
        masked_preds[masked_preds > 30] = 30 # set predicted values over 30 to 30

    r2 = sklearn.metrics.r2_score(masked_labels, masked_preds) # r^2 score
    mae = sklearn.metrics.mean_absolute_error(masked_labels, masked_preds) # mean absolute error
    mse = sklearn.metrics.mean_squared_error(masked_labels, masked_preds) # mean squared error
    me = np.mean(masked_preds - masked_labels) # mean error
    errors = masked_preds - masked_labels # residuals
    perc_10 = np.percentile(errors, 10)
    q1 = np.percentile(errors, 25)
    median = np.median(errors)
    q3 = np.percentile(errors, 75)
    perc_90 = np.percentile(errors, 90)

    if return_vals:
        return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse), 'me': me, 'errors': errors, 'data_mask': mask, 'labels': labels, 'preds': preds}
    else:
        return {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': np.sqrt(mse), 'me': me, 'errors': errors, 'perc_10': perc_10, 'q1': q1, 'median': median, 'q3': q3, 'perc_90': perc_90}

def plot_aligned_data(labels, preds, vis=None, title='title me!'):
    
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
                     context=False,
                     plot=True, 
                     return_data=False):
        
    # save predictions
    cropped_map_fn_this = match_map_to_labels(eval_site_id = eval_site_id, 
                                              compare_identifier = compare_identifier,
                                              data_dir=data_dir, 
                                              model_output_dir=model_output_dir) 
    # also save context map if applicable
    if context:
        context_map_fn_this = match_map_to_labels(eval_site_id = eval_site_id, 
                                              compare_identifier = compare_identifier,
                                              data_dir=data_dir, 
                                              model_output_dir=model_output_dir,
                                              appender='_context')    

    print(cropped_map_fn_this)
    with rasterio.open(cropped_map_fn_this) as f:
        cropped_map = f.read()
        nodata_val = f.nodata
        
    if context:
        with rasterio.open(context_map_fn_this) as f:
            context_map = f.read()
        
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

    
    if plot:
        plot_aligned_data(labels, cropped_map, vis=None,
                          title=f'site {eval_site_id}')

    if return_data:
        if context:
            return compare_aligned_data(labels, cropped_map), labels, cropped_map, context_map
        else: 
            return compare_aligned_data(labels, cropped_map), labels, cropped_map
    
    else: 
        return compare_aligned_data(labels, cropped_map)

def baseline_predict_mean_per_cite(eval_site_id, data_dir ='data'):
    labels_fn = get_site_lidar_tif_fn(eval_site_id,data_dir)
    with rasterio.open(labels_fn) as f:
        labels = f.read()
    mask = labels >= 0 
    return compare_aligned_data(labels, np.ones_like(labels) * labels[mask].mean())