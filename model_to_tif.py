# import argparse
import numpy as np
import os
import rasterio
import sys
import torch
import tqdm 
import yaml
from datamodules.chm_datamodule import ChmDataModule, make_site_dataset, transforms_4_channel_rgbnir_no_mask_imagestats
from eval_utils import get_lowest_val_checkpoint, init_model_from_checkpoint
from experiment_utils import read_config_file, get_site_splits
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, Sentinel2, stack_samples, AsterGDEM, IntersectionDataset
from torchgeo.samplers import GridGeoSampler
from trainers.regression_with_nans import PixelwiseRegressionTask
from utils import get_project_dir

DATA_DIR = get_project_dir()
sentinel_layer_codes = {'b': 'B02',
                        'g': 'B03',
                        'r': 'B04',
                        'nir':'B08',
                        'vis':'TCI'}

def load_ckpt_weight_to_model(task_conditions_dict, checkpoint_dir):
    lowest_val_ckpt_fp = get_lowest_val_checkpoint(checkpoint_dir)
    model = PixelwiseRegressionTask(**task_conditions_dict).model

    init_model_from_checkpoint(model, lowest_val_ckpt_fp)

    return model, lowest_val_ckpt_fp

# predict and save as tiff for a given site
def predict_site(site_id, 
                 checkpoint_dir,
                 task_conditions_dict,
                 img_layers,
                 output_fp,
                 transforms_pred_model,
                 pred_args = {},
                 nodata_value = -9999.,
                 img_nodata_value = -9999.,
                 nodata_pad= 5,
                 device='cuda',
                 data_dir='data'):
    '''
    nodata_pad will pad any predictions within that many pixels of a nodata input image as nodata output.
    '''
    
    # load the model for evaluation
    model, lowest_val_ckpt_fp = load_ckpt_weight_to_model(task_conditions_dict, checkpoint_dir)
    model = model.to(device).eval() 

    return predict_site_with_model(site_id,
                                   model, 
                                   output_fp,
                                   img_layers,
                                   pred_args = pred_args, 
                                   transforms = transforms_pred_model,
                                   nodata_value = nodata_value,
                                   img_nodata_value = img_nodata_value,
                                   nodata_pad = nodata_pad,
                                   device = device,
                                   data_dir = data_dir)
    
def predict_site_with_model(site_id, 
                            model,
                            output_fp,
                            img_layers,
                            transforms,
                            pred_args = {},
                            nodata_value = -9999.,
                            img_nodata_value = -9999.,
                            nodata_pad= 5,
                            device='cuda',
                            data_dir='data'):

    model = model.to(device).eval()                              
    pad = pred_args['padding']
    
    dataset = make_site_dataset(site_id, 
                                layers = img_layers, # no CHM
                                transforms = transforms)
    sampler = GridGeoSampler(dataset, pred_args['patch_size'], stride = pred_args['stride'])
    dataloader = DataLoader(dataset,
                            sampler = sampler,
                            batch_size = pred_args['batch_size'],
                            num_workers = pred_args['num_workers'],
                            collate_fn = stack_samples)
    
    # run inference
    # make an array of the right shape
    eval_input_dir = f'{DATA_DIR}/int/sentinel/sentinel_by_site_32736_10m/{site_id}'
    example_tif = [x for x in os.listdir(eval_input_dir) if x.endswith('.tif')][0]
    
    with rasterio.open(os.path.join(eval_input_dir, example_tif)) as f:
            input_height, input_width = f.shape
            profile = f.profile
            transform = profile['transform']
    
    output = np.ones((input_height, input_width), dtype = np.float32) * nodata_value
    dl_enumerator = tqdm.tqdm(dataloader)

    # modification of code from Caleb
    for batch in dl_enumerator:
            images = batch['image'].to(device)
            bboxes = batch['bbox']

            with torch.inference_mode():
                predictions = model(images)
                predictions = predictions[:,0].cpu().numpy()

                #anything within pad pixels of a nodata img pixels gets a nodata label
                img_nodata_ = (images == img_nodata_value).any(dim=1).cpu().numpy().astype(float)
                padder = torch.nn.modules.Conv2d(in_channels=1, out_channels=1, kernel_size=nodata_pad*2+1, stride=1, padding=0)
                
                padder.weight = torch.nn.Parameter(torch.ones_like(padder.weight), requires_grad=False)
            
                img_nodata_padded = padder(torch.Tensor(img_nodata_)).cpu().numpy()
                predictions[:,nodata_pad:-nodata_pad, nodata_pad:-nodata_pad][img_nodata_padded > 1 ] = nodata_value

            for i in range(len(bboxes)):
                bb = bboxes[i]

                left, top = ~transform * (bb.minx, bb.maxy)
                right, bottom = ~transform * (bb.maxx, bb.miny)
                left, right, top, bottom = int(np.round(left)), int(np.round(right)), int(np.round(top)), int(np.round(bottom))

                patch_size = pred_args['patch_size']
                tile_size = patch_size - 2*pad
                assert right - left == patch_size
                assert bottom - top == patch_size

                x1, y1 = top + pad, left + pad
            
                # fix if at ends
                pred_tile = predictions[i]

                if x1 < 0:
                    width = patch_size+x1
                    pred_tile = pred_tile[-(width):,:]

                if y1 + tile_size > output.shape[1]:                
                    width = output.shape[1] - y1 + 2 * pad
                    pred_tile = pred_tile[:,:width]

                x2 = x1 + tile_size
                y2 = y1 + tile_size
                if x1 < 0: x1 = 0
                
                output[x1:x2,y1:y2] = pred_tile[pad:-pad, pad:-pad]
                    
    #save the output
    if output_fp is not None:
        profile_out = profile.copy()
        profile_out.update(dtype = 'float32', count = '1')

        print('saving predictions in ', output_fp)
        with rasterio.open(output_fp, 'w', **profile_out) as dst:
            dst.write(output, 1)
            
    return output

def make_model_output_dirs(exp_name, seed, split_number):
    out_dir = 'predicted_maps'
    out_dir_exp = f'{out_dir}/{exp_name}'
    out_dir_split = f'{out_dir}/{exp_name}/seed_{split_seed}_split_{split_number}'
    
    for path_name in [out_dir, out_dir_exp, out_dir_split]:
        if not os.path.exists(path_name): os.mkdir(path_name)
        
    return out_dir_split

def run_experiment_models_through_one_split(test_sites, 
                                            checkpoint_dir,
                                            out_dir_split,
                                            version_id, 
                                            experiment_cfg,
                                            device = 'cuda'):
    
    num_image_channels = cfg['data']['num_image_channels']

    if num_image_channels == 4:
        img_layers = ['r','g','b','nir','vis']
        transforms_pred_model=transforms_4_channel_rgbnir_no_mask_imagestats
    else:
        print('no directive for {num_image_channels} image channels')
    
    pred_args = {'patch_size': 64, 
                 'padding': 5,
                 'batch_size': 1,
                 'num_workers': 1,
                 'stride': 54}

    task_conditions_dict = experiment_cfg['task']
    nodata_pad = task_conditions_dict['pad_pixels']
    
    for site_id in test_sites:
        output_dir = f'{out_dir_split}/{site_id}'
        if not os.path.exists(output_dir): os.mkdir(output_dir)
        output_fp = f'{output_dir}/preds_{site_id}_{version_id}.tif'

        output =  predict_site(site_id, 
                               checkpoint_dir,
                               task_conditions_dict,
                               img_layers = img_layers,
                               output_fp = output_fp,
                               transforms_pred_model = transforms_pred_model,
                               pred_args = pred_args,
                               nodata_value = -9999.,
                               img_nodata_value = -9999.,
                               nodata_pad = 5,
                               device = device)
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config_fp')
    # args = parser.parse_args()
    # config_fp = args.config_fp
    #config_fp = 'experiment_configs/train_baseline_local_models.yaml'
    cfg = read_config_file(config_yaml = sys.argv[1])
    base_name = cfg['base_name']
    exp_name = cfg['exp_name']
    version_id_base = cfg['version_id_base']
    split_seed = cfg['data']['split_seed']
    
    for split_number in range(1):
        version_id = f'{version_id_base}_split_{split_number}_seed_{split_seed}'
        exp_root_dir = f'{base_name}/{exp_name}'
        checkpoint_dir = f'{exp_root_dir}/models/{version_id}'
        out_dir_split = make_model_output_dirs(exp_name, split_seed, split_number)
        test_sites = get_site_splits(split_seed)[split_number]['test_sites']
    
        run_experiment_models_through_one_split(test_sites,
                                                checkpoint_dir,
                                                out_dir_split,
                                                version_id, 
                                                cfg)
                                                