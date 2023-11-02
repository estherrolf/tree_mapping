import torch
import pixelwise_regression_task_with_mask
import rasterio
import tqdm 
import numpy as np
import os
from torch.utils.data import DataLoader
from torchgeo.datasets import RasterDataset, Sentinel2, stack_samples, AsterGDEM, IntersectionDataset
from torchgeo.samplers import GridGeoSampler

from chm_datamodule import ChmDataModule, transforms_4_channel_rgbnir_plus_mask_imagestats, transforms_4_channel_rgbnir_imagestats, make_site_dataset

import matplotlib.pyplot as plt 

sentinel_layer_codes = {"b": "B02",
                        "g": "B03",
                        "r": "B04",
                        "nir":"B08",
                        "vis":"TCI",
                       }

# predict and save as tif for a given site
def predict_site(eval_input_dir, 
                 exp_dir,
                 model_id,
                 model_conditions_dict,
                 output_fp,
                 pred_args = {},
                 transforms_pred_model = transforms_4_channel_rgbnir_plus_mask_imagestats,
                 nodata_value = -9999.,
                 img_nodata_value = -9999.,
                 nodata_pad= 5,
                 device='gpu',
                 model_suffix='train_sites_17_de_de_08_01_06_15',
                 path_to_data='data'
                ):
    """
    nodata_pad will pad any predictions within that many pixels of a nodata input image as nodata output.
    """
    
    # read the relavant filenames
    # load the model for evaluation
    model_dir = f"{exp_dir}/models"
    print(model_suffix)
    task_eval, model_fp = load_model_to_task(model_conditions_dict, model_dir, model_suffix=model_suffix)
    task_eval = task_eval
    
    return predict_site_with_model(task_eval, 
                                   output_fp,
                                   pred_arg=pred_args, 
                                   transforms_pred_model = transforms_4_channel_rgbnir_plus_mask_imagestats,
                                   nodata_value = nodata_value,
                                   img_nodata_value = img_nodata_value,
                                   nodata_pad= nodata_pad,
                                   device=device,
                                   path_to_data=path_to_data)

    
def predict_site_with_model(site_id, 
                            model,
                            output_fp,
                            pred_args = {},
                  #          needs_dem=False,
                            context_model=False,
                            num_contexts=2,
                            nodata_value = -9999.,
                            img_nodata_value = -9999.,
                            nodata_pad= 5,
                            device='gpu',
                            model_suffix='train_sites_17_de_de_08_01_06_15',
                            path_to_data='data'
                            ):

    
    model = model.to(device).eval()                              
    pad = pred_args['padding']
    
    dataset = make_site_dataset(site_id, 
                                layers=['r','g','b','nir','vis', 'canopy'], # no CHM
                                transforms=transforms_4_channel_rgbnir_imagestats)

#     get the input image file for evaluation
#     if needs_dem:
#         dem_data_dir = f'{path_to_data}/dem'
#         def aster_transform(sample):
#             sample['dem'] = sample['mask'].clone()
#             return sample

#         dem = AsterGDEM(root=dem_data_dir, transforms=aster_transform)


#         s2_dataset = Single4ChannelSentinel2Dataset(eval_input_dir, 
#                                                     transforms=None)
#         s2_dataset = IntersectionDataset(s2_dataset, dem, transforms=transforms_4_channel_rgbnir_plus_mask_imagestats)
        
 #   canopy_dataset = RasterDataset(root=os.path.join(data_dir, canopy_relative_dir, site_id))
    # s2_dataset = Single4ChannelSentinel2Dataset(eval_input_dir, 
    #                                                 transforms=transforms_4_channel_rgbnir_plus_mask_imagestats)

    sampler = GridGeoSampler(dataset, pred_args['patch_size'], stride=pred_args['stride'])

    dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=pred_args['batch_size'],
            num_workers=pred_args['num_workers'],
            collate_fn=stack_samples,
        )
    
    # run inference
    # make an array of the right shape
    eval_input_dir = f'/n/home10/erolf/tree_mapping/data/int/sentinel/sentinel_by_site_32736_10m/{site_id}'
    example_tif = [x for x in os.listdir(eval_input_dir) if x.endswith(".tif")][0]
    with rasterio.open(os.path.join(eval_input_dir, example_tif)) as f:
            input_height, input_width = f.shape
            profile = f.profile
            transform = profile["transform"]
    output = np.ones((input_height, input_width), dtype=np.float32) * nodata_value
    
    if context_model:
        output_context = np.ones((num_contexts, input_height, input_width), dtype=np.float32) * nodata_value

    dl_enumerator = tqdm.tqdm(dataloader)

    # modification of code from Caleb
    for batch in dl_enumerator:
            images = batch["image"].to(device)
            bboxes = batch["bbox"]
            with torch.inference_mode():
                
                if context_model:
                    context = batch["context"].to(device)
                    
                    (predictions, y_hat_by_context), context = model(images, context)
                    
#                     if model.context_integrator_type in ['weighted_avg','attn']:
#                         predictions, context = model(images, context)
#                     elif model.context_integrator_type in ['separated_by_context']:
                        
#                     else:
#                         print(f"context integrator type {model.context_integrator_type} not recognized")
                else:
                    predictions = model(images)

                predictions = predictions[:,0].cpu().numpy()
                if context_model:
                    context = context.cpu().numpy()
                #anything within pad pixels of a nodata img pixels gets a nodata label
                img_nodata_ = (images == img_nodata_value).any(dim=1).cpu().numpy().astype(float)
                
                padder = torch.nn.modules.Conv2d(in_channels=1, out_channels=1, kernel_size=nodata_pad*2+1, stride=1, padding=0)
                padder.weight = torch.nn.Parameter(torch.ones_like(padder.weight), requires_grad=False)
            
                img_nodata_padded = padder(torch.Tensor(img_nodata_)).cpu().numpy()
                predictions[:,nodata_pad:-nodata_pad, nodata_pad:-nodata_pad][img_nodata_padded > 1 ] = nodata_value
                
                if context_model:
                    for c in range(num_contexts):
                        context[:,c,nodata_pad:-nodata_pad, nodata_pad:-nodata_pad][img_nodata_padded > 1 ] = nodata_value

                
            for i in range(len(bboxes)):
                bb = bboxes[i]

                left, top = ~transform * (bb.minx, bb.maxy)
                right, bottom = ~transform * (bb.maxx, bb.miny)
                left, right, top, bottom = int(np.round(left)), int(np.round(right)), int(np.round(top)), int(np.round(bottom))

                assert right - left == pred_args['patch_size']
                assert bottom - top == pred_args['patch_size']

                
                # image indexed so in integegers top < bottom
                output[top+pad:bottom-pad, left+pad:right-pad] = predictions[i][pad:-pad, pad:-pad]
                
                if context_model:
                    output_context[:,top+pad:bottom-pad, left+pad:right-pad] = context[i][:,pad:-pad, pad:-pad]

                    
    #save the output
    if output_fp is not None:
        profile_out = profile.copy()
        profile_out.update(dtype='float32', count='1')

        print('saving predictions in ',output_fp)
        with rasterio.open(output_fp, 'w',**profile_out) as dst:
            dst.write(output, 1)
            
        if context_model:
            profile_out.update(dtype='float32', count=num_contexts)
            with rasterio.open(output_fp.replace('.tif', '_context.tif'), 'w',**profile_out) as dst:
                dst.write(output_context)
        
    return output

    
                    
class Single4ChannelSentinel2Dataset(Sentinel2):
    
    img_layers = ['r','g','b','nir','vis']
        
    bands = [sentinel_layer_codes[l.lower()] for l in img_layers]
    
    def __init__(self, root_dir, transforms):
        super().__init__(root=root_dir, 
                         bands=self.bands,
                         transforms=transforms)
        
        
def load_model_to_task(conditions_dict, 
                       model_dir, 
                       model_type='fcn',
                       model_suffix='train_sites_17_de_de_08_01_06_15'
                      ):
    if model_type != 'fcn': 
        print('only using FCN for now')
        return
    
    lr, num_filters = conditions_dict['lr'], conditions_dict['num_filters']
    
    this_model_fn = f"filters_{num_filters}_lr_{lr}_{model_suffix}.fcn"

    model_load_path = f"{model_dir}/{this_model_fn}"
    
    task = pixelwise_regression_task_with_mask.PixelwiseRegressionTaskWithMask(model='fcn', 
                                   loss='mse',
                                   learning_rate_schedule_patience=10,
                                   weights=None,
                                   in_channels=4,
                                   num_classes=1, 
                                   **conditions_dict
                            )

    task.model.load_state_dict(torch.load(model_load_path))
    
    return task, model_load_path
