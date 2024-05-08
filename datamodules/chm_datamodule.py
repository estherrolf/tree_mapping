import json
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import sys
import torch
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import IntersectionDataset, RasterDataset, Sentinel2, UnionDataset
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchvision.transforms import Compose
from typing import Any, Optional, Dict

sys.path.insert(0, '') # necessary since utils is outside the datamodules folder
from utils import get_project_dir

nir_band = 3
label_band = 7
vis_band_start = 4
vis_band_end = 7

project_dir = get_project_dir()
data_stats_dir = f'{project_dir}/data/int/data_stats'

S2_stats_by_channel = json.load(open(f'{data_stats_dir}/S2_stats_by_channel.json'))

sentinel_layer_codes = {'b': 'B02',
                        'g': 'B03',
                        'r': 'B04',
                        'nir': 'B08',
                        'vis': 'TCI'}

rgbnir_codes = ['B04', 'B03','B02', 'B08']
sentinel_layer_means_4_channel = [S2_stats_by_channel[channel]['mean'] for channel in rgbnir_codes]
sentinel_layer_stds_4_channel = [S2_stats_by_channel[channel]['std'] for channel in rgbnir_codes]

rgb_codes = ['B04', 'B03','B02']
sentinel_layer_means_3_channel = [S2_stats_by_channel[channel]['mean'] for channel in rgb_codes]
sentinel_layer_stds_3_channel = [S2_stats_by_channel[channel]['std'] for channel in rgb_codes]

s2_12_channel_codes = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
# as per Lang et al.
s2_12_channel_plus_latlon_codes = s2_12_channel_codes + ['lat', 'sin(lon)', 'cos(lon)']

sentinel_layer_means_12_channel_plus_latlon = [S2_stats_by_channel[channel]['mean'] for channel in s2_12_channel_plus_latlon_codes]
sentinel_layer_stds_12_channel_plus_latlon = [S2_stats_by_channel[channel]['std'] for channel in s2_12_channel_plus_latlon_codes]

def get_default_layers_and_transforms(num_image_channels, predict_mode = False):
    if num_image_channels == 3:
        data_layers = ['r', 'g', 'b', 'vis', 'chm']
        batch_transforms = transforms_3_channel
    elif num_image_channels == 4:
        data_layers = ['r', 'g', 'b', 'nir', 'vis', 'chm']
        batch_transforms = transforms_4_channel
    elif num_image_channels == 12:
        batch_transforms = transforms_12_channel
        data_layers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] + ['vis', 'chm']
    elif num_image_channels == 13:
        data_layers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] + ['vis', 'chm']
        batch_transforms = transforms_13_channel
    elif num_image_channels == 15:
        data_layers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] + ['vis', 'chm']
        batch_transforms = transforms_15_channel
    else:
        print(f'No directive for {num_image_channels} image channels')
        
    # no CHM if in predict mode
    if predict_mode: data_layers = data_layers[:-1]
    return data_layers, batch_transforms


def degree_to_radian(x):
        return (2 * np.pi) * x / 360.  
    
def bounds_to_latlon_encoding(bds, h,w, src_crs):
    # convert bounds and pixel size to a lat-lon encoding per pixel, according to the encoding
    # used in Lang et al.
    
    dst_crs = rasterio.crs.CRS.from_epsg('4326')
    bds_degrees = rasterio.warp.transform_bounds(src_crs, dst_crs, *bds)

    # get latitude and longitude per pixel (in degrees)
    # top to bottom for latitude
    lat_per_pixel_vals = np.linspace(bds_degrees[3], bds_degrees[1], num=h)
    lat_per_pixel = np.vstack([lat_per_pixel_vals for x in range(w)]).T
    lon_per_pixel_vals = np.linspace(bds_degrees[0], bds_degrees[2], num=w)
    lon_per_pixel = np.vstack([lon_per_pixel_vals for x in range(h)])
    
    latlon_encoding = np.zeros((3,h,w))
    # lat (in degrees)
    latlon_encoding[0] = lat_per_pixel
    # sin(lon)
    latlon_encoding[1] = np.sin(degree_to_radian(lon_per_pixel))
    latlon_encoding[2] = np.cos(degree_to_radian(lon_per_pixel))
    return latlon_encoding


def transforms_15_channel(sample, img_nodata_val=-9999., mask_nodata_val=-9999., use_image_stats=True):
    # s2 bands
    num_image_bands = len(s2_12_channel_codes)
    num_vis_bands = 3
    img_nodata_mask = (sample['image'][:num_image_bands] == img_nodata_val).any(axis=0)
    
    # seventh band is the label, separate it 
    label_band = num_image_bands + num_vis_bands
    has_mask = len(sample['image']) > label_band
    if has_mask:
        sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
    
    # these three bands are the visual image, separate them
    if len(sample['image']) > num_image_bands+1:
        sample['vis'] = sample['image'][num_image_bands:num_image_bands+num_vis_bands].clone()    
        
    # bands 0-11 are the image
    sample['image'] = sample['image'][:num_image_bands].clone() 
    
    # append the latlon to the sample
    bbox = sample['bbox']
    bds = [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]   
    h,w = sample['image'].shape[1], sample['image'].shape[2]
    src_crs = sample['crs']
    
    latlon_layers = bounds_to_latlon_encoding(bds, h,w, src_crs)
    sample['image'] = torch.vstack((sample['image'], torch.Tensor(latlon_layers)))    
    
    if use_image_stats:
        means = sentinel_layer_means_12_channel_plus_latlon
        stds = sentinel_layer_stds_12_channel_plus_latlon
    else:
        # NOTE: this won't make sense for the latlon values...
        means = torch.Tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.Tensor([10000. for x in range(len(sentinel_layer_means))])
    
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample


    
def transforms_13_channel(sample, img_nodata_val=-9999., mask_nodata_val=-9999., use_image_stats=True):
    # use e.g. when you need the B10 band for pretrained models
    
    # s2 bands
    num_image_bands = len(s2_12_channel_codes)
    num_vis_bands = 3
    img_nodata_mask = (sample['image'][:num_image_bands] == img_nodata_val).any(axis=0)
    
    # seventh band is the label, separate it 
    label_band = num_image_bands + num_vis_bands
    has_mask = len(sample['image']) > label_band
    if has_mask:
        sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
    
    # these three bands are the visual image, separate them
    if len(sample['image']) > num_image_bands+1:
        sample['vis'] = sample['image'][num_image_bands:num_image_bands+num_vis_bands].clone()    

    # bands 0-11 are the image
    sample['image'] = sample['image'][:num_image_bands].clone() 
    
    if use_image_stats:
        means = sentinel_layer_means_12_channel_plus_latlon
        stds = sentinel_layer_stds_12_channel_plus_latlon
    else:
        # NOTE: this won't make sense for the latlon values...
        means = torch.Tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.Tensor([10000. for x in range(len(sentinel_layer_means))])
    
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    B10 = np.zeros((1, *sample['image'].shape[1:]), dtype=sample['image'].numpy().dtype)
    image = np.concatenate([sample['image'][:10], B10, sample['image'][10:]], axis=0)
    sample['image'] = torch.tensor(image)
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample

def transforms_12_channel(sample, img_nodata_val=-9999., mask_nodata_val=-9999., use_image_stats=True):
    # s2 bands
    num_image_bands = len(s2_12_channel_codes)
    num_vis_bands = 3
    img_nodata_mask = (sample['image'][:num_image_bands] == img_nodata_val).any(axis=0)
    
    # separate label band if it's in the list of layers
    label_band = num_image_bands + num_vis_bands
    has_mask = len(sample['image']) > label_band
    if has_mask:
        sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
    
    # these three bands are the visual image, separate them
    if len(sample['image']) > num_image_bands+1:
        sample['vis'] = sample['image'][num_image_bands:num_image_bands+num_vis_bands].clone()    

    # bands 0-11 are the image
    sample['image'] = sample['image'][:num_image_bands].clone() 
    
    if use_image_stats:
        means = sentinel_layer_means_12_channel_plus_latlon
        stds = sentinel_layer_stds_12_channel_plus_latlon
    else:
        # NOTE: this won't make sense for the latlon values...
        means = torch.Tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.Tensor([10000. for x in range(len(sentinel_layer_means))])
    
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample

def transforms_3_channel(sample, img_nodata_val=-9999., mask_nodata_val=-9999., use_image_stats=True):
    img_nodata_mask = (sample['image'][:3] == img_nodata_val).any(axis=0)
    
    # sixth band is the label, separate it 
    label_band = 6
    has_mask = len(sample['image']) > label_band
    if has_mask:
        sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
        
    # these three bands are the visual image, separate them
    if len(sample['image']) > 4:
        sample['vis'] = sample['image'][3:6].clone()    
        
    # bands 0-3 are the image
    sample['image'] = sample['image'][:3].clone() 

    if use_image_stats:
        means = sentinel_layer_means_3_channel
        stds = sentinel_layer_stds_3_channel
    else:
        means = torch.Tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.Tensor([10000. for x in range(len(sentinel_layer_means))])
    
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample

def transforms_4_channel(sample, img_nodata_val=-9999., mask_nodata_val=-9999., use_image_stats=True):
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
    
    # seventh band is the label, separate it 
    label_band = 7
    has_mask = len(sample['image']) > label_band
    if has_mask:
        sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
        
    # these three bands are the visual image, separate them
    if len(sample['image']) > 5:
        sample['vis'] = sample['image'][4:7].clone()    
        
    # bands 0-4 are the image
    sample['image'] = sample['image'][:4].clone() 

    if use_image_stats:
        means = sentinel_layer_means_4_channel
        stds = sentinel_layer_stds_4_channel
    else:
        means = torch.Tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.Tensor([10000. for x in range(len(sentinel_layer_means))])
    
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample

def make_site_dataset(site_id,  
                      transforms, 
                      layers=[],
                      data_dir=f'{project_dir}/data',
                      chm_relative_dir='int/lidar/lidar_by_site_32736_10m',
                      sentinel_relative_dir='int/sentinel/sentinel_by_site_32736_10m',
                      canopy_relative_dir='int/alos/alos_by_site_20_FNF'):
    '''
    Returns a dataset with layers in this order: 
        sentinel, then CHM (if requested), then context data (e.g. canopy -- if requested).
    '''
    
    non_img_layers = ['dem','chm', 'canopy']
    img_layers = [l for l in layers if not l in non_img_layers]
    s2_bands = []
    for l in img_layers:
        # convert e.g. nir to the S2 code, keep e.g. B02 as is
        if l.lower() in ['r','g','b','nir','vis']: 
            s2_bands.append(sentinel_layer_codes[l.lower()])
        else:
            s2_bands.append(l)
                                         
    
    sentinel_data_dir = os.path.join(data_dir,sentinel_relative_dir, site_id)
    # dataset with chm labels
    if 'chm' in layers: 
        # gather the sentinel data
        sentinel = Sentinel2(
            sentinel_data_dir,
            bands=s2_bands
        )
        
        # gathers data
        chm_dataset = RasterDataset(paths=os.path.join(data_dir,chm_relative_dir, site_id))
      
        if 'canopy' in layers:            
            dataset = IntersectionDataset(sentinel, chm_dataset)            
            canopy = RasterDataset(paths=os.path.join(data_dir, canopy_relative_dir, site_id))
            dataset = IntersectionDataset(ds, canopy, transforms=transforms)
            
        else:    
            dataset = IntersectionDataset(sentinel, chm_dataset, transforms=transforms)
        
    # dataset without labels
    else:
        if 'canopy' in layers:  
            sentinel = Sentinel2(
                sentinel_data_dir,
                bands=s2_bands
            )
        
            canopy = RasterDataset(paths=os.path.join(data_dir, canopy_relative_dir, site_id))
            dataset = IntersectionDataset(sentinel, canopy, transforms=transforms)
        
        else:
            dataset = Sentinel2(
                sentinel_data_dir,
                bands=s2_bands, 
                transforms=transforms
        ) 
        
    return dataset

def get_chm_sites(sites,
                  layers,
                  transforms=None):
    
    assert len(sites) > 0
    
    # transofrms should only be applied to the last one 
    # so in case there is just one site:
    if len(sites) == 1: transforms_site_0  = transforms
    else: transforms_site_0 = None
        
    chms = make_site_dataset(sites[0],
                             transforms=transforms_site_0,
                             layers=layers, 
                             data_dir=f'{project_dir}/data')
        
   
    if len(sites) > 1:
        remaining_sites = sites[1:]
        for i, site in enumerate(remaining_sites):
            if i == len(remaining_sites) - 1:
                transforms_this = transforms
            else:
                transforms_this = None
            
            chm_this = make_site_dataset(site,
                                         transforms=None,
                                         layers=layers, 
                                         data_dir=f'{project_dir}/data')
            
            chms = UnionDataset(chms, 
                                chm_this,
                                transforms=transforms_this)
            
    return chms


class ChmDataModule(GeoDataModule):
    def __init__(
        self,
        train_sites: list[str],
        val_sites: list[str],
        test_sites: list[str],
        layers: list[str] = ['r','g','b','nir','vis','chm'],
        batch_size: int = 8,
        patch_size: int = 64,
        eval_pad: int = 5,
        length: int = 1000,
        num_workers: int = 0,
        batch_transforms =  transforms_4_channel,
        **kwargs: Any,
    ) -> None:
        
        super().__init__(
            get_chm_sites, batch_size, patch_size, length, num_workers, **kwargs
        )

        self.train_sites = train_sites
        self.val_sites = val_sites
        self.test_sites = test_sites
        self.layers = layers
        self.original_patch_size = self.patch_size #* 2
        self.eval_stride = self.patch_size - 2 *eval_pad
        self.plt_vmax = 15
        self.plot_dem = 'dem' in layers
        
        self.train_transforms = Compose(
            [
                batch_transforms,
            ]
        )
        
        self.val_transforms= Compose(
            [
                batch_transforms,
            ]
        )
        
        self.test_transforms = Compose(
            [
                batch_transforms,
            ]
        )
        
    def setup(self, stage: str) -> None:
        if stage in ['fit']:
            self.train_dataset = get_chm_sites(
                sites=self.train_sites, layers=self.layers, transforms = self.train_transforms, **self.kwargs
            )
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset,
                self.original_patch_size,
                self.batch_size,
                self.length,
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = get_chm_sites(
                sites=self.val_sites, layers=self.layers, transforms = self.val_transforms, **self.kwargs
            )
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.original_patch_size, self.eval_stride, units=Units.PIXELS
            )
        if stage in ['test']:
            self.test_dataset = get_chm_sites(
                sites=self.test_sites, layers=self.layers, transforms = self.test_transforms, **self.kwargs
            )
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.original_patch_size, self.eval_stride, units=Units.PIXELS
            )
            
    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        pad: Optional[int] = 0,
        zero_out_nan_vis=False,
        nan_val = -9999.,
    ) -> plt.Figure:
        '''Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        '''
        
        if 'mask' in sample.keys():
            mask = sample['mask'].squeeze(0).cpu().numpy()
        else:
            mask = np.zeros_like(sample['vis'])
        vis = sample['vis'].cpu().numpy() 
        nan_plot_val = 0
        
        if pad > 0:
            vis = vis[:,pad:-pad,pad:-pad]
            mask = mask[pad:-pad,pad:-pad]
                        
        nan_mask = mask == nan_val
        mask[nan_mask] = nan_plot_val
         
        if zero_out_nan_vis:
            for c in range(vis.shape[0]):
                vis[c][nan_mask] = nan_plot_val
        
        
        showing_predictions = 'prediction' in sample
        
        ncols = 2
        if showing_predictions:
            ncols += 1
            pred = sample['prediction'].squeeze(0).cpu().numpy()
            pred[nan_mask] = nan_plot_val
            
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
        axs[0].imshow(
                vis.transpose(1,2,0) / 255.,
                interpolation='none',
            )
        axs[1].imshow(
                mask,
                vmin=0,
                vmax=self.plt_vmax,
                cmap='Greens',
                interpolation='none',
            )
        axs[0].axis('off')
        axs[1].axis('off')
        
        if show_titles:
            axs[0].set_title('Img (3 channel)')
            axs[1].set_title('Mask')
        
        if showing_predictions:
            axs[2].imshow(
                pred,
                vmin=0,
                vmax=self.plt_vmax,
                cmap='Greens',
                interpolation='none',
            )
            axs[2].axis('off')
            if show_titles:
                axs[2].set_title('Prediction')

        if suptitle is not None:
            plt.suptitle(suptitle)
        
        return fig
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # we normalize in the transforms
        # so override the torchgeo defaults so this function does nothing
        
        return batch