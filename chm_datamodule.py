from typing import Any, Optional, Dict
import matplotlib.pyplot as plt
import os
import torch
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, Sentinel2, stack_samples, UnionDataset
from torchgeo.samplers import RandomBatchGeoSampler,  GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.datasets import IntersectionDataset, RasterDataset, Sentinel2, stack_samples, AsterGDEM
#import torchvision.transforms 
from torchvision.transforms import Compose
import torch.nn as nn
from einops import rearrange
from typing import Any

import numpy as np
import kornia.augmentation as K
from torchgeo.transforms import AugmentationSequential
from typing import Callable
from torch import Tensor

import sys
sys.path.append('.')
#from utils import make_site_dataset

nir_band = 3
label_band = 7
vis_band_start = 4
vis_band_end = 7

DATA_DIR = "/n/home10/erolf/tree_mapping/data"

sentinel_layer_codes = {"b": "B02",
                        "g": "B03",
                        "r": "B04",
                        "nir":"B08",
                        "vis":"TCI",
                       }

# unused
# S2_transforms = AugmentationSequential(
#     K.Normalize(mean=torch.tensor(0), std=torch.tensor(10000)),
#     data_keys=["image"],
# )

# S2_transforms_image_stats = AugmentationSequential(
#     K.Normalize(mean=torch.tensor(sentinel_layer_means), std=torch.tensor(sentinel_layer_stds)),
#     data_keys=["image"],
# )



# assums in order r g b nir
# B02: (1409.4317337430034, 142.92678697740124)
# B03: (1648.2400868817622, 169.20665915193652)
# B04: (1650.8489919140115, 257.143099151872)
# B08: (3513.2088906825957, 422.61985470380745)

sentinel_layer_means = torch.Tensor([1650.8489919140115, 1648.2400868817622, 1409.4317337430034, 3513.2088906825957])
sentinel_layer_stds = torch.Tensor([257.143099151872, 169.20665915193652, 142.92678697740124, 422.61985470380745])


def transforms_4_channel_rgbnir_plus_mask_imagestats(sample, img_nodata_val=-9999., use_image_stats=True):
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
    if img_nodata_mask.any(): print('NODATA VAL detected in imagery')
        
    # seventh band is the label, separate it 
    label_band = 7
    sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
    sample['mask'][sample['mask'] > 30] = -9999.
        
    # last three bands are the visual image, separate them
    if len(sample['image']) > 5:
        sample['vis'] = sample['image'][4:7].clone()    
    
    # if there is context data to be had
    if len(sample['image']) > 8:
        sample['context'] = sample['image'][8:].clone().long() 
        # assumes contexts is a 4 channel canopy map 
        sample['context'] = torch.nn.functional.one_hot(sample['context']-1, num_classes=4).float()
        sample['context'] = sample['context'].transpose(0,3).squeeze()
        
    # bands 0-4 are the image
    sample['image'] = sample['image'][:4].clone() 
    # trying to do this later
        # divide s2 bands by 10000.
        
    if use_image_stats:
        means = sentinel_layer_means
        stds = sentinel_layer_stds
    else:
        means = torch.tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.tensor([10000. for x in range(len(sentinel_layer_means))])
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample

def transforms_4_channel_rgbnir_imagestats(sample, img_nodata_val=-9999., use_image_stats=True):
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
    if img_nodata_mask.any(): print('NODATA VAL detected in imagery')
        
    # last three bands are the visual image, separate them
    if len(sample['image']) > 5:
        sample['vis'] = sample['image'][4:7].clone()    
    
    # if there is context data to be had
    if len(sample['image']) > 7:
        sample['context'] = sample['image'][7:].clone().long() 
        # assumes contexts is a 4 channel canopy map 
        sample['context'] = torch.nn.functional.one_hot(sample['context']-1, num_classes=4).float()
        sample['context'] = sample['context'].transpose(0,3).squeeze()
        
    # bands 0-4 are the image
    sample['image'] = sample['image'][:4].clone() 
    # trying to do this later
        # divide s2 bands by 10000.
        
    if use_image_stats:
        means = sentinel_layer_means
        stds = sentinel_layer_stds
    else:
        means = torch.tensor([0. for x in range(len(sentinel_layer_means))])
        stds = torch.tensor([10000. for x in range(len(sentinel_layer_means))])
    for b in range(len(sample['image'])):
        sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample

# def transforms_4_channel_rgbnir_plus_mask(sample, img_nodata_val=-9999., use_image_stats=False):
#     img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
#     if img_nodata_mask.any(): print('NODATA VAL detected in imagery')
        
#     # last band is the label, separate it 
#     label_band = 7 
#     sample['mask'] = torch.Tensor(sample['image'][label_band]).clone()
#     sample['mask'][sample['mask'] > 30] = -9999.
    
#     # last three bands are the visual image, separate them
#     if len(sample['image']) > 5:
#         sample['vis'] = sample['image'][4:7].clone()    
        
#     # if there is context data to be had
    
#     if len(sample['image']) > 9:
#         sample['context'] = sample['image'][8:].clone()  
#         # assumes contexts is a 4 channel canopy map 
#         print(sample['context'].shape)
#         sample['context'] = torch.nn.functional(sample['context']-1, num_classes=4)
#         print(sample['context'].shape)
#         print(sample['context'].shape)
    
#     # bands 0-4 are the image
#     sample['image'] = sample['image'][:4].clone() 
#     # trying to do this later
#         # divide s2 bands by 10000.
        
#     if use_image_stats:
#         means = sentinel_layer_means
#         stds = sentinel_layer_stds
#     else:
#         means = torch.tensor([0. for x in range(len(sentinel_layer_means))])
#         stds = torch.tensor([10000. for x in range(len(sentinel_layer_means))])
#     for b in range(len(sample['image'])):
#         sample['image'][b] = (sample['image'][b].float() - means[b]) / stds[b]
        
#     sample['image'][:,img_nodata_mask] = img_nodata_val
#     return sample


def make_site_dataset(site_id,  
                      transforms, 
                      layers=[],
                      sentinel_data_dir=None,
                      canopy_relative_dir = 'int/alos/alos_by_site_20_FNF',
                      data_dir=DATA_DIR):
    
    non_img_layers = ['dem','chm', 'canopy']
    img_layers = [l for l in layers if not l in non_img_layers]
    s2_bands = [sentinel_layer_codes[l.lower()] for l in img_layers]
    
    if sentinel_data_dir is None:
        sentinel_dir_this_site = f'int/sentinel/sentinel_by_site_32736_10m/{site_id}'
        sentinel_data_dir = os.path.join(data_dir,sentinel_dir_this_site)

    if "chm" in layers: 
        # gather the sentinel data
        sentinel = Sentinel2(
            sentinel_data_dir,
            bands=s2_bands
        )
        
        # gathers data
        chm_dataset = RasterDataset(root=os.path.join(data_dir,f'int/lidar/lidar_by_site_32736_10m/{site_id}'))

        # CHM will go last
        ds = IntersectionDataset(sentinel, chm_dataset, transforms=transforms)
        
        if "canopy" in layers:            
            ds = IntersectionDataset(sentinel, chm_dataset)
           # ds = IntersectionDataset(chm_dataset, sentinel)
            
            canopy = RasterDataset(root=os.path.join(data_dir, canopy_relative_dir, site_id))
            ds = IntersectionDataset(ds, canopy, transforms=transforms)
        else:    
            ds = IntersectionDataset(sentinel, chm_dataset, transforms=transforms)
        
    else:
        if "canopy" in layers:  
            sentinel = Sentinel2(
                sentinel_data_dir,
                bands=s2_bands
            )
        
            canopy = RasterDataset(root=os.path.join(data_dir, canopy_relative_dir, site_id))
            ds = IntersectionDataset(sentinel, canopy, transforms=transforms)
        
        else:
            ds = Sentinel2(
                sentinel_data_dir,
                bands=s2_bands, 
                transforms=transform
        ) 
        
        
    return ds

def get_chm_sites(sites,
                  layers,
                  transforms=None):
    
    assert len(sites) > 0
    
    # in case there is just one site
    if len(sites) == 1:
        transforms_site_0  = transforms
    else:
        transforms_site_0 = None
        
    chms = make_site_dataset(sites[0],
                             transforms=transforms_site_0,
                             layers=layers, 
                             data_dir=DATA_DIR)
        
   
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
                                         data_dir=DATA_DIR,
                                        )
            
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
        eval_pad = 5,
        length: int = 1000,
        num_workers: int = 0,
        batch_transforms =  transforms_4_channel_rgbnir_plus_mask_imagestats,
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
        print(self.eval_stride)
        self.plt_vmax = 15
        self.plot_dem = 'dem' in layers
        
        self.train_transforms = Compose(
            [
                batch_transforms,
       #         S2_transforms
            ]
        )
        
        self.val_transforms= Compose(
            [
                batch_transforms,
         #       S2_transforms
            ]
        )
        
        self.test_transforms = Compose(
            [
                batch_transforms,
          #      S2_transforms
            ]
        )
        
    def setup(self, stage: str) -> None:
        if stage in ["fit"]:
            self.train_dataset = get_chm_sites(sites=self.train_sites, 
                                               layers=self.layers, 
                                               transforms = self.train_transforms,
                                               **self.kwargs)
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset,
                self.original_patch_size,
                self.batch_size,
                self.length,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = get_chm_sites(
                sites=self.val_sites, layers=self.layers, transforms = self.val_transforms, **self.kwargs
            )
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.original_patch_size, self.eval_stride, 
                units=Units.PIXELS
            )
        if stage in ["test"]:
            self.test_dataset = get_chm_sites(
                sites=self.test_sites, layers=self.layers, transforms = self.test_transforms, **self.kwargs
            )
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.original_patch_size, self.eval_stride, 
                units=Units.PIXELS
            )
            
    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        pad: Optional[int] = 0
    ) -> plt.Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`RasterDataset.__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional suptitle to use for figure

        Returns:
            a matplotlib Figure with the rendered sample

        .. versionchanged:: 0.3
           Method now takes a sample dict, not a Tensor. Additionally, possible to
           show subplot titles and/or use a custom suptitle.
        """
        mask = sample["mask"].squeeze(0).cpu().numpy()
        vis = sample["vis"].cpu().numpy() #/ 255.
        nan_plot_val = 0
        
        if pad > 0:
            vis = vis[:,pad:-pad,pad:-pad]
            mask = mask[pad:-pad,pad:-pad]
            
        nan_mask = mask < 0
        mask[nan_mask] = nan_plot_val
        
        for c in range(vis.shape[0]):
            vis[c][nan_mask] = nan_plot_val
        
        
        showing_predictions = "prediction" in sample
        showing_context = "context" in sample
        
        ncols = 2
        if showing_predictions:
            ncols += 1
            pred = sample["prediction"].squeeze(0).cpu().numpy()
            pred[nan_mask] = nan_plot_val
            
        if showing_context:
            ncols += 2
            c = sample["context"].squeeze(0).cpu().numpy()
            if len(c.shape) > 2:
                c[:,nan_mask] = nan_plot_val
            else:
                c[nan_mask] = nan_plot_val
        
        if self.plot_dem:
            ncols += 1
            dem = sample["image"][-1].squeeze(0).cpu().numpy()
            
        
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
        axs[0].imshow(
                vis.transpose(1,2,0) / 255.,
                interpolation="none",
            )
        axs[1].imshow(
                mask,
                vmin=0,
                vmax=self.plt_vmax,
                cmap='Greens',
                interpolation="none",
            )
        axs[0].axis("off")
        axs[1].axis("off")
        
        if show_titles:
            axs[0].set_title("Img (3 channel)")
            axs[1].set_title("Mask")
        
        if showing_predictions:
            axs[2].imshow(
                pred,
                vmin=0,
                vmax=self.plt_vmax,
                cmap='Greens',
                interpolation="none",
            )
            axs[2].axis("off")
            if show_titles:
                axs[2].set_title("Prediction")
                
        # if self.plot_dem:
        #     axs_dem = axs[3]
        #     axs_dem.imshow(
        #         dem * dem_stats['std'] + dem_stats['mean'],
        #         vmin=0, vmax=500,
        #         cmap='gray',
        #         interpolation="none",
        #     )
        #     axs_dem.axis("off")
        #     if show_titles:
        #         axs_dem.set_title("DEM (km)")
            

        if showing_context:
            if len(c.shape) > 2:
                axs_c1 = axs[len(axs)-2]
                axs_c2 = axs[len(axs)-1]
                axs_c1.imshow(
                    c[0],
                   vmin=0, 
                   vmax=1,
                    interpolation="none",
                )
                axs_c2.imshow(
                    c[1],
                   vmin=0, 
                   vmax=1,
                    interpolation="none",
                )
                axs_c1.axis("off")
                axs_c2.axis("off")
                if show_titles:
                    axs_c1.set_title("Context (dim 1)")
                    axs_c2.set_title("Context (dim 2)")
            else:
                
                axs_c1 = axs[len(axs)-2]
                axs_c2 = axs[len(axs)-1]
                axs_c1.imshow(
                    c,
                   vmin=0, 
                   vmax=1,
                    interpolation="none",
                )
                axs_c2.imshow(
                    1-c,
                   vmin=0, 
                   vmax=1,
                    interpolation="none",
                )
                axs_c1.axis("off")
                axs_c2.axis("off")
                if show_titles:
                    axs_c1.set_title("Context (dim 1)")
                    axs_c2.set_title("Context (dim 2)")

        if suptitle is not None:
            plt.suptitle(suptitle)
        
        return fig
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # we normalize in the transforms
        # so override the torchgeo defaults so this function does nothing
        
        return batch