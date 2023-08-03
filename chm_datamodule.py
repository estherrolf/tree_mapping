from typing import Any, Optional, Dict
import matplotlib.pyplot as plt
import os
import torch
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import RasterDataset, Sentinel2, stack_samples, UnionDataset
from torchgeo.samplers import RandomBatchGeoSampler,  GridGeoSampler
from torchgeo.samplers.constants import Units
from torchgeo.datasets import IntersectionDataset, RasterDataset, Sentinel2, stack_samples
#import torchvision.transforms 
from torchvision.transforms import Compose
import torch.nn as nn
from einops import rearrange


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

# based on one sample sentinel 2 image
image_stats = {
     'B04': {'mean': 1594.1889749204547, 'std': 258.3441421186568},
     'B08': {'mean': 3513.7490286528578, 'std': 401.2509162414844},
     'B02': {'mean': 1361.7406947886702, 'std': 150.90479222977774},
     'B03': {'mean': 1605.624734241094, 'std': 177.70337752114975}
}


def my_transforms_4_channel_rgbnir_plus_mask(sample, img_nodata_val=-9999.):
    
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
          
    sample['image'][0] = (sample['image'][0] - image_stats["B04"]["mean"]) / image_stats["B04"]["std"]
    sample['image'][1] = (sample['image'][1] - image_stats["B03"]["mean"]) / image_stats["B03"]["std"]
    sample['image'][2] = (sample['image'][2] - image_stats["B02"]["mean"]) / image_stats["B02"]["std"]
    sample['image'][3] = (sample['image'][3] - image_stats["B08"]["mean"]) / image_stats["B08"]["std"]
    
    # last band for label
    label_band = len(sample['image']) - 1 
    sample['mask'] = torch.Tensor(sample['image'][label_band]).clone()
    
    if len(sample['image']) > 5:
        sample['vis'] = sample['image'][4:7]
              
            
    sample['image'] = sample['image'][:4]
    
    if img_nodata_mask.any(): print('NODATA VAL detected in imagery')
    sample['image'][:,img_nodata_mask] = img_nodata_val
    return sample
    
# def my_transforms(sample):
        
#     sample['image'][0] = (sample['image'][0] - 1400.) / 60.
#     sample['image'][1] = (sample['image'][1] - 1600.) / 60.
#     sample['image'][2] = (sample['image'][2] - 1600.) / 120.
#     sample['image'][nir_band] = (sample['image'][nir_band] - 3000.) / 300.
    
#     sample['vis'] = sample['image'][vis_band_start:vis_band_end]#.detach().clone()
        
#     # last band for label
#     label_band = len(sample['image']) - 1 
#     sample['mask'] = torch.Tensor(sample['image'][label_band]) #.detach().clone()
    
#     # 4 channel image
#     sample['remaining_data'] = sample['image'][label_band:]
#     sample['image'] = sample['image'][:4]#.detach().clone()
    
    
#     print(sample['image'].mean(axis=(1,2)))
#    # print()
    
#     return sample

def make_site_dataset(site_id,  
                      transforms, 
                      layers,
                      sentinel_data_dir = "sentinel/S2A_MSIL2A_20230418T073611_R092_T36KVU_20230419T022704",
                      data_dir=DATA_DIR):
    
    sentinel_data_dir = os.path.join(data_dir,sentinel_data_dir)
    non_img_layers = ['chm']
    img_layers = [l for l in layers if not l in non_img_layers]
    
    print(img_layers)

    s2_bands = [sentinel_layer_codes[l.lower()] for l in img_layers]
        
    if "chm" in layers: 
        sentinel = Sentinel2(
            sentinel_data_dir,
            bands=s2_bands
        )
        
        assert layers[-1] == "chm"
        chm_dataset = RasterDataset(root=data_dir + f'/lidar/Karingani_merged_crs_10/{site_id}')
        ds = IntersectionDataset(sentinel, chm_dataset, transforms=transforms)
    
    else:
        print('chm missing')
        sentinel = Sentinel2(
            sentinel_data_dir,
            bands=s2_bands,
            transforms=transforms
        )
        return sentinel
    
    return ds

def get_chm_sites(sites,
                  layers,
                  transforms=None):
    
    assert len(sites) > 0
    
    # first site
    if len(sites) == 1:
        transforms_0  = transforms
    else:
        transforms_0 = None
    chms = make_site_dataset(sites[0],
                             transforms=transforms_0,
                             layers=layers, 
                             data_dir=DATA_DIR)
        
    if len(sites) > 1:
        for i, site in enumerate(sites[1:]):
            if i == len(sites) - 2:
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

class _Transform(nn.Module):
    """Version of AugmentationSequential designed for samples, not batches."""

    def __init__(self, aug: nn.Module) -> None:
        """Initialize a new _Transform instance.

        Args:
            aug: Augmentation to apply.
        """
        super().__init__()
        self.aug = aug

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Apply the augmentation.

        Args:
            sample: Input sample.

        Returns:
            Augmented sample.
        """
        for key in ["image"]:#, "mask"]:
            dtype = sample[key].dtype
            # All inputs must be float
            sample[key] = sample[key].float()
            sample[key] = self.aug(sample[key])
            sample[key] = sample[key].to(dtype)
            # Kornia adds batch dimension
            sample[key] = rearrange(sample[key], "() c h w -> c h w")
        return sample
    

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
        batch_transforms =  None,
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
        
        self.train_transforms = Compose(
            [
         #       _Transform(K.CenterCrop(patch_size)),
                batch_transforms
            ]
        )
        
        self.val_transforms= Compose(
            [
         #       _Transform(K.CenterCrop(patch_size)),
                batch_transforms
            ]
        )
        
        self.test_transforms = Compose(
            [
           #     _Transform(K.CenterCrop(patch_size)),
                batch_transforms
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
        
        ncols = 2
        if showing_predictions:
            ncols = 3
            pred = sample["prediction"].squeeze(0).cpu().numpy()
            pred[nan_mask] = nan_plot_val
            
        
        fig, axs = plt.subplots(nrows=1, ncols=ncols, figsize=(4 * ncols, 4))
        
        if showing_predictions:
            axs[0].imshow(
                vis.transpose(1,2,0) / 255.,
                interpolation="none",
            )
            axs[1].axis("off")
            axs[1].imshow(
                mask,
                vmin=0,
                vmax=self.plt_vmax,
                cmap='Greens',
                interpolation="none",
            )
            axs[1].axis("off")
            axs[2].imshow(
                pred,
                vmin=0,
                vmax=self.plt_vmax,
                cmap='Greens',
                interpolation="none",
            )
            axs[2].axis("off")
            if show_titles:
                axs[0].set_title("Img (3 channel)")
                axs[1].set_title("Mask")
                axs[2].set_title("Prediction")

        else:
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
            axs[1].axis("off")
            if show_titles:
                axs[0].set_title("Img (3 channel)")
                axs[1].set_title("Mask")

        if suptitle is not None:
            plt.suptitle(suptitle)
        
        return fig
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # we normalize in the transforms
        # so override the torchgeo defaults so this function does nothing
        
        return batch