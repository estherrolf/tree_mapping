import json
import matplotlib.pyplot as plt
import os
import torch
from torchgeo.datamodules import GeoDataModule
from torchgeo.datasets import IntersectionDataset, RasterDataset, Sentinel2, UnionDataset
from torchgeo.samplers import RandomBatchGeoSampler, GridGeoSampler
from torchgeo.samplers.constants import Units
from torchvision.transforms import Compose
from typing import Any, Optional, Dict

nir_band = 3
label_band = 7
vis_band_start = 4
vis_band_end = 7

DATA_DIR = "/n/home10/erolf/tree_mapping/data"
# DATA_DIR = '../../../tambe_lab/Users/luciagordon/tree_mapping_lucia_branch/data'
data_stats_dir = os.path.join(DATA_DIR,"int/data_stats")


S2_stats_by_channel = json.load(open(os.path.join(data_stats_dir, "S2_stats_by_channel.json")))

sentinel_layer_codes = {"b": "B02",
                        "g": "B03",
                        "r": "B04",
                        "nir":"B08",
                        "vis":"TCI",
                       }

rgbnir_codes = ["B04", "B03","B02","B08"]
sentinel_layer_means_4_channel = [S2_stats_by_channel[channel]['mean'] for channel in rgbnir_codes]
sentinel_layer_stds_4_channel = [S2_stats_by_channel[channel]['std'] for channel in rgbnir_codes]

def transforms_4_channel_rgbnir_plus_mask_imagestats(sample, img_nodata_val=-9999., mask_nodata_val=-9999., use_image_stats=True):
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
    
    # seventh band is the label, separate it 
    label_band = 7
    sample['mask'] = torch.Tensor(sample['image'][label_band:label_band+1]).clone()
    
    # clip extreme values 
    sample['mask'][sample['mask'] > 30] = 30.
    # less than 0 is a NaN
    sample['mask'][sample['mask'] < 0] = -9999.
    
    # make sure no imagery has nodata vals if mask has vals
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
    mask_nodata_mask = (sample['mask'] == mask_nodata_val)[0]#.any(axis=0)
    if img_nodata_mask[~mask_nodata_mask].any(): print('NODATA VAL detected in imagery')
    
        
    # last three bands are the visual image, separate them
    if len(sample['image']) > 5:
        sample['vis'] = sample['image'][4:7].clone()    
    
    # if there is extra context data to be had
    if len(sample['image']) > 8:
        sample['context'] = sample['image'][8:].clone().long() 
        # assumes contexts is a 4 channel canopy map 
        sample['context'] = torch.nn.functional.one_hot(sample['context']-1, num_classes=4).float()
        sample['context'] = sample['context'].transpose(0,3).squeeze()
        
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

def transforms_4_channel_rgbnir_no_mask_imagestats(sample, img_nodata_val=-9999.,use_image_stats=True, verbose=False):
    img_nodata_mask = (sample['image'][:4] == img_nodata_val).any(axis=0)
        
    # make sure no imagery has nodata vals 
    if  verbose and img_nodata_mask.any(): print('NODATA VAL detected in imagery')
    
    # last three bands are the visual image, separate them
    if len(sample['image']) > 5:
        sample['vis'] = sample['image'][4:7].clone()    
    
    # if there is extra context data to be had
    if len(sample['image']) > 8:
        sample['context'] = sample['image'][8:].clone().long() 
        # assumes contexts is a 4 channel canopy map 
        sample['context'] = torch.nn.functional.one_hot(sample['context']-1, num_classes=4).float()
        sample['context'] = sample['context'].transpose(0,3).squeeze()
        
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
                      data_dir=DATA_DIR,
                      chm_relative_dir="int/lidar/lidar_by_site_32736_10m",
                      sentinel_relative_dir="int/sentinel/sentinel_by_site_32736_10m",
                      canopy_relative_dir = 'int/alos/alos_by_site_20_FNF',
                     ):
    '''
    Returns a dataset with layers in this order: 
        sentinel, then CHM (if requested), then context data (e.g. canopy -- if requested).
    '''
    
    non_img_layers = ['dem','chm', 'canopy']
    img_layers = [l for l in layers if not l in non_img_layers]
    s2_bands = [sentinel_layer_codes[l.lower()] for l in img_layers]
    
    sentinel_data_dir = os.path.join(data_dir,sentinel_relative_dir, site_id)

    # dataset with chm labels
    if "chm" in layers: 
        # gather the sentinel data
        sentinel = Sentinel2(
            sentinel_data_dir,
            bands=s2_bands
        )
        
        # gathers data
        chm_dataset = RasterDataset(paths=os.path.join(data_dir,chm_relative_dir, site_id))
      
        if "canopy" in layers:            
            dataset = IntersectionDataset(sentinel, chm_dataset)            
            canopy = RasterDataset(paths=os.path.join(data_dir, canopy_relative_dir, site_id))
            dataset = IntersectionDataset(ds, canopy, transforms=transforms)
            
        else:    
            dataset = IntersectionDataset(sentinel, chm_dataset, transforms=transforms)
        
    # dataset without labels
    else:
        if "canopy" in layers:  
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
        eval_pad: int = 5,
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
        if stage in ["fit"]:
            self.train_dataset = get_chm_sites(
                sites=self.train_sites, layers=self.layers, transforms = self.train_transforms, **self.kwargs
            )
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
                self.val_dataset, self.original_patch_size, self.eval_stride, units=Units.PIXELS
            )
        if stage in ["test"]:
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
        
        if 'mask' in sample.keys():
            mask = sample["mask"].squeeze(0).cpu().numpy()
        else:
            mask = np.zeros_like(sample["vis"])
        vis = sample["vis"].cpu().numpy() 
        nan_plot_val = 0
        
        if pad > 0:
            vis = vis[:,pad:-pad,pad:-pad]
            mask = mask[pad:-pad,pad:-pad]
                        
        nan_mask = mask == nan_val
        mask[nan_mask] = nan_plot_val
         
        if zero_out_nan_vis:
            for c in range(vis.shape[0]):
                vis[c][nan_mask] = nan_plot_val
        
        
        showing_predictions = "prediction" in sample
        
        ncols = 2
        if showing_predictions:
            ncols += 1
            pred = sample["prediction"].squeeze(0).cpu().numpy()
            pred[nan_mask] = nan_plot_val
            
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

        if suptitle is not None:
            plt.suptitle(suptitle)
        
        return fig
    
    def on_after_batch_transfer(self, batch, dataloader_idx):
        # we normalize in the transforms
        # so override the torchgeo defaults so this function does nothing
        
        return batch