#from osgeo import gdal
from experiment_utils import get_site_splits
from train_models import setup_chm_datamodule

import numpy as np
import torchgeo
from torchgeo.samplers.constants import Units
from torch.utils.data import DataLoader
from torchgeo.samplers import GridGeoSampler
from torchgeo.datasets import stack_samples

def make_tabular_dataset(data_config, split_number, return_test=False, patch_size=32):
    chm = chm_from_config(data_config, split_number)
    chm.setup('fit')
        
    x_val, y_val = datset_to_xy(chm.val_dataset, patch_size=32)
    x_train, y_train= datset_to_xy(chm.train_dataset, patch_size=32)
    
    if return_test:
        x_test, y_test = datset_to_xy(chm.test_dataset, patch_size=32)
        return x_train, y_train, x_val, y_val, x_test, y_test
    
    return x_train, y_train, x_val, y_val

def chm_from_config(data_config, split_number): 
    # get this data split  
    split_seed = data_config['split_seed']
    splits = get_site_splits(split_seed)
    sites_per_split = splits[split_number]
    
    # make the data module
    chm = setup_chm_datamodule(sites_per_split, data_config)
    
    return chm

def datset_to_xy(dataset, patch_size=32):
    # going lower than 16 takes a long time, but might include more pixels at the boundary of the tiles

    sampler = GridGeoSampler(
                dataset, patch_size,patch_size, units=Units.PIXELS
            )
    
    dataloader = DataLoader(dataset=dataset, 
                              sampler=sampler,
                             collate_fn=stack_samples)

    return dataloader_to_static_dataset(dataloader)

def dataloader_to_static_dataset(dataloader,
                                 mask_nodata=-9999.,
                                 pixel_sample_rate = 1.0,
                                 random_seed = 0):

    rs = np.random.RandomState(random_seed)
    
    x_train = []
    y_train = []
    
    for batch in dataloader:
        images = batch["image"].numpy()
        mask = batch["mask"].numpy()
        # sample from or shuffle the pixels that are not nodata
        for in_batch_idx in range(len(images)):
            not_nans = np.where(mask[in_batch_idx][0] != mask_nodata) 
            # subsample or shuffle the idxs in each patch
            num_samples_this_chip = int(np.floor(pixel_sample_rate * len(not_nans[0])))
            idxs = rs.choice(len(not_nans[0]), num_samples_this_chip, replace=False)
            not_nans_sampled = (not_nans[0][idxs], not_nans[1][idxs])

            x_train.append(images[in_batch_idx,:,not_nans_sampled[0], not_nans_sampled[1]])
            y_train.append(mask[in_batch_idx,0,not_nans_sampled[0], not_nans_sampled[1]])
            
    x_train_all = np.vstack(x_train)
    y_train_all = np.hstack(y_train)
    
    return x_train_all, y_train_all