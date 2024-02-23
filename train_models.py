# imports
import itertools
import os
import sys
import torch
import yaml

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer

from datamodules.chm_datamodule import ChmDataModule, transforms_4_channel_rgbnir_plus_mask_imagestats, transforms_12_channel_plus_mask_imagestats, transforms_12_channel_latllon_plus_mask_imagestats
from experiment_utils import get_site_splits
from trainers.regression_with_nans import PixelwiseRegressionTask
from utils import get_project_dir

def setup_chm_datamodule(sites_per_split, cfg_data):
    train_sites = sites_per_split['train_sites']
    val_sites = sites_per_split['val_sites']
    test_sites = sites_per_split['test_sites']
    num_image_channels = cfg_data['num_image_channels']
    
    if num_image_channels == 4:
        data_layers = ['r', 'g', 'b', 'nir', 'vis', 'chm']
        batch_transforms = transforms_4_channel_rgbnir_plus_mask_imagestats
    elif num_image_channels == 12:
        data_layers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] + ['vis', 'chm']
        batch_transforms = transforms_12_channel_plus_mask_imagestats
    elif num_image_channels == 15:
        data_layers = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12'] + ['vis', 'chm']
        batch_transforms = transforms_12_channel_latllon_plus_mask_imagestats
    else:
        print(f'No directive for {num_image_channels} image channels')

    chm = ChmDataModule(train_sites=train_sites, 
                        val_sites=val_sites, 
                        test_sites=test_sites, 
                        layers=data_layers,
                        batch_transforms=batch_transforms,
                        **cfg_data['datamodule'])
    
    return chm

def read_config_file(config_yaml):
    with open(config_yaml, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg

def train(datamodule, task, base_name, exp_name, version_id=None, **trainer_kwargs):
    # set up log dirs
    exp_root_dir = f'{get_project_dir()}/{base_name}/{exp_name}'
    os.makedirs(f'{exp_root_dir}/logs', exist_ok=True)
    os.makedirs(f'{exp_root_dir}/models', exist_ok=True)
    
    # set up callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          dirpath=f'{exp_root_dir}/models/{version_id}', 
                                          save_top_k=1, 
                                          save_last=True)

    logger = TensorBoardLogger(save_dir=exp_root_dir, name='logs', version=version_id)
    print(f'Logs will go in {exp_root_dir}/logs')

    trainer = Trainer(callbacks=[checkpoint_callback],
                      fast_dev_run=False,
                      log_every_n_steps=1,
                      logger=logger,
                      num_sanity_val_steps=0,
                      **trainer_kwargs)

    trainer.fit(model=task, datamodule=datamodule)
    
def run_experiment(config_fp, lr=None, weight_decay=None):
    cfg = read_config_file(config_fp)

    # get this data split
    split_seed = cfg['data']['split_seed']
    splits = get_site_splits(split_seed)
    splits_to_do = cfg['splits_to_do']
    base_name = cfg['base_name']
    exp_name = cfg['exp_name']
    version_id_base = cfg['version_id_base']

    for split_number in splits_to_do:
        task = cfg['task']
        
        # override training hyperparameters in config file if specified directly
        if lr is not None:
            task['lr'] = lr
        if weight_decay is not None:
            task['weight_decay'] = weight_decay
        
        # get the assignment for this split number
        sites_per_split = splits[split_number]
        chm = setup_chm_datamodule(sites_per_split, cfg['data'])
        chm_task = PixelwiseRegressionTask(**task)
        version_id = f'{version_id_base}_split_{split_number}_seed_{split_seed}_lr_{lr}_wd_{weight_decay}' if lr is not None else f'{version_id_base}_split_{split_number}_seed_{split_seed}'

        train(chm, chm_task, base_name, exp_name, version_id, **cfg['trainer'])
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    if len(sys.argv) == 2:
        run_experiment(config_fp=sys.argv[1])
    else:
        run_experiment(config_fp=sys.argv[1], lr=float(sys.argv[2]), weight_decay=float(sys.argv[3]))


