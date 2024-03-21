# imports
from datamodules.chm_datamodule import ChmDataModule, get_default_layers_and_transforms
from experiment_utils import get_site_splits, read_config_file
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from trainers.regression_with_nans import PixelwiseRegressionTask
from utils import get_project_dir
import itertools
import os
import sys
import torch
import yaml

def setup_chm_datamodule(sites_per_split, cfg_data):
    train_sites = sites_per_split['train_sites']
    val_sites = sites_per_split['val_sites']
    test_sites = sites_per_split['test_sites']
    num_image_channels = cfg_data['num_image_channels']
    
    data_layers, batch_transforms = get_default_layers_and_transforms(num_image_channels)

    chm = ChmDataModule(train_sites=train_sites, 
                        val_sites=val_sites, 
                        test_sites=test_sites, 
                        layers=data_layers,
                        batch_transforms=batch_transforms,
                        **cfg_data['datamodule'])
    
    return chm

def train(datamodule, task, base_name, exp_name, version_id_base, num_channels, split_number, version_id, **trainer_kwargs):
    # set up log dirs
    exp_root_dir = f'{get_project_dir()}/{base_name}/{exp_name}/{version_id_base}/{num_channels}_channels/split_{split_number}'
    os.makedirs(f'{exp_root_dir}/logs', exist_ok=True)
    os.makedirs(f'{exp_root_dir}/models', exist_ok=True)
    
    # set up callbacks
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', 
                                          dirpath=f'{exp_root_dir}/models/{version_id}', 
                                          save_top_k=1, 
                                          save_last=True)

    logger = TensorBoardLogger(save_dir=exp_root_dir, name='logs', version=version_id)
    print(f'Logs will go in {exp_root_dir}/logs/{version_id}')

    trainer = Trainer(callbacks=[checkpoint_callback],
                      fast_dev_run=False,
                      log_every_n_steps=1,
                      logger=logger,
                      num_sanity_val_steps=0,
                      **trainer_kwargs)
    print('task', task)
    print('datamodule', datamodule)
    trainer.fit(model=task, datamodule=datamodule)
    
def run_experiment(config_fp, lr=None, weight_decay=None, channels=None):
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
        if channels is not None:
            task['in_channels'] = channels
            cfg['data']['num_image_channels'] = channels
        
        # get the assignment for this split number
        sites_per_split = splits[split_number]
        chm = setup_chm_datamodule(sites_per_split, cfg['data'])
        chm_task = PixelwiseRegressionTask(**task)
        version_id = f'lr_{task["lr"]}_wd_{task["weight_decay"]}'
        train(chm, chm_task, base_name, exp_name, version_id_base, task['in_channels'], split_number, version_id, **cfg['trainer'])
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    
    if len(sys.argv) == 2:
        run_experiment(config_fp=sys.argv[1])
    elif len(sys.argv) == 4:
        run_experiment(config_fp=sys.argv[1], lr=float(sys.argv[2]), weight_decay=float(sys.argv[3]))
    elif len(sys.argv) == 5:
        run_experiment(config_fp=sys.argv[1], lr=float(sys.argv[2]), weight_decay=float(sys.argv[3]), channels=int(sys.argv[4]))
