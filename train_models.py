# imports
from datamodules.chm_datamodule import ChmDataModule, get_default_layers_and_transforms
from experiment_utils import get_site_splits, read_config_file
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
from trainers.regression_with_nans import PixelwiseRegressionTask
from utils import get_project_dir, str_to_bool
import argparse
import itertools
import numpy as np
import os
import random
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

def train(datamodule, task, base_name, exp_name, version_id_base, split_number, version_id, num_channels, layers_tuned, freeze_backbone, num_train_sites, seed, **trainer_kwargs):
    # set up log dirs
    if num_train_sites is not None:
        base_name += '/subsetted_train_sites'
        end = f'{num_train_sites}_train_sites/seed_{seed}/split_{split_number}'
    else:
        end = f'split_{split_number}'

    if num_channels is not None:
        exp_root_dir = f'{get_project_dir()}/{base_name}/{exp_name}/{version_id_base}/{num_channels}_channels/{end}'
    elif layers_tuned is not None:
        exp_root_dir = f'{get_project_dir()}/{base_name}/{exp_name}/{version_id_base}/{layers_tuned}_layers_tuned/{end}'
    elif freeze_backbone is not None:
        exp_root_dir = f'{get_project_dir()}/{base_name}/{exp_name}/{version_id_base}/freeze_backbone_{freeze_backbone}/{end}'
    else:
        exp_root_dir = f'{get_project_dir()}/{base_name}/{exp_name}/{version_id_base}/{end}'

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

def get_train_sites(split_seed, split, train_sample_seed, num_train_sites):
    original_train_sites = get_site_splits(random_seed=split_seed)[split]['train_sites']
    random_state_train_sample = np.random.RandomState(train_sample_seed)
    train_site_sample = random_state_train_sample.choice(len(original_train_sites), num_train_sites, replace=False)
    sampled_train_sites = [original_train_sites[x] for x in train_site_sample]

    return sampled_train_sites

def run_experiment(config_fp, lr, weight_decay, channels, layers_tuned, freeze_backbone, train_site_count, split, train_sample_seed):
    cfg = read_config_file(config_fp)

    # get this data split
    split_seed = cfg['data']['split_seed']
    splits = get_site_splits(split_seed)
    splits_to_do = cfg['splits_to_do'] if split is None else [split]
    base_name = cfg['base_name']
    exp_name = cfg['exp_name']
    version_id_base = cfg['version_id_base']

    for split_number in splits_to_do:
        task = cfg['task']
        sites_per_split = splits[split_number]
        
        # override training hyperparameters in config file if specified directly
        if lr is not None:
            task['lr'] = lr
        if weight_decay is not None:
            task['weight_decay'] = weight_decay
        if channels is not None:
            task['in_channels'] = channels
            cfg['data']['num_image_channels'] = channels
        if layers_tuned is not None:
            task['num_layers_to_unfreeze'] = layers_tuned
        if freeze_backbone is not None:
            task['freeze_backbone'] = freeze_backbone
        if train_sample_seed is not None:
            sites_per_split['train_sites'] = get_train_sites(split_seed=split_seed, split=split_number, train_sample_seed=train_sample_seed, num_train_sites=train_site_count)
        
        chm = setup_chm_datamodule(sites_per_split, cfg['data'])
        chm_task = PixelwiseRegressionTask(**task)
        version_id = f'lr_{task["lr"]}_wd_{task["weight_decay"]}'

        train(chm, chm_task, base_name, exp_name, version_id_base, split_number, version_id, channels, layers_tuned, freeze_backbone, train_site_count, train_sample_seed, **cfg['trainer'])
    
if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_fp', type=str, required=True)
    parser.add_argument('--lr', type=float, required=False, default=None)
    parser.add_argument('--weight_decay', type=float, required=False, default=None)
    parser.add_argument('--channels', type=int, required=False, default=None)
    parser.add_argument('--layers_tuned', type=int, required=False, default=None)
    parser.add_argument('--freeze_backbone', type=str_to_bool, required=False, default=None)
    parser.add_argument('--train_site_count', type=int, required=False, default=None)
    parser.add_argument('--split', type=int, required=False, default=None)
    parser.add_argument('--seed', type=int, required=False, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    run_experiment(config_fp=args.config_fp,
                   lr=args.lr,
                   weight_decay=args.weight_decay,
                   channels=args.channels,
                   layers_tuned=args.layers_tuned,
                   freeze_backbone=args.freeze_backbone,
                   train_site_count=args.train_site_count,
                   split=args.split,
                   train_sample_seed=args.seed)
