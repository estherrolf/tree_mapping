import argparse
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import os
import torch
import yaml

from datamodules.chm_datamodule import ChmDataModule, transforms_4_channel_rgbnir_plus_mask_imagestats
from experiment_utils import get_site_splits
from trainers.regression_with_nans import PixelwiseRegressionTask

def read_config_file(config_yaml):
    with open(config_yaml, "r") as cfg_file:
        cfg = yaml.safe_load(cfg_file)
    return cfg

def setup_chm_datamodule(sites_per_split, cfg_data):
    
    train_sites = sites_per_split["train_sites"]
    val_sites = sites_per_split["val_sites"]
    test_sites = sites_per_split["test_sites"]
    
    num_image_channels = cfg_data["num_image_channels"]
    
    if num_image_channels == 4:
        data_layers = ['r','g','b','nir','vis','chm']
        batch_transforms = transforms_4_channel_rgbnir_plus_mask_imagestats
    else:
        print('no directive for {num_image_channels} image channels')
    

    chm = ChmDataModule(
        train_sites=train_sites, 
        val_sites=val_sites, 
        test_sites=test_sites, 
        layers=data_layers,
        batch_transforms = batch_transforms,
        **cfg_data['datamodule']
    )
    
    return chm

def setup_chm_task(cfg_task):
    
    task = PixelwiseRegressionTask(**cfg_task)
    
    return task

def setup_log_dirs(base_name, exp_name):
    exp_root_dir = os.path.join(base_name,exp_name)
    for path in [base_name, exp_root_dir, f"{exp_root_dir}/logs", f"{exp_root_dir}/models"]:
        if not os.path.exists(path): os.mkdir(path)

def train(datamodule, task, base_name, exp_name, version_id=None, **trainer_kwargs):

    setup_log_dirs(base_name, exp_name)
    
    # Set up callbacks
    exp_root_dir = os.path.join(base_name, exp_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", 
        dirpath=os.path.join(exp_root_dir, 'models', version_id), 
        save_top_k=1, 
        save_last=True
    )

    logger = TensorBoardLogger(save_dir=exp_root_dir, name="logs", version=version_id)
    print(f"logs will go in {exp_root_dir}/logs")

    trainer = Trainer(
                    callbacks=[checkpoint_callback],
                    fast_dev_run=False,
                    log_every_n_steps=1,
                    logger=logger,
                    **trainer_kwargs
    )

    trainer.fit(model=task, datamodule=datamodule)
    
def run_experiment(config_fp):
    cfg = read_config_file(config_fp)

    # get this data split
    split_seed = cfg['data']['split_seed']
    splits = get_site_splits(split_seed)
    splits_to_do = cfg['splits_to_do']

    base_name = cfg['base_name']
    exp_name = cfg['exp_name']
    version_id_base = cfg['version_id_base']

    for split_number in splits_to_do:
        # get the assignment for this split number
        sites_per_split = splits[split_number]

        chm = setup_chm_datamodule(sites_per_split,
                                   cfg['data'])

        task = setup_chm_task(cfg['task'])

        version_id = f'{version_id_base}_split_{split_number}_seed_{split_seed}'
        train(chm, task, base_name, exp_name, version_id, **cfg['trainer'])


    import argparse

    
if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config_fp")
    args = parser.parse_args()
    config_fp = args.config_fp
    #config_fp = 'experiment_configs/train_baseline_local_models.yaml'
    run_experiment(config_fp)