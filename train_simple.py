import itertools
import subprocess
from multiprocessing import Process, Queue

import os
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import torch
import utils
from chm_datamodule import ChmDataModule, my_transforms_4_channel_rgbnir_plus_mask
import pixelwise_regression_task_with_mask

GPUS = [0] 
TEST_MODE = False

train_sites_all = [
    ['KaringaniSite17'],
    ['KaringaniSouthSouthDevNode'],
    ['KaringaniSungoloDevNode'],
    ['KaringaniSite08'],
    ['KaringaniVultureSite01'],
    ['KaringaniSite06'],
    ['KaringaniSite15'],
    ['KaringaniSite13'],
    ['KaringaniSite11DevNodeF'],
    ['KaringaniSite12'],
    ['KaringaniSite16']
]


val_sites = ["KaringaniSite14"]
test_sites = ["KaringaniSite11DevNodeF"]#,"KaringaniSite12","KaringaniSite16"]

batch_size = 32
length= 10000

num_filter_options = [128]
lr_options = [
    0.001,
   # 0.01,
   # 0.0001
]

num_workers = 32
max_epochs = 25
fast_dev_run = False    

exp_name = 'one_site_training_all'

for path in [exp_name, f"{exp_name}/logs", f"{exp_name}/models"]:
    if not os.path.exists(path): os.mkdir(path)

def main():
    
    for (num_filters,lr) in itertools.product(num_filter_options, lr_options):
        for train_sites in train_sites_all:
            #train_sites = [train_site_this]
            print('train sites ', train_sites, '-- params: ',  num_filters, lr)
            chm = ChmDataModule(train_sites=train_sites, 
                                val_sites=val_sites,
                                test_sites=test_sites, 
                                batch_size=batch_size, 
                                length=length,
                                batch_transforms = my_transforms_4_channel_rgbnir_plus_mask
                               )

            task = pixelwise_regression_task_with_mask.PixelwiseRegressionTask(model='fcn', 
                                    loss='mse',
                                    learning_rate=lr,
                                    learning_rate_schedule_patience=10,
                                    weights=None,
                                    in_channels=4,
                                    num_classes=1, 
                                    num_filters=num_filters,
                            )

            # Set up callbacks
            accelerator = "gpu" if torch.cuda.is_available() else "cpu"
            default_root_dir = os.path.join(exp_name)
            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss", dirpath=default_root_dir, save_top_k=1, save_last=True
            )

            # save name
            version_id = f"filters_{num_filters}_lr_{lr}_train_sites" 
            for x in train_sites:
                version_id += f"_{x}" 

            logger = TensorBoardLogger(
                save_dir=default_root_dir, name="logs", version=version_id)
            print(f"logs will go in {default_root_dir}/logs")


            trainer = Trainer(
                accelerator=accelerator,
                callbacks=[checkpoint_callback, 
             #              early_stopping_callback
                          ],
                fast_dev_run=fast_dev_run,
                log_every_n_steps=1,
                logger=logger,
                min_epochs=1,
                max_epochs=max_epochs,
            )

            trainer.fit(model=task, datamodule=chm)

            model_save_path = f'{exp_name}/models/{version_id}.fcn'
            torch.save(task.model.state_dict(), model_save_path)

        return

if __name__ == "__main__":
    main()