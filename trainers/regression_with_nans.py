# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Trainers for regression. Modified from https://github.com/microsoft/torchgeo/blob/main/torchgeo/trainers/regression.py. """

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# imports
import csv
import lightning
import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import sys
import timm
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from collections.abc import Sequence
from lightning.pytorch import LightningModule
from typing import Any, Optional, Union
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchvision.models._api import WeightsEnum
from torchgeo.datasets import unbind_samples
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils, BaseTask
from torchgeo.models import ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights

# so that we can use the pretrained models from this repo
sys.path.append('../global-canopy-height-model')
from gchm.models.xception_sentinel2 import xceptionS2_08blocks_256


class BaseTask(LightningModule, ABC):
    """Abstract base class for all TorchGeo trainers.

    .. versionadded:: 0.5
    """

    #: Model to train.
    model: Any

    #: Performance metric to monitor in learning rate scheduler and callbacks.
    monitor = "val_loss"

    #: Whether the goal is to minimize or maximize the performance metric to monitor.
    mode = "min"

    def __init__(self, ignore: Optional[Union[Sequence[str], str]] = None) -> None:
        """Initialize a new BaseTask instance.

        Args:
            ignore: Arguments to skip when saving hyperparameters.
        """
        super().__init__()
        self.save_hyperparameters(ignore=ignore)
        self.configure_losses()
        self.configure_metrics()
        self.configure_models()

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""

    @abstractmethod
    def configure_models(self) -> None:
        """Initialize the model."""

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)
    


class RegressionTask(BaseTask):
    """Regression."""

    target_key = "label"

    def __init__(
        self,
        model: str = "resnet50",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_outputs: int = 1,
        num_filters: int = 3,
        loss: str = "mse",
        lr: float = 1e-3,
        weight_decay: float = 1e-2, 
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        num_layers_to_unfreeze: int=0,
        nan_val_mask: Optional[int] = None, 
        pad_pixels: Optional[int] = 0
    ) -> None:
        """Initialize a new RegressionTask instance.

        Args:
            model: Name of the
                `timm <https://huggingface.co/docs/timm/reference/models>`__ or
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the
                `timm <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or
                `smp <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone
                to use. Only applicable to PixelwiseRegressionTask.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_outputs: Number of prediction outputs.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: One of 'mse' or 'mae'.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegressionTask.
            num_layers_to_unfreeze: Number of layers to unfreeze, used only for xception model.

        .. versionchanged:: 0.4
           Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.
        """
        self.weights = weights
        
        # new things:
        self.nan_val_mask = nan_val_mask
        self.pad_pixels = pad_pixels
        self.num_batch_points = []
        self.val_loss_batches = []
        self.val_loss_epochs = []
        
        super().__init__(ignore="weights")

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        if loss == "mse":
            self.criterion: nn.Module = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'mse' or 'mae' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_models(self) -> None:
        """Initialize the model."""
        # Create model
        weights = self.weights
        self.model = timm.create_model(
            self.hparams["model"],
            num_classes=self.hparams["num_outputs"],
            in_chans=self.hparams["in_channels"],
            pretrained=weights is True,
        )
        # self.model = nn.DataParallel(self.model)
        # Load weights
        print('weights')
        if weights and weights is not True:
            print('weights')
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model = utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True
                
    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams["patience"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch["image"]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y_ = batch[self.target_key].to(torch.float)
        y_hat_ = self(x)
        if y_hat_.ndim != y_.ndim:
            y_ = y_.unsqueeze(dim=1)
            
        # pad e.g. for receptive field
        if self.pad_pixels > 0:
            y_hat_ = y_hat_[:,:,self.pad_pixels:-self.pad_pixels,self.pad_pixels:-self.pad_pixels]
            y_ = y_[:,:,self.pad_pixels:-self.pad_pixels,self.pad_pixels:-self.pad_pixels]       
        
        # ignore pixels where mask data is nan
        if self.nan_val_mask is not None:
            non_nan_mask = y_ != self.nan_val_mask
            y_hat = y_hat_[non_nan_mask]
            y = y_[non_nan_mask].to(torch.float)
        else:
            y_hat = y_hat_
            y = y_
            
        # if nothing in the batch is not nan:
        if len(y) == 0: return
            
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y_ = batch[self.target_key].to(torch.float)
        y_hat_ = self(x)
        
        if y_hat_.ndim != y_.ndim:
            y_ = y_.unsqueeze(dim=1)
            
        # pad e.g. for receptive field
        if self.pad_pixels > 0:
            y_hat_ = y_hat_[:,:,self.pad_pixels:-self.pad_pixels,self.pad_pixels:-self.pad_pixels]
            y_ = y_[:,:,self.pad_pixels:-self.pad_pixels,self.pad_pixels:-self.pad_pixels]    
        
        # ignore pixels where mask data is nan
        if self.nan_val_mask is not None:
            non_nan_mask = y_ != self.nan_val_mask
            #non_nan_mask = y_ >= 0
            y_hat = y_hat_[non_nan_mask]
            y = y_[non_nan_mask].to(torch.float)
        else:
            y_hat = y_hat_
            y = y_
        
        # if nothing in the batch is not nan:
        if len(y) == 0: return

        loss = self.criterion(y_hat, y)

        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics)
        self.num_batch_points += [len(y)]
        self.val_loss_batches += [loss.cpu().item()]

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            datamodule = self.trainer.datamodule
            if self.target_key == "mask":
                y_ = y_.squeeze(dim=1)
                y_hat_ = y_hat_.squeeze(dim=1)
            batch["prediction"] = y_hat_
            keys = ["image", self.target_key, "prediction"]
            for key in keys:
                batch[key] = batch[key].cpu()
            for i in range(4):
                samples_this_batch = unbind_samples(batch)
                if i < len(samples_this_batch):
                    sample = samples_this_batch[i]                    
                    if (sample[self.target_key] >= 0).any():
                        fig = datamodule.plot(sample, pad=self.pad_pixels)
                        summary_writer = self.logger.experiment
                        summary_writer.add_figure(
                            f"image/{batch_idx}-{i}", fig, global_step=self.global_step
                        )
                        plt.close()

    def on_validation_epoch_end(self):
        mean_loss_epoch = np.sum([self.num_batch_points[batch] * self.val_loss_batches[batch] for batch in range(len(self.num_batch_points))]) / np.sum(self.num_batch_points)
        
        self.val_loss_epochs += [mean_loss_epoch] # adds the current epoch's mean loss to the list of epoch losses
        self.val_loss_batches = []
        self.num_batch_points = [] 
        np.save(f'{self.logger.log_dir}/val_loss', self.val_loss_epochs) # saves the list of val losses

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y_ = batch[self.target_key].to(torch.float)
        y_hat_ = self(x)
        if y_hat_.ndim != y_.ndim:
            y_ = y_.unsqueeze(dim=1)
            
        # pad e.g. for receptive field
        if self.pad_pixels > 0:
            y_hat_ = y_hat_[:,:,self.pad_pixels:-self.pad_pixels,self.pad_pixels:-self.pad_pixels]
            y_ = y_[:,:,self.pad_pixels:-self.pad_pixels,self.pad_pixels:-self.pad_pixels]     
        
        # ignore pixels where mask data is nan
        if self.nan_val_mask is not None:
            non_nan_mask = y_ != self.nan_val_mask
            y_hat = y_hat_[non_nan_mask]
            y = y_[non_nan_mask].to(torch.float)
        else:
            y_hat = y_hat_
            y = y_
            
        # if nothing in the batch is not nan:
        if len(y) == 0: return
            
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted regression values.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        y_hat: Tensor = self(x)
        return y_hat


class PixelwiseRegressionTask(RegressionTask):
    """LightningModule for pixelwise regression of images.

    .. versionadded:: 0.5
    """

    target_key = "mask"

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights
        # print('hyperparameters:', self.hparams)

        if self.hparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["backbone"],
                in_channels=self.hparams["in_channels"],
                classes=1,
            )
            if weights is not None:
                print(f'loading encoder weights from {weights}')
                
                # which backbone
                if self.hparams["backbone"] == "resnet18":
                    pretrained_weights = "ResNet18_Weights"
                elif self.hparams["backbone"] == "resnet50":
                    pretrained_weights = "ResNet50_Weights"

                # which weights
                if weights == 'SENTINEL2_ALL_MOCO':
                    pretrained_weights += ".SENTINEL2_ALL_MOCO"
                elif weights == 'SENTINEL2_RGB_MOCO':
                    pretrained_weights += ".SENTINEL2_RGB_MOCO"
                
#                 self.model.encoder.load_state_dict(encoder_weights.get_state_dict(progress=True), strict=False)
                weights = pretrained_weights
            
        elif self.hparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hparams["in_channels"],
                classes=1,
            )
        elif self.hparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hparams["in_channels"],
                classes=1,
                num_filters=self.hparams["num_filters"],
            )
        elif self.hparams["model"] == 'xceptionS2_08blocks_256':
            self.model = xceptionS2_08blocks_256(in_channels=self.hparams["in_channels"], 
                                                 out_channels=1, 
                                                 model_weights=weights,
                                                 returns="targets")
            
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+', 'xceptionS2_08blocks_256', and 'fcn'."
            )

        if self.hparams["model"] not in ["fcn", "xceptionS2_08blocks_256"]:
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams["freeze_backbone"] and self.hparams["model"] in [
            "unet",
            "deeplabv3+",
        ]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
                
        elif self.hparams["model"] in ['xceptionS2_08blocks_256']:
            # default to everything frozen
            for param in self.model.parameters():
                param.requires_grad = False

            if self.hparams["num_layers_to_unfreeze"] == 1:
                # just the linear layer
                parts_to_unfreeze = [self.model.predictions]
            elif self.hparams["num_layers_to_unfreeze"] == 2:
                # linear and second to last layer
                parts_to_unfreeze = [self.model.predictions, self.model.sepconv_blocks[-1]]
            elif self.hparams["num_layers_to_unfreeze"] == 3:
                # linear and last two layers
                parts_to_unfreeze = [self.model.predictions, self.model.sepconv_blocks[-1],self.model.sepconv_blocks[-2]]
            elif self.hparams["num_layers_to_unfreeze"] == 9:
                # 9 all layers
                parts_to_unfreeze = [self.model]
            else:
                print(f'asked to unfreeze {self.hparams["num_layers_to_unfreeze"]} layers, can only handle 1, 2, 3, or 9')
            #unfreeze the last parameters
            for model_part in parts_to_unfreeze:
                for param in model_part.parameters():
                    param.requires_grad = True
                

        # Freeze decoder
        if self.hparams["freeze_decoder"] and self.hparams["model"] in [
            "unet",
            "deeplabv3+",
        ]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False