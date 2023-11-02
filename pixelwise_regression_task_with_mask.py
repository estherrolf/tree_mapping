# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tasks."""

import os
from typing import Any, cast

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection, MeanMetric, Metric
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import unbind_samples
from torchgeo.models import FCN, get_weight
from torchgeo.trainers import utils
import sklearn.metrics
        
class RegressionTaskWithMask(LightningModule):  # type: ignore[misc]
    """LightningModule for training models on regression datasets.

    Supports any available `Timm model
    <https://huggingface.co/docs/timm/index>`_
    as an architecture choice. To see a list of available
    models, you can do:

    .. code-block:: python

        import timm
        print(timm.list_models())
    """

    target_key: str = "label"

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters."""
        # Create model
        weights = self.hyperparams["weights"]
        self.model = timm.create_model(
            self.hyperparams["model"],
            num_classes=self.hyperparams["num_outputs"],
            in_chans=self.hyperparams["in_channels"],
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model = utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hyperparams.get("freeze_backbone", False):
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters."""
        self.config_model()

        self.loss: nn.Module
        if self.hyperparams["loss"] == "mse":
            self.loss = nn.MSELoss()
        elif self.hyperparams["loss"] == "mae":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'mse' or 'mae' loss."
            )
            
        if "context_reg" in self.hyperparams.keys():
            self.context_reg = self.hyperparams["context_reg"]
        else:
            self.context_reg = 0.0
            print(f'context reg is {self.context_reg}')
            
        if "avg_loss_per_pixel" in self.hyperparams.keys() and self.hyperparams["avg_loss_per_pixel"]:
            self.avg_loss_per_pixel = True
        else:
            self.avg_loss_per_pixel = False
            
            
    
        
    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the timm model to use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            num_outputs: Number of prediction outputs
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegressionTask.

        .. versionchanged:: 0.4
            Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(dict[str, Any], self.hparams)
        self.config_task()

        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE":  MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")
        self.pad_predictions=5
        self.has_context_model=False
        
        
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        pad = self.pad_predictions
        batch = args[0]
        x = batch["image"]
        if self.has_context_model:
            c = batch["context"]

        y_ = batch[self.target_key]
        if self.has_context_model:
            if self.model.context_integrator_type in ['weighted_avg','attn']:
                y_hat_, context_ = self(x, c)
            elif self.model.context_integrator_type in ['separated_by_context', 'sigmoid_separated_by_context']:
                (y_hat_, y_hat_by_context_), context_ = self(x, c)
            else:
                print(f"context integrator type {self.model.context_integrator_type} not recognized")
        else:
            y_hat_ = self(x)
        
        if y_hat_.ndim != y_.ndim:
            y_ = y_.unsqueeze(dim=1)
            
        y_hat_ = y_hat_[:,:,pad:-pad,pad:-pad]
        y_ = y_[:,:,pad:-pad,pad:-pad]
        
       
        if self.has_context_model:
            context_ = context_[:,:,pad:-pad,pad:-pad]
            if self.model.context_integrator_type in ['separated_by_context', 'sigmoid_separated_by_context']:
                    y_hat_by_context_ = y_hat_by_context_[:,:,pad:-pad,pad:-pad]
        
        non_nan_mask = y_ >= 0
        y_hat = y_hat_[non_nan_mask]
        y = y_[non_nan_mask].to(torch.float)
        if self.has_context_model:            
            
            if self.model.context_integrator_type in ['weighted_avg','attn']:
                # loss on the context-integreated output y_hat
                loss: Tensor = self.loss(y_hat, y)
                # penalize low entropy
                loss += - self.context_reg * torch.mean(context_ * torch.log(context_))
                
            elif self.model.context_integrator_type in ['separated_by_context']:
                loss: Tensor = 0.0
                # each model respsonible for some part of the distribution, weighted MSE
                # loss  = sum_k [k * MSE(yhat_k, y)]
                for k in range(context_.shape[1]):
                    
                    context_k = context_[:,k].unsqueeze(1)[non_nan_mask]                    
                    y_hat_by_context_k = y_hat_by_context_[:,k].unsqueeze(1)[non_nan_mask]
                    
                    loss += (context_k * (y_hat_by_context_k - y)**2).mean() #/ context_k.mean()
                    
            
                    loss += self.context_reg *(context_k.mean())**2
                
        else:
            # no context model
            loss: Tensor = self.loss(y_hat, y)
            
        # normalize for the number of non nan values
        if self.avg_loss_per_pixel:
            loss = loss * non_nan_mask.sum() / torch.ones_like(non_nan_mask).sum()
            print( non_nan_mask.sum() / torch.ones_like(non_nan_mask).sum())
        self.log("train_loss", loss)  # logging to TensorBoard
        self.train_metrics(y_hat, y.to(torch.float))

        return loss

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        pad = self.pad_predictions
        
        y_ = batch[self.target_key]
        
        if self.has_context_model:
            c = batch["context"]
            
        y_ = batch[self.target_key]
        if self.has_context_model:
            if self.model.context_integrator_type in ['weighted_avg','attn']:
                y_hat_, context_ = self(x, c)
            elif self.model.context_integrator_type in ['separated_by_context']:
                (y_hat_, y_hat_by_context_), context_ = self(x, c)
            else:
                print(f"context integrator type {self.model.context_integrator_type} not recognized")
        else:
            y_hat_ = self(x)
            
        y_hat_ = y_hat_[:,:,pad:-pad,pad:-pad]
        y_ = y_[:,:,pad:-pad,pad:-pad]
                
        if self.has_context_model:
            context_ = context_[:,:,pad:-pad,pad:-pad]
            
        non_nan_mask = y_ >= 0
        y_hat = y_hat_[non_nan_mask]
        y = y_[non_nan_mask]

        loss = self.loss(y_hat, y.to(torch.float))
        if self.avg_loss_per_pixel:
            loss = loss * non_nan_mask.sum() / torch.ones_like(non_nan_mask).sum()
        
        self.log("val_loss", loss)
        self.val_metrics(y_hat, y.to(torch.float))

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:                 
                datamodule = self.trainer.datamodule
                if self.target_key == "mask":
                    y_ = y_.squeeze(dim=1)
                    y_hat_ = y_hat_.squeeze(dim=1)
                batch["prediction"] = y_hat_
                keys = ["image", self.target_key, "prediction"]
                if self.has_context_model:
                    batch["context"] = context_ 
                    keys += ["context"]
                for key in keys:
                    batch[key] = batch[key].cpu()
                for i in range(4):
                    sample = unbind_samples(batch)[i]
 #                   print(f"{batch_idx}-{i}-c-{sample['context'][:,32,32]}")
                    if (sample[self.target_key] > 0).any():
                        fig = datamodule.plot(sample, pad=self.pad_predictions)
                        summary_writer = self.logger.experiment
                        summary_writer.add_figure(
                            f"image/{batch_idx}-{i}", fig, global_step=self.global_step
                        )
                        plt.close()
            except ValueError:
                pass

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        pad = self.pad_predictions
        
        y_ = batch[self.target_key]
        
        if self.has_context_model:
            c = batch["context"]

        y_ = batch[self.target_key]
        if self.has_context_model:
            if self.model.context_integrator_type in ['weighted_avg','attn']:
                y_hat_, context_ = self(x, c)
            elif self.model.context_integrator_type in ['separated_by_context', 'sigmoid_separated_by_context']:
                (y_hat_, y_hat_by_context_), context_ = self(x, c)
            else:
                print(f"context integrator type {self.model.context_integrator_type} not recognized")
        else:
            y_hat_ = self(x)
        
        if y_hat_.ndim != y_.ndim:
            y_ = y_.unsqueeze(dim=1)
            
        y_hat_ = y_hat_[:,:,pad:-pad,pad:-pad]
        y_ = y_[:,:,pad:-pad,pad:-pad]
        if self.has_context_model:
            context = context_[:,:,pad:-pad,pad:-pad]
            
        non_nan_mask = y_ != -9999.
        y_hat = y_hat_[non_nan_mask]
        y = y_[non_nan_mask]
        
        loss = self.loss(y_hat, y.to(torch.float))
        
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y.to(torch.float))

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader
        Returns:
            predicted values
        """
        batch = args[0]
        x = batch["image"]
        y_hat: Tensor = self(x)
        return y_hat

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hyperparams["learning_rate"], weight_decay=self.hyperparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }


class PixelwiseRegressionTask(RegressionTaskWithMask):
    """LightningModule for pixelwise regression of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM backbones <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.

    .. versionadded:: 0.5
    """

    target_key: str = "mask"

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters."""
        weights = self.hyperparams["weights"]

        if self.hyperparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=1,
            )
        elif self.hyperparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=1,
            )
        elif self.hyperparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hyperparams["in_channels"],
                classes=1,
                num_filters=self.hyperparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["model"] != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hyperparams.get("freeze_backbone", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hyperparams.get("freeze_decoder", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
