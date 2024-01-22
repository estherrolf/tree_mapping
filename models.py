from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

from pixelwise_regression_task_with_mask import RegressionTaskWithMask
import torchgeo.models
from typing import Any, cast


class FCN_3_layers(nn.Module):
    """A simple 3 layer FCN with leaky relus and 'same' padding."""

    def __init__(self, in_channels: int, classes: int, num_filters: int = 64, batchnorm: bool=False) -> None:
        """Initializes the 5 layer FCN model.

        Args:
            in_channels: Number of input channels that the model will expect
            classes: Number of filters in the final layer
            num_filters: Number of filters in each convolutional layer
        """
        super().__init__()
        self.batchnorm = batchnorm
        
        conv1 = nn.modules.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv2 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv3 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )

        if self.batchnorm:
            self.backbone = nn.modules.Sequential(
                conv1,
                nn.BatchNorm2d(num_filters, affine=True),
                nn.modules.LeakyReLU(inplace=True), 
                conv2,
                nn.BatchNorm2d(num_filters, affine=True),
                nn.modules.LeakyReLU(inplace=True),
                conv3,
                nn.BatchNorm2d(num_filters, affine=True),
                nn.modules.LeakyReLU(inplace=True),
            )
        else:
            self.backbone = nn.modules.Sequential(
                conv1,
                nn.modules.LeakyReLU(inplace=True),
                conv2,
                nn.modules.LeakyReLU(inplace=True),
                conv3,
                nn.modules.LeakyReLU(inplace=True),

            )

        self.last = nn.modules.Conv2d(
            num_filters, classes, kernel_size=1, stride=1, padding=0
        )


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        x = self.backbone(x)
        x = self.last(x)
        return x
    
