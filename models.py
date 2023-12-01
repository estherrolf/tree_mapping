from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import Module

from pixelwise_regression_task_with_mask import RegressionTaskWithMask
import torchgeo.models
from typing import Any, cast


class TiledRCF(Module):
    """This model extracts random convolutional features (RCFs) from its input.

    RCFs are used in Multi-task Observation using Satellite Imagery & Kitchen Sinks
    (MOSAIKS) method proposed in https://www.nature.com/articles/s41467-021-24638-z.

    .. note::

        This Module is *not* trainable. It is only used as a feature extractor.
    """

    weights: Tensor
    biases: Tensor

    def __init__(
        self,
        in_channels: int = 4,
        features: int = 16,
        kernel_size: int = 3,
        bias: float = -1.0,
        seed: Optional[int] = None,
        pool_kernel_size: Optional[int] = None,
    ) -> None:
        """Initializes the RCF model.

        This is a static model that serves to extract fixed length feature vectors from
        input patches.

        .. versionadded:: 0.2
           The *seed* parameter.

        Args:
            in_channels: number of input channels
            features: number of features to compute, must be divisible by 2
            kernel_size: size of the kernel used to compute the RCFs
            bias: bias of the convolutional layer
            seed: random seed used to initialize the convolutional layer
        """
        super().__init__()

        assert features % 2 == 0

        if seed is None:
            generator = None
        else:
            generator = torch.Generator().manual_seed(seed)

        # We register the weight and bias tensors as "buffers". This does two things:
        # makes them behave correctly when we call .to(...) on the module, and makes
        # them explicitely _not_ Parameters of the model (which might get updated) if
        # a user tries to train with this model.
        self.register_buffer(
            "weights",
            torch.randn(
                features // 2,
                in_channels,
                kernel_size,
                kernel_size,
                requires_grad=True,
                generator=generator,
            ),
        )
        self.register_buffer(
            "biases", torch.zeros(features // 2, requires_grad=False) + bias
        )
        
        self.rcf_kernel_size = kernel_size
        self.pool_kernel_size = pool_kernel_size


    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the RCF model.

        Args:
            x: a tensor with shape (B, C, H, W)

        Returns:
            a tensor of size (B, ``self.num_features``)
        """
      #  if self.pool_kernel_size is not None: padding=int((self.pool_kernel_size - 1) / 2)
      #  else: padding = 1
        pad_rcf = int((self.rcf_kernel_size-1)/2)
        x1a = F.relu(
            F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=pad_rcf),
            inplace=True,
        )
        x1b = F.relu(
            -F.conv2d(x, self.weights, bias=self.biases, stride=1, padding=pad_rcf),
            inplace=False,
        )
        

        if self.pool_kernel_size is not None:
            pad_agg = int((self.pool_kernel_size - 1) / 2)
            x1a = F.avg_pool2d(x1a, kernel_size=self.pool_kernel_size, stride=1, padding=pad_agg,count_include_pad=False).squeeze()
            x1b = F.avg_pool2d(x1b, kernel_size=self.pool_kernel_size, stride=1, padding=pad_agg,count_include_pad=False).squeeze()
                
        else:
            x1a = F.adaptive_avg_pool2d(x1a, (1, 1)).squeeze()
            x1b = F.adaptive_avg_pool2d(x1b, (1, 1)).squeeze()
                          
        if len(x1a.shape) == 1 or len(x1a.shape) == 3:  # case where we passed a single input
            output = torch.cat((x1a, x1b), dim=0)
            return output
        else:  # case where we passed a batch of > 1 inputs
            output = torch.cat((x1a, x1b), dim=1)
            return output
 

class FixedElevModel(nn.Module):
    def __init__(self,
                 num_dim=2,
                 in_channels=1,
                ) -> None:
        super().__init__()
        self.num_dim = num_dim
        
        dem_stats = {
            'mean': 231.69882, 'std': 142.62463
        }
        
        self.cutoff = (180. - dem_stats['mean']) / dem_stats['std']

    def forward(self, x: Tensor) -> Tensor:
        # don't do gradient on the RCF parr
      #      
    
        with torch.no_grad():
            x1 = (x > self.cutoff).float()
            x2 = (x <= self.cutoff).float()
            x = torch.hstack((x1,x2))
            return x
    

       

class FCN(nn.Module):
    """A simple 5 layer FCN with leaky relus and 'same' padding."""

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
        conv4 = nn.modules.Conv2d(
            num_filters, num_filters, kernel_size=3, stride=1, padding=1
        )
        conv5 = nn.modules.Conv2d(
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
                conv4,
                nn.BatchNorm2d(num_filters, affine=True),
                nn.modules.LeakyReLU(inplace=True),
                conv5,
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
                conv4,
                nn.modules.LeakyReLU(inplace=True),
                conv5,
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
    
class TinyFCN(nn.Module):
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
    
