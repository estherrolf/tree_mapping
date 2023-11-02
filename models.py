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
        
class RCFContextModel(nn.Module):
    def __init__(self,
                 num_dim=2,
                 in_channels=4,
                 num_rcf_features=32,
                 rcf_kernel_size=5,
                 pool_kernel_size=None,
                ) -> None:
        super().__init__()
        self.num_dim = num_dim
        self.num_rcf_features = num_rcf_features
        self.per_pixel_context_layers = (pool_kernel_size is not None)
    
        self.rcf = TiledRCF(features = self.num_rcf_features, 
                                           in_channels=in_channels, 
                                           kernel_size=rcf_kernel_size,
                                               pool_kernel_size=pool_kernel_size)
        if pool_kernel_size is None:
            self.last = nn.modules.Linear(self.num_rcf_features, self.num_dim)
        else:
            self.last = nn.modules.Conv2d(in_channels=self.num_rcf_features, out_channels=self.num_dim, 
                                          kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
 
        device = x.device
            
        singleton_batch = False
        if x.shape[0] == 1: singleton_batch=True
        # h,w for expanding back to image size
        h,w = x.shape[-2], x.shape[-1]
        # don't do gradient on the RCF part        
        with torch.no_grad():  
            x = self.rcf(x)
            
        # yes gradient wrt the linear
        x = self.last(x)
            
        if singleton_batch: x = x.unsqueeze(0)
            
        if not self.per_pixel_context_layers:
            # expand back out
            x = x.unsqueeze(2).unsqueeze(3).expand(-1,-1,h,w)
        return x
    
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
    
class FixedCanopyModel(nn.Module):
    def __init__(self,
                 num_dim=2,
                 in_channels=4,
                ) -> None:
        super().__init__()
        
        # dim 2 forest dim 2 non-forest/water
        self.linear = nn.Conv2d(4,2,1,1,0,bias=False)        
    #    weights = torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.]]).unsqueeze(2).unsqueeze(3)
        weights = torch.tensor([[0.8,0.8,0.2,0.2],[0.2,0.2,0.8,0.8]]).unsqueeze(2).unsqueeze(3)
        self.linear.weight = torch.nn.parameter.Parameter(weights, requires_grad=True)
     #   self.linear.weight = torch.nn.parameter.Parameter(weights, requires_grad=True)
        
        self.sm = torch.nn.Softmax(dim=1)
        
    def forward(self, c: Tensor) -> Tensor:
        x = self.linear(c)
        x = self.sm(x)
        return x       
       

class FCN(nn.Module):
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
    
        
def get_context_model(context_model_type, num_contexts, input_channels=1, layer_size=4):
    if context_model_type == 'avg':
        return nn.modules.Sequential(nn.modules.AvgPool2d(kernel_size=31,
                                                          stride=1,
                                                          padding=15,
                                                          count_include_pad=False),
                                     nn.modules.Conv2d(
                                         input_channels, num_contexts, kernel_size=1, stride=1, padding=0
                                     ),
                                     torch.nn.Softmax(dim=1),
                                    )
    elif context_model_type == 'avg-':
        return nn.modules.Sequential(nn.modules.AvgPool2d(kernel_size=31,
                                                          stride=1,
                                                          padding=15,
                                                          count_include_pad=False),
                                     nn.modules.Conv2d(
                                         input_channels, 1, kernel_size=1, stride=1, padding=0
                                     ),
                                     torch.nn.Sigmoid(),
                                    )
    
    elif context_model_type == 'avg+':
        return nn.modules.Sequential(nn.modules.AvgPool2d(kernel_size=15,
                                                          stride=1,
                                                          padding=7,
                                                          count_include_pad=False),
                                     nn.modules.Conv2d(
                                         input_channels, layer_size, kernel_size=1, stride=1, padding=0
                                     ),
                                     nn.modules.ReLU(inplace=True),
                                     nn.modules.Conv2d(
                                         layer_size, num_contexts, kernel_size=1, stride=1, padding=0
                                     ),
                                     torch.nn.Softmax(dim=1),
                                    )
    
    
    elif context_model_type == 'rcf':
        return nn.modules.Sequential(RCFContextModel(num_dim=num_contexts,
                                                     in_channels=input_channels,
                                                     num_rcf_features=64,
                                                     rcf_kernel_size=3,
                                                     pool_kernel_size=11,
                                                    ),
                                     # nn.modules.Conv2d(
                                     #     layer_size, num_contexts, kernel_size=1, stride=1, padding=0
                                     # ),
                                     torch.nn.Softmax(dim=1),
                                    )
    
    elif context_model_type == 'fixed_elev':
        return nn.modules.Sequential(FixedElevModel(),
                                     nn.modules.Conv2d(
                                         2, num_contexts, kernel_size=1, stride=1, padding=0
                                     ),
                                     torch.nn.Softmax(dim=1),)
    
    elif context_model_type == 'fixed_canopy':
        return nn.modules.Sequential(FixedCanopyModel(),
                                    )

def get_context_integrator(context_integrator_type='weighted_avg'):
    
    if context_integrator_type=='weighted_avg':
        def integrator(x_by_context, context):
            
            return (x_by_context * context).sum(dim=1).unsqueeze(dim=1)
    
    elif context_integrator_type=='attn':
        
        def integrator(x_by_context, context):
            tau = 1.0
            weights = F.softmax(x_by_context * context / tau)
            return torch(x_by_context * weights).sum(dim=1).unsqueeze(dim=1)
        
    elif context_integrator_type == 'separated_by_context':
        def integrator(x_by_context, context):
            return (x_by_context * context).sum(dim=1).unsqueeze(dim=1), x_by_context
        

    return integrator
    
            

class ContextModelPrototypeOuter(nn.Module):
    def __init__(self, 
                 num_filters=64, 
                 in_channels=3, 
                 in_channels_context=4,
                 num_layers_fcn=3, 
                 num_contexts=2,
                 context_model_type='avg',
                 context_integrator_type='weighted_avg',
                 fcn_batchnorm=False,
                 freeze_models=False) -> None:
        """Initializes the context model.
        """
        super().__init__()
        
        self.num_contexts=num_contexts
        self.in_channels=in_channels
        self.context_model_type=context_model_type
        self.context_integrator_type=context_integrator_type
        
        self.freeze_models = freeze_models
                                             
        # base models f_1(x),...,f_K(x)
        if num_layers_fcn == 3:
            self.fcns = torch.nn.ModuleList([TinyFCN(num_filters=num_filters, in_channels=in_channels, classes=1, batchnorm=fcn_batchnorm) for x in range(num_contexts)])
           # [fcn.cuda() for fcn in self.fcns]
        elif num_layers_fcn == 5:     
            self.fcns = self.fcns = torch.nn.ModuleList([FCN(num_filters=num_filters, in_channels=in_channels, classes=1, batchnorm=fcn_batchnorm) for x in range(num_contexts)])
        else:
            print(f'num_layers_fcn {num_layers_fcn} not an option')
            
        # softmax
        self.sm = torch.nn.Softmax(dim=0)

        # context model g(z)
        self.context_model = get_context_model(self.context_model_type, 
                                               self.num_contexts, 
                                               input_channels=in_channels_context)
            
        # moel h(f(x),g(z))
        self.context_integrator = get_context_integrator(self.context_integrator_type)
            
    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """Forward pass of the model."""
        
        ensemble_outputs = torch.hstack([fcn(x) for fcn in self.fcns])
        
        # context modeled
        context_output = self.context_model(c)
        
        if self.context_model_type == 'avg-':
            context_output = torch.hstack((context_output, 1-context_output))

        x = self.context_integrator(ensemble_outputs, context_output)
        
        return x, context_output
    
    
class NoContextModelPrototypeOuterAblation(nn.Module):
    def __init__(self, num_filters=64, in_channels=3,num_layers_fcn=3, num_contexts=2, fcn_batchnorm=False) -> None:
        """Initializes the context model.
        """
        super().__init__()
        
        self.num_contexts=num_contexts
        self.in_channels=in_channels
        
        # softmax
        self.sm = torch.nn.Softmax(dim=1)
        
        if num_layers_fcn == 3:
            self.fcns = torch.nn.ModuleList([TinyFCN(num_filters=num_filters, in_channels=in_channels, classes=1, batchnorm=fcn_batchnorm) for x in range(num_contexts)])
        elif num_layers_fcn == 5:     
            self.fcns = torch.nn.ModuleList([FCN(num_filters=num_filters, in_channels=in_channels, classes=1, batchnorm=fcn_batchnorm) for x in range(num_contexts)])
                            

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model."""
        if self.num_contexts > 1:
            xs = torch.hstack([fcn(x[:,:self.in_channels]) for fcn in self.fcns])


            # shared (learned) context for every instance
            c = nn.Parameter(torch.randn(self.num_contexts), requires_grad=True).to(x.device)

            # combine xs as a sum weighted by c
            c = self.sm(c).unsqueeze(1).unsqueeze(2)
            x = (xs * c).sum(dim=1).unsqueeze(dim=1)
            
        else: 
            fcn = self.fcns[0]
            x = fcn(x[:,:self.in_channels])
        return x
    