import os
from typing import Any, cast
import torch

from pixelwise_regression_task_with_mask import RegressionTaskWithMask
from models import ContextModelPrototypeOuter,NoContextModelPrototypeOuterAblation

class ContextualizedPixelwiseRegressionTask(RegressionTaskWithMask):
    """LightningModule for pixelwise regression of images.

    # adaptation of https://torchgeo.readthedocs.io/en/latest/_modules/torchgeo/trainers/regression.html#PixelwiseRegressionTask
    """

    target_key: str = "mask"
    
    def config_model(self) -> None:
        """Configures the model based on kwargs parameters."""
        if "weights_fp" in self.hyperparams: 
            weights_fp = self.hyperparams["weights_fp"]
            weights = True
        else: weights = False
            
        use_context = self.hyperparams["context"]
        
        
        # instantiates the model 
        if use_context:
            self.model = ContextModelPrototypeOuter(in_channels=self.hyperparams["in_channels"],
                                                    num_filters=self.hyperparams["num_filters"],
                                                    num_layers_fcn=self.hyperparams["num_layers_fcn"],
                                                    fcn_batchnorm=self.hyperparams["fcn_batchnorm"],
                                                    context_model_type=self.hyperparams["context_model_type"],
                                                    context_integrator_type=self.hyperparams["context_integrator_type"],
                                                    num_contexts=self.hyperparams["num_contexts"],
                                                    freeze_models=self.hyperparams["freeze_models"]
                                                   )
            
            # self.context_model = self.model.context_model
            self.has_context_model = True
            
            if 'context_reg' in self.hyperparams.keys():
                self.context_reg = self.hyperparams["context_reg"]
            else:
                self.context_reg = 0.0
        else:
            self.model = NoContextModelPrototypeOuterAblation(in_channels=self.hyperparams["in_channels"],
                                                              num_filters=self.hyperparams["num_filters"],
                                                              num_layers_fcn=self.hyperparams["num_layers_fcn"],
                                                              fcn_batchnorm=self.hyperparams["fcn_batchnorm"],
                                                              num_contexts=self.hyperparams["num_contexts"])
        
        
        # load weights if given
        print(weights)
        if weights:
            print(f'using weights from {weights_fp}')
            state_dict = torch.load(self.hyperparams["weights_fp"])
            self.model.load_state_dict(state_dict)

                
    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.
        """
        super().__init__(**kwargs)
        
        self.config_model()