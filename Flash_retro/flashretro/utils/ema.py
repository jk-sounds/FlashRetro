"""
Exponential Moving Average (EMA) utilities for model training.

Implementation follows MeanFlow-main/train.py exactly.
Reference: https://github.com/xxx/MeanFlow-main
"""

import torch
from collections import OrderedDict


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    
    This function updates the EMA model parameters using exponential moving average:
        ema_param = decay * ema_param + (1 - decay) * current_param
    
    Args:
        ema_model: The EMA model (will be updated in-place)
        model: The current training model
        decay: EMA decay rate (default: 0.9999)
            - Higher values (closer to 1.0) = slower update, more stable
            - Lower values (closer to 0.0) = faster update, less stable
            - Special case: decay=0 means complete copy (initialization)
    
    Note:
        - This function uses @torch.no_grad() to avoid tracking gradients
        - It handles DDP models by removing "module." prefix from parameter names
        - The function modifies ema_model parameters in-place
    
    Example:
        >>> model = MyModel()
        >>> ema = deepcopy(model)
        >>> # Initialize EMA with current model weights
        >>> update_ema(ema, model, decay=0)
        >>> # Training loop
        >>> for batch in dataloader:
        >>>     loss = train_step(model, batch)
        >>>     optimizer.step()
        >>>     # Update EMA after each training step
        >>>     update_ema(ema, model)  # Uses default decay=0.9999
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # Handle DDP models: remove "module." prefix
        name = name.replace("module.", "")
        # Update EMA parameter: ema = decay * ema + (1 - decay) * param
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    
    This is useful for freezing EMA model parameters to save memory
    and prevent accidental gradient computation.
    
    Args:
        model: PyTorch model
        flag: If True, parameters will require gradients (trainable)
              If False, parameters will not require gradients (frozen)
    
    Example:
        >>> ema = deepcopy(model)
        >>> requires_grad(ema, False)  # Freeze EMA model
    """
    for p in model.parameters():
        p.requires_grad = flag



