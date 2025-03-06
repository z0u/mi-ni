import torch
from torch.optim import AdamW

from experiment.config import OptimizerConfig


def configure_optimizer(model: torch.nn.Module, config: OptimizerConfig):
    # apply weight decay to weight matrices (2+D tensors) but not to biases/scales (1-D tensors)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    decay_params = [p for p in params if p.dim() >= 2]
    nodecay_params = [p for p in params if p.dim() < 2]

    optim_groups = [
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ]

    return AdamW(optim_groups, lr=config.learning_rate, betas=config.betas)
