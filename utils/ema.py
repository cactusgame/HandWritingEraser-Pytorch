"""Exponential moving average weights for stable validation and deployment."""

import copy
import math

import torch

from .checkpoint import unwrap_model


class ModelEMA:
    def __init__(self, model, decay=0.999, updates=0):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be between 0 and 1")
        self.module = copy.deepcopy(unwrap_model(model)).eval()
        self.decay = decay
        self.updates = updates
        for parameter in self.module.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        # A short ramp avoids averaging the randomly initialized decoder for
        # thousands of iterations at the beginning of fine-tuning.
        decay = self.decay * (1.0 - math.exp(-self.updates / 2000.0))
        source = unwrap_model(model).state_dict()
        for name, value in self.module.state_dict().items():
            incoming = source[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(incoming, alpha=1.0 - decay)
            else:
                value.copy_(incoming)

    def load_state_dict(self, state_dict, updates=None):
        self.module.load_state_dict(state_dict)
        if updates is not None:
            self.updates = int(updates)

