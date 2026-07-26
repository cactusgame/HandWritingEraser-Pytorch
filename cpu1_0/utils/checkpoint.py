"""Portable checkpoint helpers shared by training, export and inference."""

from collections import OrderedDict

import torch


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def clean_state_dict(state_dict):
    cleaned = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


def load_checkpoint(path, model, map_location="cpu", strict=True):
    checkpoint = torch.load(path, map_location=map_location)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
        checkpoint = {"model_state": state_dict}
    unwrap_model(model).load_state_dict(clean_state_dict(state_dict), strict=strict)
    return checkpoint


def checkpoint_model_config(checkpoint, defaults=None):
    config = dict(defaults or {})
    if isinstance(checkpoint, dict):
        config.update(checkpoint.get("model_config", {}))
    return config
