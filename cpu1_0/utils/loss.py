"""Losses for imbalanced handwriting segmentation."""

import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """Class-aware multi-class focal loss.

    ``alpha`` may be a scalar or one weight per class. Unlike the previous
    implementation, a sequence actually changes the relative class weights.
    """

    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, reduction="mean"):
        super().__init__()
        if alpha is None:
            alpha_tensor = torch.empty(0)
        elif isinstance(alpha, (list, tuple)):
            alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
        else:
            alpha_tensor = torch.tensor([float(alpha)], dtype=torch.float32)
        self.register_buffer("alpha", alpha_tensor)
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, inputs, targets):
        valid = targets != self.ignore_index
        if not torch.any(valid):
            return inputs.sum() * 0.0

        ce = F.cross_entropy(
            inputs, targets, ignore_index=self.ignore_index, reduction="none"
        )
        pt = torch.exp(-ce)
        loss = (1.0 - pt).pow(self.gamma) * ce

        if self.alpha.numel() == 1:
            loss = loss * self.alpha[0]
        elif self.alpha.numel() > 1:
            safe_targets = targets.clamp(0, self.alpha.numel() - 1)
            loss = loss * self.alpha[safe_targets]

        loss = loss[valid]
        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class SoftDiceLoss(nn.Module):
    """Multi-class soft Dice loss, normally evaluated on foreground classes."""

    def __init__(self, include_background=False, ignore_index=255, smooth=1.0):
        super().__init__()
        self.include_background = include_background
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        num_classes = inputs.shape[1]
        valid = targets != self.ignore_index
        safe_targets = targets.masked_fill(~valid, 0)
        target_1h = F.one_hot(safe_targets, num_classes).permute(0, 3, 1, 2)
        target_1h = target_1h.to(dtype=inputs.dtype)
        valid = valid.unsqueeze(1)

        probs = F.softmax(inputs, dim=1) * valid
        target_1h = target_1h * valid
        if not self.include_background and num_classes > 1:
            probs = probs[:, 1:]
            target_1h = target_1h[:, 1:]

        dims = (0, 2, 3)
        intersection = (probs * target_1h).sum(dims)
        denominator = probs.sum(dims) + target_1h.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        # Classes absent from the target do not dominate a small batch.
        present = target_1h.sum(dims) > 0
        if torch.any(present):
            dice = dice[present]
            return 1.0 - dice.mean()
        return inputs.sum() * 0.0


class HybridSegmentationLoss(nn.Module):
    """Weighted cross entropy plus foreground Dice."""

    def __init__(self, class_weights=None, dice_weight=0.5, ignore_index=255):
        super().__init__()
        if class_weights is None:
            weight = torch.empty(0)
        else:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        self.register_buffer("class_weights", weight)
        self.dice = SoftDiceLoss(False, ignore_index)
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        weight = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
        )
        return ce + self.dice_weight * self.dice(inputs, targets)


def build_loss(name, class_weights=None, dice_weight=0.5, ignore_index=255):
    if name == "hybrid":
        return HybridSegmentationLoss(class_weights, dice_weight, ignore_index)
    if name == "focal":
        return FocalLoss(class_weights, ignore_index=ignore_index)
    if name == "cross_entropy":
        weight = None
        if class_weights is not None:
            weight = torch.tensor(class_weights, dtype=torch.float32)
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    raise ValueError("unknown loss: %s" % name)
