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


class SoftTverskyLoss(nn.Module):
    """Handwriting overlap loss with separately tunable FP and FN penalties."""

    def __init__(
        self,
        class_index=1,
        alpha=0.35,
        beta=0.65,
        gamma=0.75,
        ignore_index=255,
        smooth=1.0,
    ):
        super().__init__()
        if alpha <= 0 or beta <= 0:
            raise ValueError("Tversky alpha and beta must be positive")
        if gamma <= 0:
            raise ValueError("Tversky gamma must be positive")
        self.class_index = class_index
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, inputs, targets):
        valid = targets != self.ignore_index
        # Accumulate page-sized masks in fp32 even when CUDA autocast is active;
        # fp16 sums can overflow above roughly 65k pixels.
        probability = F.softmax(inputs.float(), dim=1)[:, self.class_index]
        target = (targets == self.class_index).to(probability.dtype)
        valid = valid.to(probability.dtype)
        probability = probability * valid
        target = target * valid

        dims = (1, 2)
        true_positive = (probability * target).sum(dims)
        false_positive = (probability * (1.0 - target) * valid).sum(dims)
        false_negative = ((1.0 - probability) * target).sum(dims)
        score = (true_positive + self.smooth) / (
            true_positive
            + self.alpha * false_positive
            + self.beta * false_negative
            + self.smooth
        )
        return (1.0 - score).pow(self.gamma).mean()


class HandwritingBoundaryLoss(nn.Module):
    """Binary hand/non-hand loss restricted to a target boundary band."""

    def __init__(self, class_index=1, radius=2, ignore_index=255):
        super().__init__()
        if radius < 1:
            raise ValueError("boundary radius must be at least 1")
        self.class_index = class_index
        self.radius = radius
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        valid = targets != self.ignore_index
        target = (targets == self.class_index).to(torch.float32)
        kernel = self.radius * 2 + 1
        target_4d = target.unsqueeze(1)
        dilated = F.max_pool2d(target_4d, kernel, stride=1, padding=self.radius)
        eroded = -F.max_pool2d(-target_4d, kernel, stride=1, padding=self.radius)
        boundary = (dilated - eroded).squeeze(1) > 0
        boundary = boundary & valid
        if not torch.any(boundary):
            return inputs.sum() * 0.0

        float_inputs = inputs.float()
        handwriting_logit = float_inputs[:, self.class_index]
        other_indices = [
            index for index in range(inputs.shape[1]) if index != self.class_index
        ]
        non_handwriting_logit = torch.logsumexp(
            float_inputs[:, other_indices], dim=1
        )
        binary_logit = handwriting_logit - non_handwriting_logit
        loss = F.binary_cross_entropy_with_logits(
            binary_logit, target, reduction="none"
        )
        return loss[boundary].mean()


class StructureAwareLoss(nn.Module):
    """CE + handwriting Tversky + boundary supervision.

    Weighted CE preserves all three semantic classes. Tversky directly
    penalizes missed handwriting more than false positives, and the boundary
    term reduces halos and broken thin strokes after the mask is erased.
    """

    def __init__(
        self,
        class_weights=None,
        tversky_weight=0.7,
        boundary_weight=0.2,
        tversky_alpha=0.35,
        tversky_beta=0.65,
        tversky_gamma=0.75,
        boundary_radius=2,
        label_smoothing=0.02,
        ignore_index=255,
    ):
        super().__init__()
        if tversky_weight < 0 or boundary_weight < 0:
            raise ValueError("loss component weights must be non-negative")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        weight = (
            torch.empty(0)
            if class_weights is None
            else torch.tensor(class_weights, dtype=torch.float32)
        )
        self.register_buffer("class_weights", weight)
        self.tversky = SoftTverskyLoss(
            class_index=1,
            alpha=tversky_alpha,
            beta=tversky_beta,
            gamma=tversky_gamma,
            ignore_index=ignore_index,
        )
        self.boundary = HandwritingBoundaryLoss(
            class_index=1, radius=boundary_radius, ignore_index=ignore_index
        )
        self.tversky_weight = tversky_weight
        self.boundary_weight = boundary_weight
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        self.last_components = {}

    def forward(self, inputs, targets):
        weight = self.class_weights if self.class_weights.numel() else None
        ce = F.cross_entropy(
            inputs,
            targets,
            weight=weight,
            ignore_index=self.ignore_index,
            label_smoothing=self.label_smoothing,
        )
        tversky = self.tversky(inputs, targets)
        boundary = self.boundary(inputs, targets)
        total = (
            ce
            + self.tversky_weight * tversky
            + self.boundary_weight * boundary
        )
        self.last_components = {
            "cross_entropy": ce.detach(),
            "tversky": tversky.detach(),
            "boundary": boundary.detach(),
        }
        return total


class JointRestorationLoss(nn.Module):
    """Segmentation plus masked, color-preserving RGB restoration losses."""

    def __init__(
        self,
        class_weights=None,
        segmentation_weight=1.0,
        reconstruction_weight=1.0,
        ssim_weight=0.4,
        edge_weight=0.3,
        color_weight=0.2,
        identity_weight=0.1,
        **segmentation_options,
    ):
        super().__init__()
        self.segmentation = StructureAwareLoss(
            class_weights=class_weights, **segmentation_options
        )
        self.segmentation_weight = segmentation_weight
        self.reconstruction_weight = reconstruction_weight
        self.ssim_weight = ssim_weight
        self.edge_weight = edge_weight
        self.color_weight = color_weight
        self.identity_weight = identity_weight
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        )
        self.register_buffer("sobel_x", sobel_x.view(1, 1, 3, 3))
        self.register_buffer("sobel_y", sobel_x.t().view(1, 1, 3, 3))
        self.register_buffer(
            "input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self.last_components = {}

    @staticmethod
    def _masked_mean(values, mask):
        if values.shape[1] != mask.shape[1]:
            mask = mask.expand(-1, values.shape[1], -1, -1)
        denominator = mask.sum().clamp_min(1.0)
        return (values * mask).sum() / denominator

    def _ssim_loss(self, prediction, target, mask):
        mu_x = F.avg_pool2d(prediction, 3, stride=1, padding=1)
        mu_y = F.avg_pool2d(target, 3, stride=1, padding=1)
        sigma_x = F.avg_pool2d(prediction * prediction, 3, 1, 1) - mu_x.pow(2)
        sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y.pow(2)
        sigma_xy = F.avg_pool2d(prediction * target, 3, 1, 1) - mu_x * mu_y
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1)
            * (sigma_x + sigma_y + c2)
        ).clamp_min(1e-6)
        return self._masked_mean((1.0 - ssim).clamp(0.0, 2.0), mask)

    def _edge_loss(self, prediction, target, mask):
        prediction_gray = (
            prediction[:, :1] * 0.299
            + prediction[:, 1:2] * 0.587
            + prediction[:, 2:3] * 0.114
        )
        target_gray = (
            target[:, :1] * 0.299
            + target[:, 1:2] * 0.587
            + target[:, 2:3] * 0.114
        )
        pred_x = F.conv2d(prediction_gray, self.sobel_x, padding=1)
        pred_y = F.conv2d(prediction_gray, self.sobel_y, padding=1)
        target_x = F.conv2d(target_gray, self.sobel_x, padding=1)
        target_y = F.conv2d(target_gray, self.sobel_y, padding=1)
        return self._masked_mean(
            (pred_x - target_x).abs() + (pred_y - target_y).abs(), mask
        )

    def forward(self, outputs, labels, clean_targets, restoration_valid, inputs):
        logits, candidate, restored = outputs
        segmentation = self.segmentation(logits, labels)
        candidate = candidate.float()
        restored = restored.float()
        clean_targets = clean_targets.float()
        inputs = inputs.float()
        raw_rgb = (inputs * self.input_std + self.input_mean).clamp(0.0, 1.0)

        hand_mask = (labels == 1).to(candidate.dtype).unsqueeze(1)
        hand_mask = F.max_pool2d(hand_mask, 5, stride=1, padding=2)
        paired = restoration_valid.to(candidate.dtype).view(-1, 1, 1, 1)
        restore_mask = hand_mask * paired

        if torch.any(restore_mask > 0):
            charbonnier = torch.sqrt(
                (candidate - clean_targets).pow(2) + 1e-6
            )
            reconstruction = self._masked_mean(charbonnier, restore_mask)
            ssim = self._ssim_loss(candidate, clean_targets, restore_mask)
            edge = self._edge_loss(candidate, clean_targets, restore_mask)
            pred_chroma = torch.stack(
                (candidate[:, 0] - candidate[:, 1],
                 candidate[:, 2] - candidate[:, 1]),
                dim=1,
            )
            target_chroma = torch.stack(
                (clean_targets[:, 0] - clean_targets[:, 1],
                 clean_targets[:, 2] - clean_targets[:, 1]),
                dim=1,
            )
            color = self._masked_mean(
                (pred_chroma - target_chroma).abs(), restore_mask
            )
        else:
            zero = candidate.sum() * 0.0
            reconstruction = ssim = edge = color = zero

        outside_mask = 1.0 - hand_mask
        identity = self._masked_mean((restored - raw_rgb).abs(), outside_mask)
        total = (
            self.segmentation_weight * segmentation
            + self.reconstruction_weight * reconstruction
            + self.ssim_weight * ssim
            + self.edge_weight * edge
            + self.color_weight * color
            + self.identity_weight * identity
        )
        self.last_components = {
            "segmentation": segmentation.detach(),
            "reconstruction": reconstruction.detach(),
            "ssim": ssim.detach(),
            "edge": edge.detach(),
            "color": color.detach(),
            "identity": identity.detach(),
        }
        return total


def build_loss(
    name,
    class_weights=None,
    dice_weight=0.5,
    ignore_index=255,
    tversky_weight=0.7,
    boundary_weight=0.2,
    tversky_alpha=0.35,
    tversky_beta=0.65,
    tversky_gamma=0.75,
    boundary_radius=2,
    label_smoothing=0.02,
    segmentation_weight=1.0,
    reconstruction_weight=1.0,
    ssim_weight=0.4,
    edge_weight=0.3,
    color_weight=0.2,
    identity_weight=0.1,
):
    if name == "joint":
        return JointRestorationLoss(
            class_weights=class_weights,
            segmentation_weight=segmentation_weight,
            reconstruction_weight=reconstruction_weight,
            ssim_weight=ssim_weight,
            edge_weight=edge_weight,
            color_weight=color_weight,
            identity_weight=identity_weight,
            tversky_weight=tversky_weight,
            boundary_weight=boundary_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
            boundary_radius=boundary_radius,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
    if name == "structure":
        return StructureAwareLoss(
            class_weights=class_weights,
            tversky_weight=tversky_weight,
            boundary_weight=boundary_weight,
            tversky_alpha=tversky_alpha,
            tversky_beta=tversky_beta,
            tversky_gamma=tversky_gamma,
            boundary_radius=boundary_radius,
            label_smoothing=label_smoothing,
            ignore_index=ignore_index,
        )
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
