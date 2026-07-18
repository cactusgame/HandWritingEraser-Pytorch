"""Single-pass handwriting segmentation and printed-content restoration."""

import torch
from torch import nn
from torch.nn import functional as F

from .lite_eraser import (
    ConvBNAct,
    DepthwiseSeparableConv,
    ResidualDepthwiseBlock,
    ServerEraserNet,
)


class JointEraserNet(ServerEraserNet):
    """One shared encoder with segmentation and RGB restoration heads.

    ``candidate`` learns the clean RGB page. ``restored`` is the safe deployed
    result: pixels outside the internally predicted handwriting mask are copied
    from the input exactly.
    """

    def __init__(self, num_classes=3, output_stride=32, pretrained_backbone=True):
        super().__init__(num_classes, output_stride, pretrained_backbone)
        self.restore_half = nn.Sequential(
            DepthwiseSeparableConv(64 + 3 + 1, 32),
            ResidualDepthwiseBlock(32),
            ResidualDepthwiseBlock(32),
        )
        self.restore_half_out = nn.Sequential(
            nn.Conv2d(32, 3, 1),
            nn.Tanh(),
        )
        self.restore_full = nn.Sequential(
            ConvBNAct(3 + 3 + 1, 12),
            ResidualDepthwiseBlock(12),
            nn.Conv2d(12, 3, 1),
            nn.Tanh(),
        )
        self.register_buffer(
            "input_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self._init_restoration_head()

    def _init_restoration_head(self):
        for module in (self.restore_half, self.restore_half_out, self.restore_full):
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        input_size = x.shape[-2:]
        raw_rgb = (x * self.input_std + self.input_mean).clamp(0.0, 1.0)

        feature_s2 = self.encoder_s2(x)
        feature_s4 = self.encoder_s4(feature_s2)
        feature_s8 = self.encoder_s8(feature_s4)
        feature_s16 = self.encoder_s16(feature_s8)
        feature_s32 = self.encoder_s32(feature_s16)
        decoded = self.context(feature_s32)
        decoded = self.decode_s16(decoded, feature_s16)
        decoded = self.decode_s8(decoded, feature_s8)
        decoded = self.decode_s4(decoded, feature_s4)
        decoded_s2 = self.decode_s2(decoded, feature_s2)

        logits_s2 = self.classifier(decoded_s2)
        logits = F.interpolate(
            logits_s2, size=input_size, mode="bilinear", align_corners=False
        )
        hand_probability = F.softmax(logits, dim=1)[:, 1:2]

        half_size = decoded_s2.shape[-2:]
        raw_half = F.interpolate(
            raw_rgb, size=half_size, mode="bilinear", align_corners=False
        )
        mask_half = F.interpolate(
            hand_probability, size=half_size, mode="bilinear", align_corners=False
        )
        restoration_feature = self.restore_half(
            torch.cat((decoded_s2, raw_half, mask_half), dim=1)
        )
        half_residual = self.restore_half_out(restoration_feature)
        residual = F.interpolate(
            half_residual,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        fine_residual = self.restore_full(
            torch.cat((residual, raw_rgb, hand_probability), dim=1)
        )
        candidate = (raw_rgb + residual + 0.25 * fine_residual).clamp(0.0, 1.0)

        # Include anti-aliased ink edges, then smoothly suppress very uncertain
        # predictions. The same operation is exportable to TorchScript/ONNX.
        blend_mask = F.max_pool2d(hand_probability, 5, stride=1, padding=2)
        blend_mask = ((blend_mask - 0.10) / 0.80).clamp(0.0, 1.0)
        restored = raw_rgb * (1.0 - blend_mask) + candidate * blend_mask
        return logits, candidate, restored


def joint_eraser(
    num_classes=3, output_stride=32, pretrained_backbone=True, **kwargs
):
    return JointEraserNet(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
