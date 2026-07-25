"""GPU quality model for layered handwriting removal and print restoration."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as torchvision_models

from .lite_eraser import EfficientPyramidContext


def _resnet34(pretrained):
    try:
        weights = (
            torchvision_models.ResNet34_Weights.DEFAULT if pretrained else None
        )
        return torchvision_models.resnet34(weights=weights)
    except AttributeError:  # torchvision < 0.13
        return torchvision_models.resnet34(pretrained=pretrained)


def _groups(channels):
    for value in (16, 8, 4, 2, 1):
        if channels % value == 0:
            return value
    return 1


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                padding=padding, bias=False,
            ),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualRefine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvGNAct(channels, channels),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs):
        return self.activation(inputs + self.block(inputs))


class LayeredDecoderBlock(nn.Module):
    def __init__(self, decoder_channels, skip_channels, out_channels):
        super().__init__()
        skip_out = max(32, out_channels // 2)
        self.skip = ConvGNAct(skip_channels, skip_out, 1)
        self.fuse = nn.Sequential(
            ConvGNAct(decoder_channels + skip_out, out_channels),
            ResidualRefine(out_channels),
        )

    def forward(self, decoder, skip):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear",
            align_corners=False,
        )
        return self.fuse(torch.cat((decoder, self.skip(skip)), dim=1))


class LayeredEraserNet(nn.Module):
    """One model with segmentation, amodal print, and clean-RGB outputs.

    The amodal print head explicitly predicts printed structure even where it is
    covered by handwriting.  The RGB head receives both hand and print
    probabilities, allowing it to learn separate behavior for paper-only and
    hand-over-print pixels.
    """

    def __init__(self, num_classes=3, output_stride=32, pretrained_backbone=True):
        super().__init__()
        if output_stride != 32:
            raise ValueError("layered_eraser requires output_stride=32")
        backbone = _resnet34(pretrained_backbone)
        self.encoder_s2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        self.encoder_s4 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.encoder_s8 = backbone.layer2
        self.encoder_s16 = backbone.layer3
        self.encoder_s32 = backbone.layer4

        self.context = EfficientPyramidContext(512, 160, 256)
        self.decode_s16 = LayeredDecoderBlock(256, 256, 256)
        self.decode_s8 = LayeredDecoderBlock(256, 128, 192)
        self.decode_s4 = LayeredDecoderBlock(192, 64, 128)
        self.decode_s2 = LayeredDecoderBlock(128, 64, 96)

        self.segmentation_head = nn.Sequential(
            ConvGNAct(96, 64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, num_classes, 1),
        )
        self.print_head = nn.Sequential(
            ConvGNAct(96, 64),
            ResidualRefine(64),
            nn.Conv2d(64, 1, 1),
        )

        # Most semantic restoration is done at stride 2.  The input is already
        # enhanced upstream, so full resolution only needs a narrow detail
        # refinement for 1-3 pixel printed strokes.
        self.restore_s2 = nn.Sequential(
            ConvGNAct(96 + 3 + 2, 128),
            ResidualRefine(128),
            ResidualRefine(128),
            ConvGNAct(128, 64),
        )
        self.restore_s2_residual = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh(),
        )
        self.restore_full_project = ConvGNAct(64, 32, 1)
        self.restore_full = nn.Sequential(
            ConvGNAct(32 + 3 + 3 + 2, 24),
            ResidualRefine(24),
            ConvGNAct(24, 16),
            nn.Conv2d(16, 3, 1),
            nn.Tanh(),
        )

        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )
        self._init_new_layers()

    def _init_new_layers(self):
        modules = (
            self.context, self.decode_s16, self.decode_s8, self.decode_s4,
            self.decode_s2, self.segmentation_head, self.print_head,
            self.restore_s2, self.restore_s2_residual,
            self.restore_full_project, self.restore_full,
        )
        for module in modules:
            for layer in module.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            # Batch 2 is the safe default for 8 GB GPUs.  Keep pretrained
            # backbone BN statistics fixed instead of updating them noisily.
            for encoder in (
                self.encoder_s2, self.encoder_s4, self.encoder_s8,
                self.encoder_s16, self.encoder_s32,
            ):
                for layer in encoder.modules():
                    if isinstance(layer, nn.BatchNorm2d):
                        layer.eval()
        return self

    def forward(self, inputs):
        output_size = inputs.shape[-2:]
        raw_rgb = (inputs * self.input_std + self.input_mean).clamp(0.0, 1.0)

        feature_s2 = self.encoder_s2(inputs)
        feature_s4 = self.encoder_s4(feature_s2)
        feature_s8 = self.encoder_s8(feature_s4)
        feature_s16 = self.encoder_s16(feature_s8)
        feature_s32 = self.encoder_s32(feature_s16)

        decoded = self.context(feature_s32)
        decoded = self.decode_s16(decoded, feature_s16)
        decoded = self.decode_s8(decoded, feature_s8)
        decoded = self.decode_s4(decoded, feature_s4)
        decoded_s2 = self.decode_s2(decoded, feature_s2)

        segmentation_s2 = self.segmentation_head(decoded_s2)
        print_s2 = self.print_head(decoded_s2)
        segmentation_logits = F.interpolate(
            segmentation_s2, size=output_size, mode="bilinear",
            align_corners=False,
        )
        print_logits = F.interpolate(
            print_s2, size=output_size, mode="bilinear",
            align_corners=False,
        )
        hand_probability = F.softmax(
            segmentation_logits, dim=1
        )[:, 1:2]
        print_probability = torch.sigmoid(print_logits)

        half_size = decoded_s2.shape[-2:]
        raw_s2 = F.interpolate(
            raw_rgb, size=half_size, mode="bilinear", align_corners=False
        )
        hand_s2 = F.interpolate(
            hand_probability, size=half_size, mode="bilinear",
            align_corners=False,
        )
        print_probability_s2 = F.interpolate(
            print_probability, size=half_size, mode="bilinear",
            align_corners=False,
        )
        restoration_s2 = self.restore_s2(torch.cat(
            (decoded_s2, raw_s2, hand_s2, print_probability_s2), dim=1
        ))
        residual_s2 = self.restore_s2_residual(restoration_s2)
        residual = F.interpolate(
            residual_s2, size=output_size, mode="bilinear",
            align_corners=False,
        )
        full_features = F.interpolate(
            self.restore_full_project(restoration_s2),
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )
        fine_residual = self.restore_full(torch.cat(
            (
                full_features, raw_rgb, residual,
                hand_probability, print_probability,
            ),
            dim=1,
        ))
        candidate = (raw_rgb + residual + 0.5 * fine_residual).clamp(0.0, 1.0)

        # A smaller expansion than the CPU model avoids erasing adjacent print.
        blend_mask = F.max_pool2d(
            hand_probability, 3, stride=1, padding=1
        )
        blend_mask = ((blend_mask - 0.15) / 0.75).clamp(0.0, 1.0)
        restored = raw_rgb * (1.0 - blend_mask) + candidate * blend_mask
        return segmentation_logits, print_logits, candidate, restored


def layered_eraser(
    num_classes=3, output_stride=32, pretrained_backbone=True, **kwargs
):
    return LayeredEraserNet(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
