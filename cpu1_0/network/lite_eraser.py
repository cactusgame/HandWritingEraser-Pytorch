"""A small segmentation network designed for document images and CPU inference."""

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import mobilenetv2


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            ConvBNAct(in_channels, in_channels, groups=in_channels),
            ConvBNAct(in_channels, out_channels, kernel_size=1),
        )


class LiteEraserNet(nn.Module):
    """MobileNetV2 encoder with a Lite R-ASPP inspired decoder.

    The stride-4 skip keeps thin printed/handwritten strokes, while the global
    gate supplies page-level context without the expensive five-branch ASPP.
    """

    def __init__(self, num_classes=3, output_stride=16, pretrained_backbone=True):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16")

        backbone = mobilenetv2.mobilenet_v2(
            pretrained=pretrained_backbone, output_stride=output_stride
        )
        # features[:4] ends at stride 4 with 24 channels. features[4:-1]
        # ends with 320 channels and avoids MobileNet's 1280-channel head.
        self.low_level = nn.Sequential(*list(backbone.features.children())[:4])
        self.high_level = nn.Sequential(*list(backbone.features.children())[4:-1])

        decoder_channels = 128
        self.high_project = ConvBNAct(320, decoder_channels, kernel_size=1)
        self.context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(320, decoder_channels, 1, bias=True),
            nn.Sigmoid(),
        )
        self.low_project = ConvBNAct(24, 32, kernel_size=1)
        self.fuse = nn.Sequential(
            DepthwiseSeparableConv(decoder_channels + 32, decoder_channels),
            nn.Dropout2d(0.1),
            nn.Conv2d(decoder_channels, num_classes, 1),
        )

        self._init_decoder()

    def _init_decoder(self):
        modules = [self.high_project, self.context, self.low_project, self.fuse]
        for module in modules:
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
        low = self.low_level(x)
        high = self.high_level(low)
        context = self.context(high)
        high = self.high_project(high) * context
        high = F.interpolate(
            high, size=low.shape[-2:], mode="bilinear", align_corners=False
        )
        logits = self.fuse(torch.cat((self.low_project(low), high), dim=1))
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def lite_eraser(
    num_classes=3, output_stride=16, pretrained_backbone=True, **kwargs
):
    return LiteEraserNet(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
