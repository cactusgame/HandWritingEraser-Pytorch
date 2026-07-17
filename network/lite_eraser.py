"""CPU-oriented segmentation networks for document handwriting removal."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models as torchvision_models

from .backbone import mobilenetv2


class ConvBNAct(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, groups=1, dilation=1
    ):
        padding = (kernel_size // 2) * dilation
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                groups=groups,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )


class DepthwiseSeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(
            ConvBNAct(
                in_channels, in_channels, groups=in_channels, dilation=dilation
            ),
            ConvBNAct(in_channels, out_channels, kernel_size=1),
        )


class ResidualDepthwiseBlock(nn.Module):
    """Cheap local refinement that remains friendly to CPU inference."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, groups=channels),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU6(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))


class EfficientPyramidContext(nn.Module):
    """Multi-scale context without the large activation cost of full ASPP."""

    def __init__(self, in_channels=320, hidden_channels=96, out_channels=128):
        super().__init__()
        self.reduce = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        branch_channels = 48
        self.branches = nn.ModuleList(
            DepthwiseSeparableConv(
                hidden_channels, branch_channels, dilation=dilation
            )
            for dilation in (1, 3, 6)
        )
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, branch_channels, 1, bias=True),
            nn.ReLU6(inplace=True),
        )
        self.project = ConvBNAct(branch_channels * 4, out_channels, kernel_size=1)

    def forward(self, x):
        reduced = self.reduce(x)
        features = [branch(reduced) for branch in self.branches]
        global_context = F.interpolate(
            self.global_context(x),
            size=reduced.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        features.append(global_context)
        return self.project(torch.cat(features, dim=1))


class DecoderFuse(nn.Module):
    def __init__(self, decoder_channels, skip_channels, skip_out, out_channels):
        super().__init__()
        self.skip_project = ConvBNAct(skip_channels, skip_out, kernel_size=1)
        self.refine = nn.Sequential(
            DepthwiseSeparableConv(decoder_channels + skip_out, out_channels),
            ResidualDepthwiseBlock(out_channels),
        )

    def forward(self, decoder, skip):
        decoder = F.interpolate(
            decoder, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(torch.cat((decoder, self.skip_project(skip)), dim=1))


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


class QualityEraserNet(nn.Module):
    """Accuracy-oriented document segmenter for 4-8 core CPU servers.

    The encoder is still MobileNetV2, while the decoder consumes stride-2,
    stride-4 and stride-8 skips. This is important for strokes that nearly
    disappear at stride 4. A compact dilated context block separates long
    printed rules from locally similar handwriting without using expensive
    attention or a full-width ASPP.
    """

    def __init__(self, num_classes=3, output_stride=16, pretrained_backbone=True):
        super().__init__()
        if output_stride != 16:
            raise ValueError("quality_eraser currently requires output_stride=16")

        backbone = mobilenetv2.mobilenet_v2(
            pretrained=pretrained_backbone, output_stride=output_stride
        )
        features = list(backbone.features.children())
        # Feature shapes for MobileNetV2: 16@s2, 24@s4, 32@s8, 320@s16.
        self.encoder_s2 = nn.Sequential(*features[:2])
        self.encoder_s4 = nn.Sequential(*features[2:4])
        self.encoder_s8 = nn.Sequential(*features[4:7])
        self.encoder_s16 = nn.Sequential(*features[7:-1])

        self.context = EfficientPyramidContext(320, 96, 128)
        self.decode_s8 = DecoderFuse(128, 32, 48, 96)
        self.decode_s4 = DecoderFuse(96, 24, 32, 64)
        self.decode_s2 = DecoderFuse(64, 16, 24, 48)
        self.classifier = nn.Sequential(
            DepthwiseSeparableConv(48, 32),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, num_classes, 1),
        )
        self._init_decoder()

    def _init_decoder(self):
        modules = (
            self.context,
            self.decode_s8,
            self.decode_s4,
            self.decode_s2,
            self.classifier,
        )
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
        feature_s2 = self.encoder_s2(x)
        feature_s4 = self.encoder_s4(feature_s2)
        feature_s8 = self.encoder_s8(feature_s4)
        feature_s16 = self.encoder_s16(feature_s8)

        decoded = self.context(feature_s16)
        decoded = self.decode_s8(decoded, feature_s8)
        decoded = self.decode_s4(decoded, feature_s4)
        decoded = self.decode_s2(decoded, feature_s2)
        logits = self.classifier(decoded)
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def quality_eraser(
    num_classes=3, output_stride=16, pretrained_backbone=True, **kwargs
):
    return QualityEraserNet(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )


def _resnet18(pretrained):
    try:
        weights = (
            torchvision_models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        return torchvision_models.resnet18(weights=weights)
    except AttributeError:  # torchvision < 0.13 compatibility
        return torchvision_models.resnet18(pretrained=pretrained)


class ServerEraserNet(nn.Module):
    """ResNet-18 encoder and fine-detail decoder for CPU server inference.

    Compared with the MobileNet variants this spends more compute on semantic
    discrimination, which is useful for confusing printed annotations, rules
    and handwriting, while remaining much smaller than legacy ResNet-101
    DeepLab.
    """

    def __init__(self, num_classes=3, output_stride=32, pretrained_backbone=True):
        super().__init__()
        if output_stride != 32:
            raise ValueError("server_eraser requires output_stride=32")
        backbone = _resnet18(pretrained_backbone)

        self.encoder_s2 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        self.encoder_s4 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.encoder_s8 = backbone.layer2
        self.encoder_s16 = backbone.layer3
        self.encoder_s32 = backbone.layer4

        self.context = EfficientPyramidContext(512, 128, 192)
        self.decode_s16 = DecoderFuse(192, 256, 64, 160)
        self.decode_s8 = DecoderFuse(160, 128, 56, 128)
        self.decode_s4 = DecoderFuse(128, 64, 40, 96)
        self.decode_s2 = DecoderFuse(96, 64, 32, 64)
        self.classifier = nn.Sequential(
            DepthwiseSeparableConv(64, 48),
            nn.Dropout2d(0.1),
            nn.Conv2d(48, num_classes, 1),
        )
        self._init_decoder()

    def _init_decoder(self):
        for module in (
            self.context,
            self.decode_s16,
            self.decode_s8,
            self.decode_s4,
            self.decode_s2,
            self.classifier,
        ):
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
        feature_s2 = self.encoder_s2(x)
        feature_s4 = self.encoder_s4(feature_s2)
        feature_s8 = self.encoder_s8(feature_s4)
        feature_s16 = self.encoder_s16(feature_s8)
        feature_s32 = self.encoder_s32(feature_s16)
        decoded = self.context(feature_s32)
        decoded = self.decode_s16(decoded, feature_s16)
        decoded = self.decode_s8(decoded, feature_s8)
        decoded = self.decode_s4(decoded, feature_s4)
        decoded = self.decode_s2(decoded, feature_s2)
        logits = self.classifier(decoded)
        return F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )


def server_eraser(
    num_classes=3, output_stride=32, pretrained_backbone=True, **kwargs
):
    return ServerEraserNet(
        num_classes=num_classes,
        output_stride=output_stride,
        pretrained_backbone=pretrained_backbone,
    )
