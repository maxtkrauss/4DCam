from dataclasses import dataclass

import torch
from torch import nn

from models.mobilenetv3 import mobilenet_v3_large


@dataclass
class ParameterReport:
    total: int
    adapter: int
    body: int
    head: int


class InputAdapterStem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int = 16):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            nn.Hardswish(inplace=True),
        )


class SharedProjectionStem(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
            nn.Hardswish(inplace=True),
        )


class RegimeMobileNetV3Large(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        base = mobilenet_v3_large(num_classes=num_classes, dropout=dropout, in_channels=3)

        self.adapter = InputAdapterStem(in_channels=in_channels, out_channels=16)
        self.body = nn.Sequential(*list(base.features.children())[1:])
        self.avg_pool = base.avg_pool
        self.classifier = base.classifier

    def forward(self, x: torch.Tensor):
        x = self.adapter(x)
        x = self.body(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def parameter_report(self):
        def count(module):
            return sum(param.numel() for param in module.parameters())

        return ParameterReport(
            total=count(self),
            adapter=count(self.adapter),
            body=count(self.body),
            head=count(self.classifier),
        )


class SharedProjectionStemMobileNetV3Large(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2, projection_channels: int = 16):
        super().__init__()
        base = mobilenet_v3_large(
            num_classes=num_classes,
            dropout=dropout,
            in_channels=projection_channels,
        )

        self.adapter = SharedProjectionStem(in_channels=in_channels, out_channels=projection_channels)
        self.body = base.features
        self.avg_pool = base.avg_pool
        self.classifier = base.classifier

    def forward(self, x: torch.Tensor):
        x = self.adapter(x)
        x = self.body(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def parameter_report(self):
        def count(module):
            return sum(param.numel() for param in module.parameters())

        return ParameterReport(
            total=count(self),
            adapter=count(self.adapter),
            body=count(self.body),
            head=count(self.classifier),
        )
