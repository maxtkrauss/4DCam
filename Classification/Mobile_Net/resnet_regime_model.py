from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ResNetParameterReport:
    total: int
    stem: int
    body: int
    head: int


def _norm(channels: int):
    groups = min(8, channels)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = _norm(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _norm(out_channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.proj = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                _norm(out_channels),
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = self.proj(x)
        out = self.act(self.norm1(self.conv1(x)))
        out = self.drop(out)
        out = self.norm2(self.conv2(out))
        return self.act(out + residual)


class SqueezeExcite2d(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return x * self.fc(self.pool(x))


class FactorizedSpectralProjectionStem(nn.Module):
    """Project each spectral group independently before spatial feature extraction."""

    def __init__(
        self,
        in_channels: int,
        spectral_bands: int = 106,
        latent_channels_per_group: int = 16,
        fallback_channels: int = 32,
        use_channel_attention: bool = False,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.spectral_bands = int(spectral_bands)
        self.latent_channels_per_group = int(latent_channels_per_group)
        self.factorized = self.in_channels >= self.spectral_bands and self.in_channels % self.spectral_bands == 0

        if self.factorized:
            self.num_groups = self.in_channels // self.spectral_bands
            self.out_channels = self.num_groups * self.latent_channels_per_group
            self.group_projection = nn.Sequential(
                nn.Conv2d(self.spectral_bands, self.latent_channels_per_group, kernel_size=1, bias=False),
                _norm(self.latent_channels_per_group),
                nn.ReLU(inplace=True),
            )
        else:
            self.num_groups = 1
            self.out_channels = int(fallback_channels)
            self.group_projection = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, bias=False),
                _norm(self.out_channels),
                nn.ReLU(inplace=True),
            )

        self.attention = SqueezeExcite2d(self.out_channels) if use_channel_attention else nn.Identity()

    def forward(self, x: torch.Tensor):
        if not self.factorized:
            return self.attention(self.group_projection(x))

        batch_size, _, height, width = x.shape
        grouped = x.reshape(batch_size * self.num_groups, self.spectral_bands, height, width)
        projected = self.group_projection(grouped)
        projected = projected.reshape(
            batch_size,
            self.num_groups * self.latent_channels_per_group,
            height,
            width,
        )
        return self.attention(projected)


class HybridSpectralProjectionStem(nn.Module):
    """Fuse grouped spectral features with a full-channel projection."""

    def __init__(
        self,
        in_channels: int,
        spectral_bands: int = 106,
        latent_channels_per_group: int = 24,
        full_projection_channels: int = 64,
        fusion_channels: int = 96,
        use_channel_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.spectral_bands = int(spectral_bands)
        self.latent_channels_per_group = int(latent_channels_per_group)
        self.factorized = self.in_channels >= self.spectral_bands and self.in_channels % self.spectral_bands == 0

        if self.factorized:
            self.num_groups = self.in_channels // self.spectral_bands
            grouped_channels = self.num_groups * self.latent_channels_per_group
            self.group_projection = nn.Sequential(
                nn.Conv2d(self.spectral_bands, self.latent_channels_per_group, kernel_size=1, bias=False),
                _norm(self.latent_channels_per_group),
                nn.ReLU(inplace=True),
            )
        else:
            self.num_groups = 1
            grouped_channels = int(full_projection_channels)
            self.group_projection = nn.Sequential(
                nn.Conv2d(self.in_channels, grouped_channels, kernel_size=1, bias=False),
                _norm(grouped_channels),
                nn.ReLU(inplace=True),
            )

        self.full_projection = nn.Sequential(
            nn.Conv2d(self.in_channels, full_projection_channels, kernel_size=1, bias=False),
            _norm(full_projection_channels),
            nn.ReLU(inplace=True),
        )
        self.out_channels = int(fusion_channels)
        self.fusion = nn.Sequential(
            nn.Conv2d(grouped_channels + full_projection_channels, self.out_channels, kernel_size=1, bias=False),
            _norm(self.out_channels),
            nn.ReLU(inplace=True),
        )
        self.attention = SqueezeExcite2d(self.out_channels) if use_channel_attention else nn.Identity()

    def forward(self, x: torch.Tensor):
        full = self.full_projection(x)
        if self.factorized:
            batch_size, _, height, width = x.shape
            grouped = x.reshape(batch_size * self.num_groups, self.spectral_bands, height, width)
            grouped = self.group_projection(grouped)
            grouped = grouped.reshape(
                batch_size,
                self.num_groups * self.latent_channels_per_group,
                height,
                width,
            )
        else:
            grouped = self.group_projection(x)

        return self.attention(self.fusion(torch.cat([grouped, full], dim=1)))


class CompactRegimeResNet(nn.Module):
    """Small residual CNN for fair comparison across all regime channel counts."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int = 32,
        dropout: float = 0.1,
        projection_channels: int | None = None,
        use_channel_attention: bool = False,
        factorized_latent: bool = False,
        hybrid_spectral: bool = False,
        spectral_bands: int = 106,
        latent_channels_per_group: int = 16,
        hybrid_fusion_channels: int = 96,
    ):
        super().__init__()
        stem_in_channels = in_channels
        widths = [base_channels, base_channels * 2, base_channels * 4, base_channels * 6]
        self.adapter = nn.Identity()
        if hybrid_spectral:
            self.adapter = HybridSpectralProjectionStem(
                in_channels=in_channels,
                spectral_bands=spectral_bands,
                latent_channels_per_group=latent_channels_per_group,
                full_projection_channels=projection_channels or base_channels * 2,
                fusion_channels=hybrid_fusion_channels,
                use_channel_attention=use_channel_attention,
            )
            stem_in_channels = self.adapter.out_channels
        elif factorized_latent:
            self.adapter = FactorizedSpectralProjectionStem(
                in_channels=in_channels,
                spectral_bands=spectral_bands,
                latent_channels_per_group=latent_channels_per_group,
                fallback_channels=projection_channels or base_channels,
                use_channel_attention=use_channel_attention,
            )
            stem_in_channels = self.adapter.out_channels
        elif projection_channels is not None:
            adapter_layers = [
                nn.Conv2d(in_channels, projection_channels, kernel_size=1, bias=False),
                _norm(projection_channels),
                nn.ReLU(inplace=True),
            ]
            if use_channel_attention:
                adapter_layers.append(SqueezeExcite2d(projection_channels))
            self.adapter = nn.Sequential(*adapter_layers)
            stem_in_channels = projection_channels

        self.stem = nn.Sequential(
            nn.Conv2d(stem_in_channels, widths[0], kernel_size=3, stride=2, padding=1, bias=False),
            _norm(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.body = nn.Sequential(
            ResidualBlock(widths[0], widths[0], stride=1, dropout=dropout),
            ResidualBlock(widths[0], widths[1], stride=2, dropout=dropout),
            ResidualBlock(widths[1], widths[1], stride=1, dropout=dropout),
            ResidualBlock(widths[1], widths[2], stride=2, dropout=dropout),
            ResidualBlock(widths[2], widths[2], stride=1, dropout=dropout),
            ResidualBlock(widths[2], widths[3], stride=2, dropout=dropout),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(widths[3], num_classes)

    def forward(self, x: torch.Tensor):
        x = self.adapter(x)
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.head(x)

    def parameter_report(self):
        def count(module):
            return sum(param.numel() for param in module.parameters())

        return ResNetParameterReport(
            total=count(self),
            stem=count(self.adapter) + count(self.stem),
            body=count(self.body),
            head=count(self.head),
        )
