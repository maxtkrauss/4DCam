from dataclasses import dataclass

from torch import nn


@dataclass
class ParameterReport:
    total: int


class RoiFeatureMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dims=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def parameter_report(self):
        total = sum(param.numel() for param in self.parameters())
        return ParameterReport(total=total)
