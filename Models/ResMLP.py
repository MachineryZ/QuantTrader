import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor


class AffineTransform(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        return self.alpha * x + self.beta


class CommunicationLayer(nn.Module):
    def __init__(self, num_features, num_patches):
        super().__init__()
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_patches, num_patches)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.aff2(x)
        out = x + residual
        return out


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.aff1 = AffineTransform(num_features)
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.aff2 = AffineTransform(num_features)

    def forward(self, x):
        x = self.aff1(x)
        residual = x
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.aff2(x)
        out = x + residual
        return out


class ResMLPLayer(nn.Module):
    def __init__(self, num_features, num_patches, expansion_factor):
        super().__init__()
        self.cl = CommunicationLayer(num_features, num_patches)
        self.ff = FeedForward(num_features, expansion_factor)

    def forward(self, x):
        x = self.cl(x)
        out = self.ff(x)
        return out


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class ResMLP(nn.Module):
    def __init__(
        self,
        image_size=256,
        patch_size=16,
        in_channels=3,
        num_features=128,
        expansion_factor=2,
        num_layers=8,
        num_classes=10,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[
                ResMLPLayer(num_features, num_patches, expansion_factor)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        logits = self.classifier(embedding)
        return logits


class ResMLPModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_len: int,
        num_layers: int,
        bias: float,
        dropout: float,
        expansion_factor: int,
    ):
        super(ResMLPModel, self).__init__()
        self.mlps = nn.Sequential(
            *[
                ResMLPLayer(
                    num_features=hidden_size, 
                    num_patches=seq_len, 
                    expansion_factor=expansion_factor,
                )
                for _ in range(num_layers)
            ]
        )
        self.gelu = nn.GELU()
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_out_1 = nn.Linear(hidden_size, hidden_size // 2, bias=bias)
        self.fc_out_2 = nn.Linear(hidden_size // 2, output_size, bias=bias)


    def forward(self, x: Tensor):
        x = self.fc_in(x)
        x = self.gelu(x)
        x = self.mlps(x)
        x = torch.mean(x, dim=1)
        x = self.fc_out_1(x)
        x = self.gelu(x)
        x = self.fc_out_2(x)
        return x.squeeze(-1)

if __name__ == "__main__":
    # sequence model
    # x = torch.randn(1, 3, 256, 256)
    # model = ResMLP()
    x = torch.randn(400, 10, 256)
    model = ResMLPModel(
        input_size=256, 
        hidden_size=32, 
        output_size=1, 
        seq_len=10, 
        num_layers=3, 
        bias=False, 
        dropout=0.4, 
        expansion_factor=2
    )
    y = model(x)
    print(y.shape)
