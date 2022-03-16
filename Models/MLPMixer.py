import torch
import torch.nn as nn

class PreNormResidual(nn.Module):
    def __init__(
        self, 
        dim: int, 
        fn,
    ):
        super(PreNormResidual, self).__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(
    dim: int,
    expansion_factor: int = 4,
    dropout: float = 0.0,
    dense: torch.nn.Module = nn.Linear,
):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout),
    )

class MLPMixer(nn.Module):
    def __init__(
        self,
    ):
        super(MLPMixer, self).__init__()
        self.gelu = nn.GELU()

    def forward(self, x):
        pass
