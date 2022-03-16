import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from functools import partial

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

# ============================== Modified Version ==============================

class MLPMixerModel(nn.Module):
    """
    MLP Mixer:
    In computer vision, we could transfer an image size of 
    (batch_size, channel_size, width, height) into
    (batch_size, patch_size * patch_size, patch_number)

    Corresponding to cv, nlp data is in shape of 
    (batch_size, seq_len, feature_size)
    
    We would like to correspond seq_len as patch_number and feature_size as patch_size**2
    So in our MLPMixerModel, we need to transpose the 2nd and 3rd channel while forwarding x
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_len: int,
        expansion_ratio: float,
        shrink_ratio: float,
        depth: int,
        bias: bool,
        dropout: float,
    ):
        super(MLPMixerModel, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        channel_first = partial(nn.Conv1d, kernel_size = 1)
        channel_last = nn.Linear
        self.net = nn.Sequential(*[nn.Sequential(
            PreNormResidual(hidden_size, FeedForward(seq_len, expansion_ratio, dropout, channel_first)),
            PreNormResidual(hidden_size, FeedForward(hidden_size, shrink_ratio, dropout, channel_last)),
        ) for _ in range(depth)])
        self.layernorm = nn.LayerNorm(hidden_size)
        self.fc_out_1 = nn.Linear(hidden_size, hidden_size // 2, bias=bias)
        self.fc_out_2 = nn.Linear(hidden_size // 2, output_size, bias=bias)
        self.gelu = nn.GELU()

    def forward(self, x: Tensor):
        x = self.fc_in(x)
        x = self.gelu(x)
        x = self.net(x)
        x = self.layernorm(x)
        x = self.fc_out_1(x)
        x = self.gelu(x)
        x = self.fc_out_2(x)
        return x.squeeze(-1)


if __name__ == '__main__':
    # Sequence Model
    number_of_stocks = 4700
    seq_len = 4
    feature_size = 800
    x = torch.randn(number_of_stocks, seq_len, feature_size)
    model = MLPMixerModel(
        input_size=800, 
        hidden_size=64, 
        output_size=1, 
        seq_len=4, 
        expansion_ratio=4, 
        shrink_ratio=0.5, 
        depth=3, 
        bias=False, 
        dropout=0.4
    )
    yhat = model(x)
    print(yhat.shape)