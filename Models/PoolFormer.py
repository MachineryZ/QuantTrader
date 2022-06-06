import torch
import torch.nn as nn
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_class(activation):
    if activation == "relu":
        return nn.ReLU
    elif activation == "gelu":
        return nn.GELU
    else:
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


class Pooling(nn.Module):
    def __init__(
        self,
        pool_size: int,
    ):
        super().__init__()
        self.pool = nn.AvgPool1d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
    
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1)
        x = self.pool(x) - x
        x = x.permute(0, 2, 1)
        return x

class PoolFormerLayer(nn.Module):
    def __init__(
        self,
        d_model:int,
        pool_size: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        bias: bool,
    ):
        super().__init__()
        self.seq_mixer = Pooling(pool_size)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias),
            _get_activation_class(activation)(),
            nn.Linear(dim_feedforward, d_model, bias),
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.dropout1(self.seq_mixer(self.norm1(x)))
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x


class PoolFormer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pool_size: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        bias: bool,
        num_layers: int,
    ):
        super().__init__()
        assert pool_size % 2 == 1, "pool_size must be odd number"
        self.fc_in = nn.Linear(input_size, hidden_size, bias)
        pfm_layer = PoolFormerLayer(
            d_model=hidden_size,
            pool_size=pool_size,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            bias=bias,
        )
        self.layers = _get_clones(pfm_layer, num_layers)
        self.fc_out_1 = nn.Linear(hidden_size, hidden_size//2, bias)
        self.fc_out_2 = nn.Linear(hidden_size//2, output_size, bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        x = x[:, -1, :]
        x = self.fc_out_1(x)
        x = self.relu(x)
        x = self.fc_out_2(x)
        return x.squeeze(-1)

if __name__ == '__main__':
    x = torch.randn(32, 12, 100)
    model = PoolFormer(
        input_size=100,
        hidden_size=64,
        output_size=1,
        pool_size=3,
        dim_feedforward=32,
        dropout=0.4,
        activation="gelu",
        bias=False,
        num_layers=3,
    )
    y = model(x)
    print(y.shape)
