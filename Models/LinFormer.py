import torch
import torch.nn as nn
import copy

"""
Normal Attention:

QKV complexity:
QK: O(n^2 d)

But we can let Q in R(n * d) -> Q in R(k * d)
so that we will have 
"""

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class _3d_Linear_dim0(nn.Linear):
    def __init__(
        self,
        input_seq_len: int,
        output_seq_len: int,
        bias: bool,
    ):
        super().__init__(input_seq_len, output_seq_len, bias)
    def forward(self, x: torch.Tensor):
        xw = torch.einsum('ibd, oi -> obd', x, self.weight)
        if self.bias is None:
            return xw
        else:
            b = self.bias[:, None, None]
            return xw + b

class LinAttention(nn.Module):
    def __init__(
        self,
        seq_len: int,
        seq_len_k: int,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        bias: bool,
    ):
        super().__init__()
        self.E = _3d_Linear_dim0(seq_len, seq_len_k, bias)
        self.F = _3d_Linear_dim0(seq_len, seq_len_k, bias)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout, bias)

    def forward(self, query, key, value):
        key_new = self.E(key)
        value_new = self.F(value)
        attn_output, attn_weight = self.mha(query, key_new, value_new)
        return attn_output

class LinFormerLayer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        seq_len_k: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        bias: bool
    ):
        super().__init__()
        self.self_attn = LinAttention(seq_len, seq_len_k, d_model, nhead, dropout, bias)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, bias),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, bias),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.self_attn(x, x, x)
        x = self.norm1(x + self.dropout1(x))
        x = self.ff(x)
        x = self.norm2(x + self.dropout2(x))
        return x

class LinFormer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        seq_len: int,
        seq_len_k: int,
        dim_feedforward: int,
        dropout: float,
        bias: bool,
        nhead: int,
        num_layers: int,
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.fc_out_1 = nn.Linear(hidden_size, hidden_size//2, bias)
        self.fc_out_2 = nn.Linear(hidden_size//2, output_size, bias)
        self.relu = nn.ReLU()
        layer = LinFormerLayer(
            seq_len=seq_len,
            seq_len_k=seq_len_k,
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
        )
        self.layers = _get_clones(layer, num_layers)

    def forward(self, x):
        x = self.fc_in(x)
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = x[-1, :, :]
        x = self.fc_out_1(x)
        x = self.relu(x)
        x = self.fc_out_2(x)
        return x.squeeze(-1)

if __name__ == '__main__':
    x = torch.randn(32, 12, 100)
    model = LinFormer(
        input_size=100,
        hidden_size=64,
        output_size=1,
        seq_len=12,
        seq_len_k=6,
        dim_feedforward=50,
        dropout=0.4,
        bias=False,
        nhead=4,
        num_layers=4,
    )
    y = model(x)
    print(y.shape)
    