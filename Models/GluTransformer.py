import torch
import torch.nn as nn
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GLU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool,
    ):
        super(GLU, self).__init__()
        self.linear_u = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_v = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_o = nn.Linear(hidden_size, output_size, bias=bias)
        self.a_u = nn.Identity()
        self.a_v = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        u = self.a_u(self.linear_u(x))
        v = self.a_v(self.linear_v(x))
        return self.linear_o(u * v)

class GLUTransformerLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        bias: bool
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_size, nhead, dropout, bias)
        self.glu = GLU(hidden_size, dim_feedforward, hidden_size, bias)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout1(x))

        x = self.glu(x)
        x = self.norm2(x + self.dropout2(x))
        return x
    
class GLUTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        bias: bool,
        num_layers: int,
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        layer = GLUTransformerLayer(hidden_size, nhead, dim_feedforward, dropout, bias)
        self.layers = _get_clones(layer, num_layers)
        self.fc_out_1 = nn.Linear(hidden_size, hidden_size//2, bias)
        self.fc_out_2 = nn.Linear(hidden_size//2, output_size, bias)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        x = self.fc_in(x)
        x = x.permute(1, 0, 2)
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out_1(x[-1, :, :])
        x = self.relu(x)
        x = self.fc_out_2(x)
        return x.squeeze(-1)

if __name__ == '__main__':
    x = torch.randn(64, 12, 100)
    model = GLUTransformer(
        input_size=100,
        hidden_size=64,
        output_size=1,
        nhead=8,
        dim_feedforward=16,
        dropout=0.4,
        bias=False,
        num_layers=3,
    )
    y = model(x)
    print(y.shape)