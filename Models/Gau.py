import torch
import torch.nn as nn
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class GAUBlock(nn.Module):
    """
    Gated Attention Unit
    """
    def __init__(
        self,
        d,
        expansion_factor=2,
        s=128,
    ):
        super().__init__()
        self.d = d
        self.e = expansion_factor * d
        self.s = s

        self.norm = nn.LayerNorm(d)
        self.linear_in = nn.Linear(d, 2 * self.e + self.s)
        self.a = nn.SiLU()
        self.linear_out = nn.Linear(self.e, d)
        
        self.gamma_q = nn.Parameter(torch.ones(size=(self.s,)))
        self.gamma_k = nn.Parameter(torch.ones(size=(self.s,)))
        self.beta_q = nn.Parameter(torch.zeros(size=(self.s,)))
        self.beta_k = nn.Parameter(torch.zeros(size=(self.s,)))

    def forward(self, x):
        bs, seq_len, d = x.shape
        assert d == self.d
        uvz = self.a(self.linear_in(self.norm(x)))
        u, v, z = torch.split(uvz, [self.e, self.e, self.s], dim=-1)

        q = z * self.gamma_q + self.beta_q
        k = z * self.gamma_k + self.beta_k
        attn = torch.bmm(q, k.permute(0, 2, 1)) / seq_len
        attn = torch.relu(attn) ** 2
        out = u * (attn @ v)
        return x + self.linear_out(out)

class GAU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        expansion_factor: int,
        s: int,
        num_layers: int,
        bias: bool
    ):
        super().__init__()
        self.fc_in = nn.Linear(input_size, hidden_size, bias)
        block = GAUBlock(hidden_size, expansion_factor, s)
        self.layers = _get_clones(block, num_layers)
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
    model = GAU(
        input_size=100,
        hidden_size=64,
        output_size=1,
        expansion_factor=2,
        s=128,
        num_layers=3,
        bias=False,
    )
    y = model(x)
    print(y.shape)

