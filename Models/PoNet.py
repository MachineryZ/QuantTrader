import torch
# import transformers
import torch.nn as nn
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PoNetBlock(nn.Module):
    def __init__(
        self,
        d: int,
        local_k: int,
        segment_k: int,
    ):
        super(PoNetBlock, self).__init__()
        self.d = d
        self.local_k = local_k
        self.segment_k = segment_k
        self.fc = nn.Linear(d, 6*d)
        self.local_pooling = nn.MaxPool1d(kernel_size=local_k, stride=1, padding=local_k//2)
        self.segment_pooling=nn.MaxPool1d(kernel_size=segment_k, stride=segment_k)

    def global_attention(self, q, k, v):
        qk = torch.einsum('bd, bld -> bl', q, k)
        attn = torch.softmax(qk,  dim=1)
        out = torch.einsum('bl, bld -> bd', attn, v)
        return out

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        hQ, hK, hV, hs, hl, ho = torch.split(h, [self.d] * 6, dim=-1)

        # global
        g = torch.mean(hQ, dim=1)
        g_prime = self.global_attention(g, hK, hV)
        g_out = g_prime.unsqueeze(1) * ho

        # segment
        s = self.segment_pooling(hs.permute(0, 2, 1)).repeat_interleave(self.segment_k, -1).permute(0, 2, 1)
        s_out = s * ho

        # local
        l_out = self.local_pooling(hl.permute(0, 2, 1)).permute(0, 2, 1)
        return g_out + s_out + l_out

class PoNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        local_k: int,
        segment_k: int,
        num_layers: int,
        bias: bool,
    ):
        super(PoNet, self).__init__()
        assert local_k % 2 == 1, "local_k must be odd number"
        self.fc_in = nn.Linear(input_size, hidden_size, bias=bias)
        self.fc_out_1 = nn.Linear(hidden_size, hidden_size // 2, bias=bias)
        self.fc_out_2 = nn.Linear(hidden_size // 2, output_size, bias=bias)
        self.relu = nn.ReLU()
        block = PoNetBlock(hidden_size, local_k, segment_k)
        self.layers = _get_clones(block, num_layers)
        self.segment_k = segment_k

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, input_size = x.shape
        assert seq_len % self.segment_k == 0, "seq_len should be dividable by segment_k"
        
        x = self.fc_in(x)
        for layer in self.layers:
            x = layer(x)
        out = x[:, -1, :]
        out = self.fc_out_1(out)
        out = self.relu(out)
        out = self.fc_out_2(out)
        out = out.squeeze(-1)
        return out

if __name__ == '__main__':
    x = torch.randn(400, 12, 50)
    model = PoNet(
        input_size=50,
        hidden_size=128,
        output_size=1,
        local_k=3,
        segment_k=3,
        num_layers=3,
        bias=False,
    )
    y = model(x)
    print(y.shape)