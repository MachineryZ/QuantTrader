import torch
import copy
import math
import torch.nn as nn

from transformer import PositionEncoding



def _get_clones(modules: nn.Module, N: int) -> nn.Module:
    return nn.ModuleList([copy.deepcopy(modules) for i in range(N)])


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
    
    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class LocalFormerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        hidden_size: int,
    ):
        super(LocalFormerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.conv = _get_clones(nn.Conv1d(hidden_size, hidden_size, 3, 1, 1), num_layers)
        self.num_layers = num_layers
    
    def forward(self, src, mask):
        output = src
        out =src

        for i, mod in enumerate(self.layers):
            # [seq, N, F] -> [N, seq, F] -> [N, F, seq]
            out = output.transpose(1, 0).transpose(2, 1)
            out = self.conv[i](out).transpose(2, 1).transpose(1, 0)
            output = mod(output + out, src_mask=mask)
        return output + out



class LocalFormerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ):
        super(LocalFormerModel, self).__init__()
        self.rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
            dropout=dropout,
        )
        self.feature_layer = nn.Linear(input_size, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dropout=dropout,
        )
        self.transformer_encoder = LocalFormerEncoder(self.encoder_layer, num_layers=num_layers, hidden_size=hidden_size)
        self.decoder_layer = nn.Linear(hidden_size, 1)
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # src [N, seq, F]
        src = self.feature_layer(x) # src [N, seq, F']
        src = src.transpose(0, 1)
        mask = None
        output = self.transformer_encoder(src, mask)
        output, _ = self.rnn(output)
        output = self.decoder_layer(output.transpose(1, 0)[:, -1, :])

        return output.squeeze()

if __name__ == '__main__':
    model = LocalFormerModel(
        input_size=962,
        hidden_size=64,
        output_size=1,
        nhead=4,
        num_layers=3,
        dropout=0.5,
    ).cuda()
    x = torch.randn(4000, 4, 962).cuda()
    y = model(x)
    print(y.shape)
    pass

