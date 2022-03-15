import torch
import math
import torch.nn as nn
from modules.base_transformer_encoder import BaseTransformerEncoder

"""
Transformer Model
arxiv:
Attention is All your need
https://arxiv.org/abs/1706.03762


"""

class PositionEncoding(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        max_len: int = 5000,
    ):
        super(PositionEncoding, self).__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0,1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        bias: bool,
        nhead: int,
        num_layers: int,
        dropout: float,
        dim_feedforward: int,
        negative_slope: float,
        use_posenc: bool,
    ):
        super(TransformerModel, self).__init__()
        self.fc_in = nn.Linear(input_size, hidden_size, bias)
        self.use_posenc = use_posenc
        self.pos_encoder = PositionEncoding(hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, 
            nhead=nhead, 
            dropout=dropout,
            dim_feedforward=dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer_1 = nn.Linear(hidden_size, hidden_size // 2, bias)
        self.decoder_layer_2 = nn.Linear(hidden_size // 2, output_size, bias)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.input_size = input_size

    def forward(self, x):
        x = self.fc_in(x)
        x = x.transpose(1,0) # batch first transformation
        mask = None
        if self.use_posenc is True:
            x = self.pos_encoder(x)
        output = self.transformer_encoder(x, mask)
        output = self.decoder_layer_1(output.transpose(1,0)[:, -1, :])
        output = self.leakyrelu(output)
        output = self.decoder_layer_2(output).squeeze(-1)
        return output


if __name__ == '__main__':
    x = torch.randn(500, 8, 722)
    model = TransformerModel(
        input_size=722,
        hidden_size=48,
        output_size=1,
        bias=False,
        nhead=8,
        num_layers=3,
        dropout=0.2,
        dim_feedforward=512,
        negative_slope=0.4,
        use_posenc=True,
    )
    y = model(x)
    print(y.shape)
