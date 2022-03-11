import torch
import torch.nn as nn

class PoNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,

    ):
        super(PoNet, self).__init__()


    def forward(self, x):
        return x
