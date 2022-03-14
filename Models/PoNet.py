import torch
import torch.nn as nn

# https://arxiv.org/abs/2110.02442

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
