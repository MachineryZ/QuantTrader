import torch
import torch.nn as nn

<<<<<<< HEAD
# https://arxiv.org/abs/2110.02442
=======
class GlobalAttention(nn.Module):
    def __init__(
        self,

    ):
        super(GlobalAttention, self).__init__()

    def forward(self, x):
        return

class SegmentMaxPooling(nn.Module):
    def __init__(
        self,
    ):
        super(SegmentMaxPooling, self).__init__()

    def forward(self, x):
        return
    
class LocalMaxPooling(nn.Module):
    def __init__(
        self,
    ):
        super(LocalMaxPooling, self).__init__()

    def forward(self, x):
        return
>>>>>>> 07011483f36fb1ab42ec7d8b9d466af098182127

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

if __name__ == '__main__':
    pass