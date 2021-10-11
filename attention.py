import torch
from torch import nn
from torch.nn.parameter import Parameter

class attention_layer(nn.Module):
    def __init__(self):
        super(attention_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, lastlayer, nextlayer):
        b, c, h, w = lastlayer.size()
        y = self.avg_pool(lastlayer)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return nextlayer * y.expand_as(lastlayer)
