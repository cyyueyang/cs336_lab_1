import torch
import torch.nn as nn
from einops import rearrange, einsum

class CyyLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(CyyLinear, self).__init__()
        w = nn.Parameter(torch.randn(out_features, in_features))
        b = nn.Parameter(torch.randn(out_features))

    def forward(self, x):
        return x