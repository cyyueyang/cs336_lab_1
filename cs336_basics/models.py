import torch
import torch.nn as nn
from einops import rearrange, einsum

class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(LinearModule, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=self.dtype).to(self.device), requires_grad=True)
        std = 2 / (in_features + out_features)
        nn.init.trunc_normal_(self.weight.data, std=std, a=-3*std, b=3*std)

    def forward(self, x):
        return x @ self.weight.t()

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        num_embeddings: 词汇表大小
        embedding_dim: 嵌入向量的维度
        """
        super(EmbeddingModule, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding_matrix = nn.Parameter(torch.randn(num_embeddings, embedding_dim).to(self.device), requires_grad=True)
        nn.init.trunc_normal_(self.embedding_matrix.data, mean=0, std=1.0, a=-3*1.0, b=3*1.0)

    def forward(self, x):
        return self.embedding_matrix[x]

class RmsNormModule(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None):
        super(RmsNormModule, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model, dtype=self.dtype, device=self.device).to(self.device), requires_grad=True)
    def forward(self, x):
        return (self.alpha * x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)).to(x.dtype)

def Swish(x):
    return x * torch.sigmoid(x)

class GLUModule(nn.Module):
    def __init__(self, dim):
        super(GLUModule, self).__init__()
        self.w1 = LinearModule(dim, dim, device=torch.device("cpu"))
        self.w2 = LinearModule(dim, dim, device=torch.device("cpu"))
    def forward(self, x):
        return torch.sigmoid(self.w1(x)) * self.w2(x)

class SwiGLUModule(nn.Module):
    def __init__(self, dim, device=None):
        super(SwiGLUModule, self).__init__()
        self.device  = device if device is not None else torch.device("cpu")
        self.w1 = LinearModule(dim, dim, device=self.device)
        self.w2 = LinearModule(dim, dim, device=self.device)

    def forward(self, x):
        return Swish
