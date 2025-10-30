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

def silu(x):
    return x * torch.sigmoid(x)

class SwiGLUModule(nn.Module):
    def __init__(self, d_model, d_ff, device=None, dtype=None):
        super(SwiGLUModule, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.d_model = d_model
        self.d_ff = d_ff

        self.linear1 = LinearModule(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.linear2 = LinearModule(d_ff, d_model, device=self.device, dtype=self.dtype)
        self.linear3 = LinearModule(d_model, d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x):
        return self.linear2(silu(self.linear1(x)) * self.linear3(x))

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device if device is not None else torch.device("cpu")
        self.t = torch.arange(max_seq_len, dtype=torch.float32).to(self.device)
        self.freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_k, 2)[0: self.d_k // 2] / self.d_k))
        self.freqs = torch.outer(self.t, self.freqs)

        self.register_buffer("cos_cache", self.freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", self.freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos_emb = self.cos_cache[token_positions]
        sin_emb = self.sin_cache[token_positions]
        cos_emb = cos_emb.repeat_interleave(repeats=2, dim=-1)
        sin_emb = sin_emb.repeat_interleave(repeats=2, dim=-1)
        x_shift = self._shift(x)
        return cos_emb * x + sin_emb * x_shift

    def _shift(self, x: torch.Tensor) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_shifted = torch.stack([-x_odd, x_even], dim=-1)
        return x_shifted.flatten(-2)

