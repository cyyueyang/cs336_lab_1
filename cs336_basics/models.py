import torch
import torch.nn as nn
from einops import rearrange, einsum
from typing import List, Tuple, Optional
import math

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

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True)[0]
    x_exp = torch.exp(x - x_max)
    return x_exp / x_exp.sum(dim=dim, keepdim=True)

class ScaledDotProductAttentionModule(nn.Module):
    def __init__(self, device=None, dtype=None):
        super(ScaledDotProductAttentionModule, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        d_k = query.shape[-1]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        attn_weights = torch.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, value)

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, device=None, dtype=None):
        super(CausalMultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.w_q = LinearModule(d_model, d_model, device=self.device, dtype=self.dtype)
        self.w_k = LinearModule(d_model, d_model, device=self.device, dtype=self.dtype)
        self.w_v = LinearModule(d_model, d_model, device=self.device, dtype=self.dtype)

        self.w_o = LinearModule(d_model, d_model, device=self.device, dtype=self.dtype)
        max_seq_len = 2048
        self.pos_encoder = RotaryPositionalEmbedding(theta=10000, d_k=int(self.d_k), max_seq_len=max_seq_len, device=self.device)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor :
        batch_size, seq_len, d_model = x.shape
        assert d_model == self.d_model
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        if token_positions is None:
            token_positions = rearrange(torch.arange(seq_len, dtype=torch.int32, device=self.device), "seq_len -> b... seq_len")

        token_positions = rearrange(token_positions, "... seq_len -> ... 1 seq_len")

