import json
import torch
import torch.nn as nn
from einops import rearrange, einsum
from typing import List, Tuple, Optional
import math
import os

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
            scores = scores.masked_fill(mask == 0, float("-inf"))
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
        self.scale_dot_product = ScaledDotProductAttentionModule(device=self.device, dtype=self.dtype)
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
            token_positions = rearrange(torch.arange(seq_len, dtype=torch.float32, device=self.device), "seq_len -> b seq_len", b=batch_size)

        token_positions = rearrange(token_positions, "... seq_len -> ... 1 seq_len")
        Q_pos = self.pos_encoder(Q, token_positions)
        K_pos = self.pos_encoder(K, token_positions)
        attn_output = self.scale_dot_product(Q_pos, K_pos, V, mask=self.mask)
        attn_output = rearrange(attn_output, "b heads, seq_len, d_k -> b seq_len (heads d_k)").contiguous()
        output = self.w_o(attn_output)
        return output

class TranformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, device=None, dtype=None):
        super(TranformerBlock, self).__init__()
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads
        max_seq_len = 2048
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device=self.device, dtype=self.dtype)
        self.ffn = SwiGLUModule(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.rmsnorm1 = RmsNormModule(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.rmsnorm2 = RmsNormModule(d_model, d_ff, device=self.device, dtype=self.dtype)

    def forward(self, x):
        attn_output = self.attn(self.rmsnorm1(x))
        x = x + attn_output
        ffn_output = self.ffn(self.rmsnorm2(x))
        x = x + ffn_output
        return x

class BasicTransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, d_ff, num_heads, device=None, dtype=None):
        super(BasicTransformerLM, self).__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_layers = num_layers
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_heads = num_heads

        assert d_model % num_heads == 0

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype if dtype is not None else torch.float32

        self.token_embedding = EmbeddingModule(self.vocab_size, self.d_model, device=self.device, dtype=self.dtype)
        self.layers = nn.ModuleList(
            [
                TranformerBlock(self.d_model, self.d_ff, self.num_heads, self.device, self.dtype)
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = RmsNormModule(d_model, d_ff, device=self.device, dtype=self.dtype)
        self.lm_head = LinearModule(d_model, vocab_size, device=self.device, dtype=self.dtype)

    def get_num_params(self, non_embedding=True):
        num_params = [p.numel() for p in self.parameters()]
        if non_embedding:
            num_params -= self.lm_head.weight.numel()
        return num_params


    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        x = self.token_embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = self.lm_head(x)
        return x

    @torch.no_grad()
    def generate(self,
                 x: torch.Tensor,
                 max_new_tokens: int,
                 temperature: float = 1.0,
                 top_k: int | None = None,
                 eos_token_id: int | None = None):

        if x.dim() == 1:
            x = x.unsqueeze(0)

        original_seq_len = x.size(-1)

        for _ in range(max_new_tokens):
            # 截断 保留最大的上下文长度
            x = x[:, -self.context_length:] if x.size(1) > self.context_length else x
            # [batch, seq_len, vocab_size]
            logits = self.forward(x)

            next_token_logits = logits[:, -1]
            temperature_scaled_next_token_logits = next_token_logits / temperature

            if top_k is not None:
                topk_values, _ = torch.topk(
                    temperature_scaled_next_token_logits,
                    min(top_k, next_token_logits.size(-1)),
                    sorted=True
                )

                threshold = topk_values[:, -1]
                topk_mask = temperature_scaled_next_token_logits < threshold
                temperature_scaled_next_token_logits.masked_fill_(topk_mask, float("-inf"))

            next_token_probabilities = softmax(temperature_scaled_next_token_logits, dim=-1)
            next_token_id = torch.multinomial(next_token_probabilities, 1)
            if eos_token_id is not None and next_token_id.item() == eos_token_id:
                break

            x = torch.cat([x, next_token_id], dim=-1)
        new_token_ids = x[: original_seq_len:]
        return new_token_ids
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        config_path = os.path.join(pretrained_model_path, "model_config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
            model = cls(**config)
            weights_path = os.path.join(pretrained_model_path, "model.pt")
            state_dict = torch.load(weights_path)

            unwanted_prefix = "_orig_mod."

            for key, value in list(state_dict.items()):
                if key.startswith(unwanted_prefix):
                    state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

                model.load_state_dict(state_dict)
            return model







