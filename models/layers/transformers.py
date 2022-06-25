import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch import linalg as LinearAlgebra


def dot_product_attention(query: Tensor, key: Tensor, value: Tensor, scaled: bool) -> Tensor:
    # batch_size, seq_length, num_features
    print("In Dot Product Code: \n")
    print("Query before: ", query.size())
    # print("Key before: ", key)
    temp = query.bmm(key.transpose(1, 2))
    # print("Query size: ", query)
    # print("Key size: ", key)
    # print("Attention size: ", temp.size())
    scale = 1
    if scaled:
        scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


def cosine_attention(query: Tensor, key: Tensor, value: Tensor, scaled: bool) -> Tensor:
    # batch_size x seq_length x d_k
    print("In Cosine Code: \n")
    print("Query before: ", query.size())
    # print("Key before: ", key)
    query_norm = query / LinearAlgebra.vector_norm(query, dim=-1)[:, :, None]
    key_norm = key / LinearAlgebra.vector_norm(key, dim=-1)[:, :, None]
    temp = torch.matmul(query_norm, key_norm.transpose(1, 2)) # query.bmm(key.transpose(1, 2))

    # print("Query Norm : \n" ,query_norm)
    # print("Key Norm : \n", key_norm)
    # query_key = query_norm.bmm(key_norm.transpose(1, 2)) # batch_size x seq_len x seq_len
    # print("Cosine score : \n", query_key)
    scale = 1
    if scaled:
        scale = query.size(-1) ** 0.5
    softmax = F.softmax(temp / scale, dim=-1)
    attention_output = softmax.bmm(value) 
    # print(attention_output)
    return attention_output # batch_size x seq_len x d_v

def position_encoding(seq_len: int, dim_model: int, device: torch.device = torch.device("cpu")) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float, device=device).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float, device=device).reshape(1, 1, -1)
    phase = pos / (1e4 ** (dim // dim_model))

    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> nn.Module:
    return nn.Sequential(
        nn.Linear(dim_input, dim_feedforward),
        nn.GELU(), # Based on BERT paper
        nn.Linear(dim_feedforward, dim_input),
    )


class Residual(nn.Module):
    def __init__(self, sublayer: nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = nn.LayerNorm(dimension)
        self.dropout = nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "query" tensor is given first, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[0] + self.dropout(self.sublayer(*tensors)))



class AttentionHead(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_k: int, attn_type: str, scaled: bool):
        super().__init__()
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)
        self.attn_type = attn_type
        self.scaled = scaled

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        q = self.q(query)
        k = self.k(key)
        v = self.v(value)
        # print("In Attention Code:\n")
        # print("Query  : \n", q.size())
        # print("Key : \n", k)
        # print("Value : \n", v)
        if self.attn_type == 'dot_product':
            return dot_product_attention(q, k, v, self.scaled)
        elif self.attn_type == 'cosine':
            return cosine_attention(q, k, v, self.scaled)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_q: int, dim_k: int, attn_type: str, scaled: bool):
        super().__init__()
        self.heads = nn.ModuleList(
            [AttentionHead(dim_in, dim_q, dim_k, attn_type, scaled) for _ in range(num_heads)]
        )
        self.linear = nn.Linear(num_heads * dim_k, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        # print("In Multihead Code:\n")
        # print("Query  : \n", query.size())
        # print("Key : \n", key)
        # print("Value : \n", value)
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        attn_type: str, 
        scaled: bool,
        dim_model: int = 512,
        num_heads: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        dim_q = dim_k = max(dim_model // num_heads, 1)
        self.attention = Residual(
            MultiHeadAttention(num_heads, dim_model, dim_q, dim_k, attn_type, scaled),
            dimension=dim_model,
            dropout=dropout,
        )
        self.feed_forward = Residual(
            feed_forward(dim_model, dim_feedforward),
            dimension=dim_model,
            dropout=dropout,
        )

    def forward(self, src: Tensor) -> Tensor:
        # print("In Transformer Encoder Code:\n")
        # print("Source Tensor  : \n", src.size())
        src = self.attention(src, src, src)
        # print("Attention Tensor  : \n", src.size())
        return self.feed_forward(src)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        attn_type: str, 
        scaled: bool,
        num_layers: int = 6,
        dim_model: int = 512,
        num_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(attn_type, scaled, dim_model, num_heads, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src: Tensor) -> Tensor:
        # print("Tensor input di Encoder: ", src.size())
        # seq_len, dimension = src.size(1), src.size(2)
        # src += position_encoding(seq_len, dimension)
        for layer in self.layers:
            src = layer(src)

        return src


