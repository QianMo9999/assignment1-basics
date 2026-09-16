import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

def linear(
        d_in: int, 
        d_out: int, 
        weights: Float[Tensor, " d_out d_in"], 
        in_features: Float[Tensor, " ... d_in"], 
) -> Float[Tensor, " ... d_out"]:

    return in_features @ weights.T

def embedding(
        vocab_size: int, 
        d_model: int, 
        weights: Float[Tensor, " vocab_size d_model"], 
        token_ids: Int[Tensor, " ..."], 
) -> Float[Tensor, " ... d_model"]:
    return weights[token_ids]

def silu(
        in_features: Float[Tensor, " ..."],
) -> Float[Tensor, " ..."]:
    return in_features * torch.sigmoid(in_features)

def softmax(
        in_features: Float[Tensor, " ..."], 
        dim: int,
) -> Float[Tensor, " ..."]:
    max_value = torch.max(in_features, dim=dim, keepdim=True).values
    in_features = in_features - max_value
    return torch.exp(in_features) / torch.sum(torch.exp(in_features), dim=dim, keepdim=True)

def cross_entropy(
        logits: Float[Tensor, " ..."], 
        targets: Int[Tensor, " ..."],
) -> Float[Tensor, " ..."]:
    max_value = torch.max(logits, dim=-1, keepdim=True).values
    logits = logits - max_value
    log_prob = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))
    indices = torch.arange(len(targets))
    loss = -log_prob[indices, targets]
    return loss.mean()

def rmsnorm(
    d_model: int,
    eps: float,
    weights: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    rms = torch.sqrt(torch.mean(in_features ** 2, dim=-1, keepdim=True) + eps)
    return in_features * weights / rms

def swiglu(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    w3_weight: Float[Tensor, " d_ff d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    return silu(in_features @ w1_weight.T) * (in_features @ w3_weight.T) @ w2_weight.T

def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... querys keys"] | None = None,
) -> Float[Tensor, " ... querys d_v"]:
    scores = Q @ K.transpose(-2, -1) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(mask == False, float("-inf"))
    return softmax(scores, dim=-1) @ V