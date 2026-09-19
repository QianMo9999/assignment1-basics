import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math
from collections.abc import Iterable

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

def multihead_self_attention(
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " d_model d_model"],
        k_proj_weight: Float[Tensor, " d_model d_model"],
        v_proj_weight: Float[Tensor, " d_model d_model"],
        o_proj_weight: Float[Tensor, " d_model d_model"],
        in_features: Float[Tensor, " ... sequence_length d_model"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    Q = in_features @ q_proj_weight.T
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T
    d_head = d_model // num_heads
    q_prefix = Q.shape[:-1]
    q_new_shape = q_prefix + (num_heads, d_head)
    Q = Q.reshape(q_new_shape)
    k_prefix = K.shape[:-1]
    k_new_shape = k_prefix + (num_heads, d_head)
    K = K.reshape(k_new_shape)
    v_prefix = V.shape[:-1]
    v_new_shape = v_prefix + (num_heads, d_head)
    V = V.reshape(v_new_shape)
    Q = Q.transpose(-2, -3) # ... num_heads sequence_length d_head
    K = K.transpose(-2, -3) # ... num_heads sequence_length d_head
    V = V.transpose(-2, -3) # ... num_heads sequence_length d_head
    mask = torch.tril(torch.ones((Q.shape[-2], Q.shape[-2]), dtype=torch.bool))
    attention_output = scaled_dot_product_attention(Q, K, V, mask) # ... num_heads sequence_length d_head
    attention_output = attention_output.transpose(-2, -3).reshape(attention_output.transpose(-2, -3).shape[:-2] + (d_model,)) # ... sequence_length d_model

    return attention_output @ o_proj_weight.T

def rope(
    d_k: int,
    theta: float,
    max_seq_len: int,
    in_query_or_key: Float[Tensor, " ... sequence_length d_k"],
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_k"]:
    indices = torch.arange(0, d_k, 2, device=in_query_or_key.device)
    freq = 1.0 / (theta ** (indices / d_k))
    pos = token_positions.unsqueeze(-1)
    angles = pos * freq
    cos_angles = torch.cos(angles)
    sin_angles = torch.sin(angles)
    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    x_even_rotated = x_even * cos_angles - x_odd * sin_angles 
    x_odd_rotated = x_even * sin_angles + x_odd * cos_angles
    x_rotated = torch.empty_like(in_query_or_key)
    x_rotated[..., 0::2] = x_even_rotated
    x_rotated[..., 1::2] = x_odd_rotated
    return x_rotated

def multihead_self_attention_with_rope(
    d_model: int,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    q_proj_weight: Float[Tensor, " d_model d_model"],
    k_proj_weight: Float[Tensor, " d_model d_model"],
    v_proj_weight: Float[Tensor, " d_model d_model"],
    o_proj_weight: Float[Tensor, " d_model d_model"],
    in_features: Float[Tensor, " ... sequence_length d_model"],
    token_positions: Int[Tensor, " ... sequence_length"] | None = None,
) -> Float[Tensor, " ... sequence_length d_model"]:
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    Q = in_features @ q_proj_weight.T # ... sequence_length d_model
    K = in_features @ k_proj_weight.T
    V = in_features @ v_proj_weight.T
    d_head = d_model // num_heads
    prefix_shape = Q.shape[:-1] # ... sequence_length
    Q = Q.reshape(prefix_shape + (num_heads, d_head)).transpose(-2, -3) # ... num_heads sequence_length d_head
    K = K.reshape(prefix_shape + (num_heads, d_head)).transpose(-2, -3)
    V = V.reshape(prefix_shape + (num_heads, d_head)).transpose(-2, -3)
    if token_positions is not None:
        Q = rope(d_head, theta, max_seq_len, Q, token_positions)
        K = rope(d_head, theta, max_seq_len, K, token_positions)
    mask = torch.tril(torch.ones(Q.shape[-2], Q.shape[-2], dtype=torch.bool))
    scores = (Q @ K.transpose(-2, -1) / math.sqrt(d_head)).masked_fill(mask == False, float("-inf")) # ... num_heads sequence_length sequence_length
    attention_output =  torch.exp(scores) / torch.sum(torch.exp(scores), dim=-1, keepdim=True) @ V # ... num_heads sequence_length d_head
    attention_output =  attention_output.transpose(-2, -3).reshape(prefix_shape + (d_model,))

    return attention_output @ o_proj_weight.T

def transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    rmsnorm_output = rmsnorm(d_model, 1e-5, weights["ln1.weight"], in_features)
    token_positions = torch.arange(in_features.shape[1], device=in_features.device)
    attention_output = multihead_self_attention_with_rope(d_model, num_heads, max_seq_len, theta, weights["attn.q_proj.weight"], weights["attn.k_proj.weight"], weights["attn.v_proj.weight"], weights["attn.output_proj.weight"], rmsnorm_output, token_positions)
    attention_output = in_features + attention_output
    rmsnorm_output = rmsnorm(d_model, 1e-5, weights["ln2.weight"], attention_output)
    swiglu_output = swiglu(d_model, d_ff, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"], rmsnorm_output)
    return attention_output + swiglu_output

def transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    rope_theta: float,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    x = embedding(vocab_size, d_model, weights["token_embeddings.weight"], in_indices) # batch_size sequence_length d_model
    for layer in range(num_layers):
        block_weights = {}
        prefix = f"layers.{layer}."
        for key, value in weights.items():
            if key.startswith(prefix):
                key = key[len(prefix):]
                block_weights[key] = value
        x = transformer_block(d_model, num_heads, d_ff, context_length, rope_theta, block_weights, x) # batch_size sequence_length d_model
    x = rmsnorm(d_model, 1e-5, weights["ln_final.weight"], x)
    x = linear(d_model, vocab_size, weights["lm_head.weight"], x) # batch_size sequence_length vocab_size
    return x

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    total_norm = 0
    for param in parameters:
        if param.grad is not None:
            total_norm += torch.sum(param.grad.data ** 2)

    total_norm = math.sqrt(total_norm)

    if(total_norm > max_l2_norm):
        scale = max_l2_norm / (total_norm + 1e-6)
        for param in parameters:
            if param.grad is not None:
                param.grad.mul_(scale)
        
    return None

def lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it <= warmup_iters:
        return max_learning_rate * it / warmup_iters
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        factor = (1 + math.cos(math.pi * progress)) / 2
        return min_learning_rate + factor * (max_learning_rate - min_learning_rate)
    else:
        return min_learning_rate