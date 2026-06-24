from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    repeat_kv as llama_repeat_kv,
)


class DecodeDenseLayer(nn.Module):
    """Full one-token decode layer for non-MoE layers."""

    def __init__(self, layer: nn.Module, config):
        super().__init__()
        self.layer = layer
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)
        attn = self.layer.self_attn
        bsz, q_len, _ = normed.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = attn.head_dim

        query_states = attn.q_proj(normed).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_new = attn.k_proj(normed).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_new = attn.v_proj(normed).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = attn.rotary_emb(value_new, position_ids)
        query_states, key_new = llama_apply_rotary_pos_emb(query_states, key_new, cos, sin)

        key_states = torch.cat([past_key, key_new], dim=2)
        value_states = torch.cat([past_value, value_new], dim=2)
        key_rep = llama_repeat_kv(key_states, attn.num_key_value_groups)
        value_rep = llama_repeat_kv(value_states, attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / (head_dim**0.5)
        attn_weights = attn_weights + attention_mask[:, :, :, : key_rep.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        hidden_states = residual + attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = self.layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.layer.mlp(hidden_states)
        return hidden_states, key_new, value_new


class DecodeMoEAttentionGate(nn.Module):
    """One-token decode layer prefix for host-dispatched MoE.

    The graph runs attention, post-attention layernorm, router gate, and shared
    experts. Host code runs only the routed experts selected by `topk_idx`, then
    computes:

        next_hidden = attn_residual + shared_out + routed_expert_sum
    """

    def __init__(self, layer: nn.Module, config):
        super().__init__()
        if not hasattr(layer.mlp, "gate") or not hasattr(layer.mlp, "experts"):
            raise ValueError(f"{type(layer.mlp).__name__} is not a routed MoE layer")
        self.layer = layer
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)
        attn = self.layer.self_attn
        bsz, q_len, _ = normed.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = attn.head_dim

        query_states = attn.q_proj(normed).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_new = attn.k_proj(normed).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_new = attn.v_proj(normed).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        cos, sin = attn.rotary_emb(value_new, position_ids)
        query_states, key_new = llama_apply_rotary_pos_emb(query_states, key_new, cos, sin)

        key_states = torch.cat([past_key, key_new], dim=2)
        value_states = torch.cat([past_value, value_new], dim=2)
        key_rep = llama_repeat_kv(key_states, attn.num_key_value_groups)
        value_rep = llama_repeat_kv(value_states, attn.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / (head_dim**0.5)
        attn_weights = attn_weights + attention_mask[:, :, :, : key_rep.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        attn_residual = residual + attn.o_proj(attn_output)

        moe_input = self.layer.post_attention_layernorm(attn_residual)
        topk_idx, topk_weight, _ = self.layer.mlp.gate(moe_input)
        shared_out = self.layer.mlp.shared_experts(moe_input)
        return attn_residual, moe_input, topk_idx, topk_weight, shared_out, key_new, value_new


class AddMoEResidual(nn.Module):
    def forward(
        self,
        attn_residual: torch.Tensor,
        shared_out: torch.Tensor,
        routed_out: torch.Tensor,
    ) -> torch.Tensor:
        return attn_residual + shared_out + routed_out


class FinalNormHead(nn.Module):
    def __init__(self, norm: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states)).float()
