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


class PackedSparseMoE(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.gate = mlp.gate
        self.shared_experts = mlp.shared_experts
        self.act_fn = mlp.experts[0].act_fn
        self.expert_count = len(mlp.experts)
        self.has_shared = getattr(mlp.config, "n_shared_experts", None) is not None
        self.register_buffer("gate_weights", torch.stack([expert.gate_proj.weight.detach() for expert in mlp.experts], dim=0))
        self.register_buffer("up_weights", torch.stack([expert.up_proj.weight.detach() for expert in mlp.experts], dim=0))
        self.register_buffer("down_weights", torch.stack([expert.down_proj.weight.detach() for expert in mlp.experts], dim=0))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        topk_idx, topk_weight, _ = self.gate(hidden_states)
        flat_topk_idx = topk_idx.reshape(flat.shape[0], -1)
        flat_topk_weight = topk_weight.reshape(flat.shape[0], -1).to(flat.dtype)

        flat_by_expert = flat.unsqueeze(0).expand(self.expert_count, -1, -1).transpose(1, 2)
        gate_proj = torch.bmm(self.gate_weights, flat_by_expert).permute(2, 0, 1)
        up_proj = torch.bmm(self.up_weights, flat_by_expert).permute(2, 0, 1)
        expert_hidden = self.act_fn(gate_proj) * up_proj
        expert_hidden_by_expert = expert_hidden.permute(1, 2, 0)
        expert_out = torch.bmm(self.down_weights, expert_hidden_by_expert).permute(2, 0, 1)
        route_mask = F.one_hot(flat_topk_idx, num_classes=self.expert_count).to(flat.dtype)
        route_weight = route_mask.mul(flat_topk_weight.unsqueeze(-1)).sum(dim=1)
        routed = expert_out.mul(route_weight.unsqueeze(-1)).sum(dim=1).reshape(orig_shape)
        if self.has_shared:
            routed = routed + self.shared_experts(identity)
        return routed


class PackedHotGatherMoE(nn.Module):
    def __init__(self, mlp: nn.Module, expert_ids: list[int]):
        super().__init__()
        if not expert_ids:
            raise ValueError("expert_ids must not be empty")
        self.gate = mlp.gate
        self.shared_experts = mlp.shared_experts
        self.act_fn = mlp.experts[int(expert_ids[0])].act_fn
        self.expert_ids = [int(expert_id) for expert_id in expert_ids]
        self.expert_count = len(self.expert_ids)
        self.has_shared = getattr(mlp.config, "n_shared_experts", None) is not None
        self.register_buffer("expert_ids_tensor", torch.tensor(self.expert_ids, dtype=torch.long))
        self.register_buffer(
            "gate_weights",
            torch.stack([mlp.experts[expert_id].gate_proj.weight.detach() for expert_id in self.expert_ids], dim=0),
        )
        self.register_buffer(
            "up_weights",
            torch.stack([mlp.experts[expert_id].up_proj.weight.detach() for expert_id in self.expert_ids], dim=0),
        )
        self.register_buffer(
            "down_weights",
            torch.stack([mlp.experts[expert_id].down_proj.weight.detach() for expert_id in self.expert_ids], dim=0),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        identity = hidden_states
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        topk_idx, topk_weight, _ = self.gate(hidden_states)
        flat_topk_idx = topk_idx.reshape(flat.shape[0], -1)
        flat_topk_weight = topk_weight.reshape(flat.shape[0], -1).to(flat.dtype)

        matches = flat_topk_idx.unsqueeze(-1) == self.expert_ids_tensor.reshape(1, 1, self.expert_count)
        local_idx = matches.to(torch.long).argmax(dim=-1)
        in_pack = matches.any(dim=-1).to(flat.dtype)
        weights = flat_topk_weight * in_pack

        selected_gate = self.gate_weights[local_idx]
        selected_up = self.up_weights[local_idx]
        selected_down = self.down_weights[local_idx]
        token = flat.unsqueeze(1).unsqueeze(-1).expand(-1, flat_topk_idx.shape[1], -1, -1)

        gate_proj = torch.matmul(selected_gate, token).squeeze(-1)
        up_proj = torch.matmul(selected_up, token).squeeze(-1)
        expert_hidden = self.act_fn(gate_proj) * up_proj
        expert_out = torch.matmul(selected_down, expert_hidden.unsqueeze(-1)).squeeze(-1)
        routed = expert_out.mul(weights.unsqueeze(-1)).sum(dim=1).reshape(orig_shape)
        if self.has_shared:
            routed = routed + self.shared_experts(identity)
        return routed


class DecodeMoEFusedLayer(nn.Module):
    """One-token MoE decode layer with routed experts packed into one graph."""

    def __init__(self, layer: nn.Module, config):
        super().__init__()
        if not hasattr(layer.mlp, "gate") or not hasattr(layer.mlp, "experts"):
            raise ValueError(f"{type(layer.mlp).__name__} is not a routed MoE layer")
        self.layer = layer
        self.config = config
        self.packed_moe = PackedSparseMoE(layer.mlp)

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
        hidden_states = residual + self.packed_moe(hidden_states)
        return hidden_states, key_new, value_new


class DecodeMoEHotGatherFusedLayer(nn.Module):
    """One-token MoE decode layer with hot top-k gather and residual in one graph."""

    def __init__(self, layer: nn.Module, config, expert_ids: list[int]):
        super().__init__()
        if not hasattr(layer.mlp, "gate") or not hasattr(layer.mlp, "experts"):
            raise ValueError(f"{type(layer.mlp).__name__} is not a routed MoE layer")
        self.layer = layer
        self.config = config
        self.hot_moe = PackedHotGatherMoE(layer.mlp, expert_ids)

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
        hidden_states = residual + self.hot_moe(hidden_states)
        return hidden_states, key_new, value_new


class DecodeHotGatherBlock(nn.Module):
    """A one-token decode block that fuses several consecutive layers."""

    def __init__(self, layers: list[nn.Module], config, layer_ids: list[int], hot_experts_by_layer: dict[int, list[int]]):
        super().__init__()
        if len(layers) != len(layer_ids):
            raise ValueError("layers and layer_ids must have the same length")
        self.layers = nn.ModuleList(layers)
        self.layer_ids = [int(layer_id) for layer_id in layer_ids]
        self.config = config
        self.hot_moes = nn.ModuleDict()
        for local_index, (layer_id, layer) in enumerate(zip(self.layer_ids, self.layers)):
            if hasattr(layer.mlp, "gate"):
                expert_ids = hot_experts_by_layer.get(int(layer_id))
                if not expert_ids:
                    raise ValueError(f"Missing hot experts for layer {layer_id}")
                self.hot_moes[str(local_index)] = PackedHotGatherMoE(layer.mlp, expert_ids)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_kv: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        if len(past_kv) != len(self.layers) * 2:
            raise ValueError("Expected one past key and one past value for each block layer")
        new_keys: list[torch.Tensor] = []
        new_values: list[torch.Tensor] = []
        for local_index, layer in enumerate(self.layers):
            past_key = past_kv[local_index * 2]
            past_value = past_kv[local_index * 2 + 1]
            residual = hidden_states
            normed = layer.input_layernorm(hidden_states)
            attn = layer.self_attn
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
            hidden_states = layer.post_attention_layernorm(hidden_states)
            if hasattr(layer.mlp, "gate"):
                hidden_states = residual + self.hot_moes[str(local_index)](hidden_states)
            else:
                hidden_states = residual + layer.mlp(hidden_states)
            new_keys.append(key_new)
            new_values.append(value_new)

        outputs: list[torch.Tensor] = [hidden_states]
        for key, value in zip(new_keys, new_values):
            outputs.extend([key, value])
        return tuple(outputs)


class FinalNormHead(nn.Module):
    def __init__(self, norm: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden_states)).float()


class FinalNormTopK(nn.Module):
    def __init__(self, norm: nn.Module, lm_head: nn.Module, k: int):
        super().__init__()
        if k < 1:
            raise ValueError("k must be >= 1")
        self.norm = norm
        self.lm_head = lm_head
        self.k = int(k)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.lm_head(self.norm(hidden_states)).float()
        values, indices = torch.topk(logits, k=self.k, dim=-1)
        return values, indices


class FinalNormArgmax(nn.Module):
    def __init__(self, norm: nn.Module, lm_head: nn.Module):
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(self.norm(hidden_states)).float()
        return torch.argmax(logits, dim=-1)
