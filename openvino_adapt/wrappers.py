from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb as llama_apply_rotary_pos_emb,
    repeat_kv as llama_repeat_kv,
)


class ProjectorOnly(nn.Module):
    """Export boundary for Unlimited-OCR's linear 2048 -> 1280 projector."""

    def __init__(self, projector: nn.Module):
        super().__init__()
        self.projector = projector

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.projector(image_features)


class TextEmbeddings(nn.Module):
    """Export boundary for decoder token embeddings."""

    def __init__(self, embed_tokens: nn.Module):
        super().__init__()
        self.embed_tokens = embed_tokens

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)


class VisionTokenExtractor(nn.Module):
    """Build base-mode visual tokens for one or more 1024x1024 images.

    This mirrors the no-crop branch in UnlimitedOCRModel.forward used by
    infer_multi(): SAM features and CLIP-style features are concatenated,
    projected to decoder hidden size, row separators are inserted, and a final
    view separator is appended per image.
    """

    def __init__(self, unlimited_model: nn.Module):
        super().__init__()
        self.sam_model = unlimited_model.sam_model
        self.vision_model = unlimited_model.vision_model
        self.projector = unlimited_model.projector
        self.image_newline = unlimited_model.image_newline
        self.view_seperator = unlimited_model.view_seperator

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        sam_features = self.sam_model(images)
        clip_features = self.vision_model(images, sam_features)
        merged = torch.cat(
            (clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)),
            dim=-1,
        )
        projected = self.projector(merged)

        batch, hw, hidden = projected.shape
        grid = int(hw**0.5)
        projected = projected.reshape(batch, grid, grid, hidden)

        newline = self.image_newline.reshape(1, 1, 1, hidden).expand(batch, grid, 1, hidden)
        projected = torch.cat([projected, newline], dim=2)
        projected = projected.reshape(batch, grid * (grid + 1), hidden)

        view_sep = self.view_seperator.reshape(1, 1, hidden).expand(batch, 1, hidden)
        return torch.cat([projected, view_sep], dim=1)


class DecoderNoCache(nn.Module):
    """A minimal decoder export boundary without KV cache.

    This is useful for early conversion tests only. The production adaptation
    needs explicit prefill/decode graphs with host-owned KV buffers.
    """

    def __init__(self, causal_lm: nn.Module):
        super().__init__()
        self.model = causal_lm.model
        self.lm_head = causal_lm.lm_head

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            return_dict=True,
        )
        return self.lm_head(outputs.last_hidden_state).float()


class DecoderPrefillWithKV(nn.Module):
    """Prefill graph boundary with explicit K/V tensor outputs."""

    def __init__(self, causal_lm: nn.Module):
        super().__init__()
        self.layers_module = causal_lm.model.layers
        self.norm = causal_lm.model.norm
        self.lm_head = causal_lm.lm_head
        self.config = causal_lm.config
        self.moe_impl = getattr(causal_lm.config, "_openvino_moe_impl", "dense")

    @staticmethod
    def _dense_moe_forward(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hasattr(mlp, "experts") or not hasattr(mlp, "gate"):
            return mlp(hidden_states)

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _ = mlp.gate(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.reshape(flat.shape[0], -1)
        flat_topk_weight = topk_weight.reshape(flat.shape[0], -1).to(flat.dtype)

        y = torch.zeros_like(flat)
        for expert_id, expert in enumerate(mlp.experts):
            if expert is None:
                continue
            weight = (flat_topk_idx == expert_id).to(flat.dtype).mul(flat_topk_weight).sum(dim=1, keepdim=True)
            y = y + expert(flat).mul(weight)

        y = y.reshape(*orig_shape)
        if getattr(mlp.config, "n_shared_experts", None) is not None:
            y = y + mlp.shared_experts(identity)
        return y

    @staticmethod
    def _sparse_moe_forward(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if not hasattr(mlp, "experts") or not hasattr(mlp, "gate"):
            return mlp(hidden_states)

        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight, _ = mlp.gate(hidden_states)
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.reshape(flat.shape[0], -1)
        flat_topk_weight = topk_weight.reshape(flat.shape[0], -1).to(flat.dtype)
        top_k = flat_topk_idx.shape[1]

        gate_weights = torch.stack([expert.gate_proj.weight for expert in mlp.experts], dim=0)
        up_weights = torch.stack([expert.up_proj.weight for expert in mlp.experts], dim=0)
        down_weights = torch.stack([expert.down_proj.weight for expert in mlp.experts], dim=0)

        selected = flat_topk_idx.reshape(-1)
        expanded = flat.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, flat.shape[-1])
        gate_proj = torch.bmm(gate_weights[selected], expanded.unsqueeze(-1)).squeeze(-1)
        up_proj = torch.bmm(up_weights[selected], expanded.unsqueeze(-1)).squeeze(-1)
        expert_hidden = mlp.experts[0].act_fn(gate_proj) * up_proj
        expert_out = torch.bmm(down_weights[selected], expert_hidden.unsqueeze(-1)).squeeze(-1)

        y = expert_out.reshape(flat.shape[0], top_k, -1).mul(flat_topk_weight.unsqueeze(-1)).sum(dim=1)
        y = y.reshape(*orig_shape)
        if getattr(mlp.config, "n_shared_experts", None) is not None:
            y = y + mlp.shared_experts(identity)
        return y

    def _mlp_forward(self, mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.moe_impl == "sparse":
            return self._sparse_moe_forward(mlp, hidden_states)
        return self._dense_moe_forward(mlp, hidden_states)

    def _layer_prefill(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        attn = layer.self_attn
        bsz, q_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = attn.head_dim
        num_kv_groups = attn.num_key_value_groups

        query_states = attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_states = attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(value_states, position_ids)
        query_states, key_states = llama_apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_rep = llama_repeat_kv(key_states, num_kv_groups)
        value_rep = llama_repeat_kv(value_states, num_kv_groups)
        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / (head_dim**0.5)
        attn_weights = attn_weights + attention_mask[:, :, :q_len, :q_len]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        hidden_states = residual + attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + self._mlp_forward(layer.mlp, hidden_states)
        return hidden_states, key_states, value_states

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = inputs_embeds
        flat: list[torch.Tensor] = []
        for layer in self.layers_module:
            hidden_states, key_states, value_states = self._layer_prefill(
                layer,
                hidden_states,
                position_ids,
                attention_mask,
            )
            flat.extend([key_states, value_states])
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        flat.insert(0, logits)
        return tuple(flat)


class DecoderDecodeOneStep(nn.Module):
    """One-token explicit-KV decoder.

    Inputs are `input_ids`, `position_ids`, an additive attention mask with shape
    `[1, 1, 1, past_len + 1]`, then 24 tensors:
    `past_key_0, past_value_0, ..., past_key_11, past_value_11`.

    Outputs are `logits`, then the new one-token K/V tensors for each layer.
    Host code owns cache append/trimming.
    """

    def __init__(self, causal_lm: nn.Module):
        super().__init__()
        self.embed_tokens = causal_lm.model.embed_tokens
        self.layers = causal_lm.model.layers
        self.norm = causal_lm.model.norm
        self.lm_head = causal_lm.lm_head
        self.config = causal_lm.config
        self.moe_impl = getattr(causal_lm.config, "_openvino_moe_impl", "dense")

    def _mlp_forward(self, mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.moe_impl == "sparse":
            return DecoderPrefillWithKV._sparse_moe_forward(mlp, hidden_states)
        return DecoderPrefillWithKV._dense_moe_forward(mlp, hidden_states)

    def _layer_forward(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)

        attn = layer.self_attn
        bsz, q_len, _ = hidden_states.shape
        num_heads = self.config.num_attention_heads
        num_kv_heads = self.config.num_key_value_heads
        head_dim = attn.head_dim
        num_kv_groups = attn.num_key_value_groups

        query_states = attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        key_new = attn.k_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
        value_new = attn.v_proj(hidden_states).view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)

        cos, sin = attn.rotary_emb(value_new, position_ids)
        query_states, key_new = llama_apply_rotary_pos_emb(query_states, key_new, cos, sin)

        key_states = torch.cat([past_key, key_new], dim=2)
        value_states = torch.cat([past_value, value_new], dim=2)
        key_rep = llama_repeat_kv(key_states, num_kv_groups)
        value_rep = llama_repeat_kv(value_states, num_kv_groups)

        attn_weights = torch.matmul(query_states, key_rep.transpose(2, 3)) / (head_dim**0.5)
        attn_weights = attn_weights + attention_mask[:, :, :, : key_rep.shape[-2]]
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, q_len, -1)
        hidden_states = residual + attn.o_proj(attn_output)

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + self._mlp_forward(layer.mlp, hidden_states)
        return hidden_states, key_new, value_new

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *past_key_values: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        hidden_states = self.embed_tokens(input_ids)
        new_cache: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(self.layers):
            past_key = past_key_values[layer_idx * 2]
            past_value = past_key_values[layer_idx * 2 + 1]
            hidden_states, key_new, value_new = self._layer_forward(
                layer,
                hidden_states,
                position_ids,
                attention_mask,
                past_key,
                past_value,
            )
            new_cache.extend([key_new, value_new])
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states).float()
        return tuple([logits] + new_cache)
