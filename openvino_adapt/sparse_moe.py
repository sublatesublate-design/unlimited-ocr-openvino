from __future__ import annotations

import torch
from torch import nn


def host_sparse_moe_forward(mlp: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Run routed MoE by dispatching only selected experts in Python.

    This mirrors DeepseekV2MoE.moe_infer(), but keeps the implementation small
    and explicit for OpenVINO adapter experiments. It is intended for decode
    shapes first, where the token count is tiny and Python dispatch overhead is
    acceptable compared with evaluating all 64 routed experts.
    """

    if not hasattr(mlp, "experts") or not hasattr(mlp, "gate"):
        return mlp(hidden_states)

    identity = hidden_states
    orig_shape = hidden_states.shape
    topk_idx, topk_weight, _ = mlp.gate(hidden_states)
    flat = hidden_states.reshape(-1, hidden_states.shape[-1])
    flat_topk_idx = topk_idx.reshape(flat.shape[0], -1)
    flat_topk_weight = topk_weight.reshape(flat.shape[0], -1).to(flat.dtype)

    y = torch.zeros_like(flat)
    for token_index in range(flat.shape[0]):
        token = flat[token_index : token_index + 1]
        token_out = torch.zeros_like(token)
        for route_index in range(flat_topk_idx.shape[1]):
            expert_id = int(flat_topk_idx[token_index, route_index].item())
            expert = mlp.experts[expert_id]
            weight = flat_topk_weight[token_index, route_index].to(token.dtype)
            token_out = token_out + expert(token) * weight
        y[token_index : token_index + 1] = token_out

    y = y.reshape(*orig_shape)
    if getattr(mlp.config, "n_shared_experts", None) is not None:
        y = y + mlp.shared_experts(identity)
    return y


class MoEGateOnly(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.gate = mlp.gate

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        topk_idx, topk_weight, _ = self.gate(hidden_states)
        return topk_idx, topk_weight


class ExpertOnly(nn.Module):
    def __init__(self, expert: nn.Module):
        super().__init__()
        self.expert = expert

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.expert(hidden_states)


class SharedExpertsOnly(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.shared_experts = mlp.shared_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)
