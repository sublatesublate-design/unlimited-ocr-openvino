from __future__ import annotations

import torch
import torch.nn.functional as F
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


class HotExpertPack(nn.Module):
    def __init__(self, mlp: nn.Module, expert_ids: list[int]):
        super().__init__()
        if not expert_ids:
            raise ValueError("expert_ids must not be empty")
        self.expert_ids = [int(expert_id) for expert_id in expert_ids]
        self.expert_count = len(self.expert_ids)
        self.act_fn = mlp.experts[self.expert_ids[0]].act_fn
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.reshape(flat.shape[0], -1)
        flat_topk_weight = topk_weight.reshape(flat.shape[0], -1).to(flat.dtype)

        flat_by_expert = flat.unsqueeze(0).expand(self.expert_count, -1, -1).transpose(1, 2)
        gate_proj = torch.bmm(self.gate_weights, flat_by_expert).permute(2, 0, 1)
        up_proj = torch.bmm(self.up_weights, flat_by_expert).permute(2, 0, 1)
        expert_hidden = self.act_fn(gate_proj) * up_proj
        expert_hidden_by_expert = expert_hidden.permute(1, 2, 0)
        expert_out = torch.bmm(self.down_weights, expert_hidden_by_expert).permute(2, 0, 1)

        match = flat_topk_idx.unsqueeze(-1) == self.expert_ids_tensor.reshape(1, 1, self.expert_count)
        route_weight = match.to(flat.dtype).mul(flat_topk_weight.unsqueeze(-1)).sum(dim=1)
        routed = expert_out.mul(route_weight.unsqueeze(-1)).sum(dim=1)
        return routed.reshape(orig_shape)


class HotExpertGatherPack(nn.Module):
    def __init__(self, mlp: nn.Module, expert_ids: list[int]):
        super().__init__()
        if not expert_ids:
            raise ValueError("expert_ids must not be empty")
        self.expert_ids = [int(expert_id) for expert_id in expert_ids]
        self.expert_count = len(self.expert_ids)
        self.act_fn = mlp.experts[self.expert_ids[0]].act_fn
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weight: torch.Tensor,
    ) -> torch.Tensor:
        orig_shape = hidden_states.shape
        flat = hidden_states.reshape(-1, hidden_states.shape[-1])
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
        routed = expert_out.mul(weights.unsqueeze(-1)).sum(dim=1)
        return routed.reshape(orig_shape)


class SharedExpertsOnly(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.shared_experts = mlp.shared_experts

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.shared_experts(hidden_states)
