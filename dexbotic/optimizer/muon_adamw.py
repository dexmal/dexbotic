"""Reference Muon + AdamW updates for ordinary tensors and FSDP2 DTensors."""

from __future__ import annotations

import inspect
import math
from typing import Iterable

import torch
from torch.distributed.tensor import DTensor, Replicate
from torch.optim.adamw import adamw as _adamw_functional

MUON_ENABLED_ATTR = "_dexbotic_muon_adamw_enabled"
_ADAMW_FUNCTIONAL_PARAMS = inspect.signature(_adamw_functional).parameters


def zero_power_via_newton_schulz(
    matrix: torch.Tensor, *, steps: int, eps: float = 1e-7
) -> torch.Tensor:
    """Apply the reference Newton-Schulz polynomial to one complete matrix."""
    if matrix.ndim != 2:
        raise ValueError("Muon orthogonalization expects a 2D tensor.")
    update = matrix.float()
    if update.numel() == 0:
        return update
    transposed = update.shape[0] > update.shape[1]
    if transposed:
        update = update.t()
    norm = update.norm()
    if not torch.isfinite(norm) or norm <= 0:
        return torch.zeros_like(matrix, dtype=torch.float32)
    update = update / norm.clamp_min(eps)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(max(int(steps), 0)):
        gram = update @ update.t()
        update = a * update + (b * gram + c * (gram @ gram)) @ update
    return update.t() if transposed else update


def muon_moonlight_scale_for_shape(shape: tuple[int, int], coefficient: float) -> float:
    rows, cols = shape
    return (
        float(coefficient) * math.sqrt(max(rows, cols)) if min(rows, cols) > 0 else 1.0
    )


def _update_like_parameter(update: torch.Tensor, param: torch.Tensor) -> torch.Tensor:
    update = update.to(dtype=param.dtype)
    if not isinstance(param, DTensor):
        return update
    # full_tensor() gave every rank the same matrix. Mark it replicated without
    # another broadcast, then use DTensor's placement-aware slicing. In particular,
    # uneven/empty shards and nonzero sharding dimensions must not be hand-sliced.
    replicated = DTensor.from_local(
        update,
        device_mesh=param.device_mesh,
        placements=[Replicate()] * param.device_mesh.ndim,
        run_check=False,
    )
    return replicated.redistribute(placements=param.placements)


class MuonAdamW(torch.optim.Optimizer):
    """Muon on selected 2-D matrices, AdamW on the remaining trainable params.

    FSDP2 keeps global parameter shapes and placement metadata in DTensors.
    Momentum stays sharded in FP32. Each Muon direction is gathered for the
    nonlinear matrix update and then redistributed to the parameter's placement.
    FSDP1 flat-parameter shard reconstruction is deliberately not supported.
    """

    def __init__(
        self,
        params: Iterable[dict],
        *,
        lr: float,
        betas: tuple[float, float],
        eps: float,
        weight_decay: float,
        muon_momentum: float = 0.95,
        muon_nesterov: bool = True,
        muon_ns_steps: int = 5,
        muon_lr_scale: float = 1.0,
        muon_moonlight_scale: bool = True,
        muon_moonlight_coefficient: float = 0.2,
    ):
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                use_muon=False,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_ns_steps=muon_ns_steps,
                muon_lr_scale=muon_lr_scale,
                muon_moonlight_scale=muon_moonlight_scale,
                muon_moonlight_coefficient=muon_moonlight_coefficient,
            ),
        )
        for group in self.param_groups:
            if group["use_muon"] and any(p.ndim != 2 for p in group["params"]):
                raise ValueError("Muon requires 2-D parameters, not FSDP1 flat shards")

    @classmethod
    def from_named_parameters(cls, named_parameters, **kwargs) -> "MuonAdamW":
        adamw_params, muon_params = [], []
        seen = set()
        for _, param in named_parameters:
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            target = (
                muon_params
                if getattr(param, MUON_ENABLED_ATTR, False)
                else adamw_params
            )
            target.append(param)
        groups = []
        if adamw_params:
            groups.append(dict(params=adamw_params, use_muon=False))
        if muon_params:
            groups.append(dict(params=muon_params, use_muon=True))
        return cls(groups, **kwargs)

    def _state_for_parameter(self, param):
        state = self.state[param]
        if not state:
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
            for key in ("exp_avg", "exp_avg_sq"):
                state[key] = torch.zeros_like(param, dtype=torch.float32)
        # Optimizer.load_state_dict may cast buffers to the parameter dtype.
        # Keep reference FP32 momentum/variance, including after native DCP load.
        for key in ("exp_avg", "exp_avg_sq"):
            value = state.get(key)
            if torch.is_tensor(value) and value.dtype != torch.float32:
                state[key] = value.float()
        return state

    def load_state_dict(self, state_dict):
        # PyTorch's default loader casts floating state to the parameter dtype.
        # Restore our FP32 buffers from the uncast input, not from already-rounded
        # BF16 copies. This applies equally to plain tensors and native DCP loads.
        super().load_state_dict(state_dict)
        for saved_group, group in zip(
            state_dict["param_groups"], self.param_groups, strict=True
        ):
            for saved_id, param in zip(
                saved_group["params"], group["params"], strict=True
            ):
                saved = state_dict["state"].get(saved_id, {})
                for key in ("exp_avg", "exp_avg_sq"):
                    if key in saved:
                        self.state[param][key] = saved[key].to(
                            device=param.device, dtype=torch.float32
                        )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group["use_muon"]:
                self._muon_step_group(group)
            else:
                self._adamw_step_group(group)
        return loss

    def _adamw_step_group(self, group):
        beta1, beta2 = group["betas"]
        options = dict(
            amsgrad=False,
            beta1=float(beta1),
            beta2=float(beta2),
            lr=float(group["lr"]),
            weight_decay=float(group["weight_decay"] or 0.0),
            eps=float(group["eps"]),
            maximize=False,
            foreach=False,
            fused=False,
            capturable=False,
            differentiable=False,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
        )
        options = {k: v for k, v in options.items() if k in _ADAMW_FUNCTIONAL_PARAMS}
        for param in group["params"]:
            if param.grad is None:
                continue
            if param.grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients")
            state = self._state_for_parameter(param)
            _adamw_functional(
                [param],
                [param.grad.detach().float()],
                [state["exp_avg"]],
                [state["exp_avg_sq"]],
                [],
                [state["step"]],
                **options,
            )

    def _muon_step_group(self, group):
        lr = float(group["lr"])
        momentum = float(group["muon_momentum"])
        for param in group["params"]:
            grad = param.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("MuonAdamW does not support sparse gradients")
            if param.ndim != 2 or grad.shape != param.shape:
                raise ValueError("Muon requires matching 2-D parameters and gradients")
            state = self._state_for_parameter(param)
            state["step"].add_(1.0)
            if group["weight_decay"]:
                param.mul_(1.0 - lr * group["weight_decay"])
            grad = grad.detach().float()
            buffer = state["exp_avg"]
            buffer.mul_(momentum).add_(grad)
            direction = (
                grad.clone().add_(buffer, alpha=momentum)
                if group["muon_nesterov"]
                else buffer.detach().clone()
            )
            # Newton-Schulz must see the full matrix, never each shard separately.
            if isinstance(direction, DTensor):
                direction = direction.full_tensor()
            update = zero_power_via_newton_schulz(
                direction, steps=int(group["muon_ns_steps"])
            )
            effective_lr = lr * float(group["muon_lr_scale"])
            if group["muon_moonlight_scale"]:
                effective_lr *= muon_moonlight_scale_for_shape(
                    tuple(param.shape), float(group["muon_moonlight_coefficient"])
                )
            param.add_(_update_like_parameter(update, param), alpha=-effective_lr)
