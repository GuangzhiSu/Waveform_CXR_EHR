"""
LoRA adapters for 1D CNN / Linear (ECG xresnet encoder).
Only LoRA parameters + classification head are trained; base weights stay frozen.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """LoRA around a frozen nn.Linear: out = linear(x) + scale * x @ A.T @ B.T"""

    def __init__(self, linear: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = linear
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0
        out_f, in_f = linear.out_features, linear.in_features
        self.lora_A = nn.Parameter(torch.randn(r, in_f) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r <= 0:
            return out
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return out + lora


class LoRAConv1d(nn.Module):
    """Bottleneck LoRA for Conv1d (groups must be 1)."""

    def __init__(self, conv: nn.Conv1d, r: int, alpha: float):
        super().__init__()
        if conv.groups != 1:
            raise ValueError("LoRAConv1d only supports groups==1")
        self.base = conv
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0
        in_c, out_c = conv.in_channels, conv.out_channels
        k = conv.kernel_size[0]
        stride, pad, dil = conv.stride[0], conv.padding[0], conv.dilation[0]
        self.lora_A = nn.Conv1d(
            in_c, r, kernel_size=k, stride=stride, padding=pad, dilation=dil, bias=False
        )
        self.lora_B = nn.Conv1d(r, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r <= 0:
            return out
        return out + self.lora_B(self.lora_A(x)) * self.scaling


def _inject_recursive(module: nn.Module, r: int, alpha: float) -> int:
    """In-place replace Conv1d (groups==1) and Linear with LoRA wrappers. Returns count replaced."""
    n_replaced = 0
    for name, child in list(module.named_children()):
        # Critical: do not recurse into LoRA wrappers — their children are Conv1d used as frozen base / LoRA branches.
        # Recursing would wrap base and lora_A/lora_B again (nested LoRA, broken graph, ~hundreds of bogus layers).
        if isinstance(child, (LoRAConv1d, LoRALinear)):
            continue
        if isinstance(child, nn.Conv1d) and child.groups == 1:
            setattr(module, name, LoRAConv1d(child, r, alpha))
            n_replaced += 1
        elif isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r, alpha))
            n_replaced += 1
        else:
            n_replaced += _inject_recursive(child, r, alpha)
    return n_replaced


def inject_lora_into_xresnet(encoder: nn.Module, r: int = 8, alpha: float = 16.0) -> int:
    """Wrap Conv1d / Linear inside xresnet with LoRA. Returns number of layers wrapped."""
    return _inject_recursive(encoder, r, alpha)


def inject_lora_signal_encoder_proj(proj: nn.Linear, r: int, alpha: float) -> nn.Module:
    """LoRA on SignalEncoder's final proj (768 -> hidden_dim)."""
    return LoRALinear(proj, r, alpha)


def set_trainable_lora_and_head(model: nn.Module) -> None:
    """After LoRA inject: train LoRA params + classification head; freeze rest."""
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
        elif "head" in name:
            p.requires_grad = True
        else:
            p.requires_grad = False
