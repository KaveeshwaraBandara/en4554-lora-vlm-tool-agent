"""Adapter methods compared in Part 1, all on the same target matrices.

  lora    : PEFT default init (A Kaiming-uniform, B = 0)
  pissa   : PEFT init_lora_weights="pissa": A, B from the top-r singular triplets,
            residual W - BA frozen (Meng et al., 2024)
  medium  : same construction with the middle-r singular triplets   (PiSSA App. A ablation)
  minor   : same construction with the smallest-r singular triplets (PiSSA App. A ablation)
  full    : full-rank update of exactly the targeted matrices (reference for the
            "PiSSA behaves like full fine-tuning" part of the claim)

Following the paper: lora_alpha = r (scaling 1) and lora_dropout = 0.
"""

from __future__ import annotations

import re

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from .model_utils import component_of, target_regex

METHODS = ("lora", "pissa", "medium", "minor", "full")


def _svd_slice(n_sv: int, r: int, which: str) -> slice:
    if which == "principal":
        return slice(0, r)
    if which == "minor":
        return slice(n_sv - r, n_sv)
    if which == "medium":
        start = (n_sv - r) // 2
        return slice(start, start + r)
    raise ValueError(which)


@torch.no_grad()
def svd_init_(peft_model: nn.Module, which: str, adapter: str = "default") -> int:
    """Initialise every LoRA layer from a slice of the base weight's SVD (PEFT's PiSSA recipe).

    A = sqrt(S_sel) Vh_sel, B = U_sel sqrt(S_sel) (with S divided by the LoRA scaling), and the
    base weight becomes W - scaling * B @ A, so the model output is unchanged at step 0.
    """
    n = 0
    for _, mod in peft_model.named_modules():
        if not (hasattr(mod, "lora_A") and adapter in getattr(mod, "lora_A", {})):
            continue
        base = mod.get_base_layer()
        W = base.weight.data
        dtype = W.dtype
        Wf = W.to(torch.float32)
        U, S, Vh = torch.linalg.svd(Wf, full_matrices=False)
        r = mod.r[adapter]
        sl = _svd_slice(S.numel(), r, which)
        scaling = mod.scaling[adapter]
        Sr = S[sl] / scaling
        A = torch.diag(torch.sqrt(Sr)) @ Vh[sl]
        B = U[:, sl] @ torch.diag(torch.sqrt(Sr))
        mod.lora_A[adapter].weight.data = A.contiguous()
        mod.lora_B[adapter].weight.data = B.contiguous()
        base.weight.data = (Wf - scaling * B @ A).to(dtype)
        n += 1
    return n


def apply_method(model: nn.Module, method: str, placement: str, rank: int = 16):
    """Returns (model, info). The returned model may be a PeftModel."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    regex = target_regex(placement)

    if method == "full":
        pattern = re.compile(regex)
        for p in model.parameters():
            p.requires_grad_(False)
        n_mats = 0
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and pattern.fullmatch(name):
                mod.weight.requires_grad_(True)
                n_mats += 1
        info = {"n_matrices": n_mats}
    else:
        init = {"lora": True, "pissa": "pissa"}.get(method, False)
        cfg = LoraConfig(r=rank, lora_alpha=rank, lora_dropout=0.0, target_modules=regex,
                         init_lora_weights=init)
        model = get_peft_model(model, cfg)
        info = {"n_matrices": sum(1 for _, m in model.named_modules() if hasattr(m, "lora_A"))}
        if method in ("medium", "minor"):
            svd_init_(model, method)

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    info["trainable_params"] = sum(p.numel() for _, p in trainable)
    info["trainable_by_component"] = {}
    for n, p in trainable:
        c = component_of(n)
        info["trainable_by_component"][c] = info["trainable_by_component"].get(c, 0) + p.numel()
    return model, info


def grad_norms_by_group(model: nn.Module) -> dict[str, float]:
    """L2 grad norm per component (lm / vision / connector) and per adapter factor (A / B)."""
    sq: dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        g = float(p.grad.detach().float().pow(2).sum())
        keys = [component_of(n)]
        if "lora_A" in n:
            keys.append("A")
        elif "lora_B" in n:
            keys.append("B")
        for k in keys:
            sq[k] = sq.get(k, 0.0) + g
    return {f"grad_norm_{k}": v ** 0.5 for k, v in sq.items()}
