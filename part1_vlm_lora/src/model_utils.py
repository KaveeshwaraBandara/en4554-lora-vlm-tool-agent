"""SmolVLM loading, per-component parameter accounting and PEFT target selection.

Module names in transformers 5.x SmolVLMForConditionalGeneration:
  vision encoder (SigLIP):  model.vision_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj
                            model.vision_model.encoder.layers.{i}.mlp.{fc1,fc2}
  connector:                model.connector.modality_projection.proj
  language model (SmolLM2): model.text_model.layers.{i}.self_attn.{q,k,v,o}_proj
                            model.text_model.layers.{i}.mlp.{gate,up,down}_proj
  lm_head                   (tied to model.text_model.embed_tokens)

"Placement" says which of these get adapters. Adapter placement is the knob for the
vision-specific part of the experiment (does PiSSA behave differently in the vision
tower than in the language model?).
"""

from __future__ import annotations

import re
from collections import OrderedDict

import torch
from torch import nn

DEFAULT_MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"
MAX_PARAMS = 500_000_000  # size rule from the brief

# PEFT treats a string target_modules as a regex that must match the full module name.
_PLACEMENT_PATTERNS = OrderedDict(
    lm=r".*text_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)",
    vision=r".*vision_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc(1|2))",
    connector=r".*connector\.modality_projection\.proj",
)
# Attention-only variants (q, v), matching the setting used in many LoRA papers.
_PLACEMENT_PATTERNS["lm_qv"] = r".*text_model\.layers\.\d+\.self_attn\.(q|v)_proj"
_PLACEMENT_PATTERNS["vision_qv"] = r".*vision_model\.encoder\.layers\.\d+\.self_attn\.(q|v)_proj"

PLACEMENTS = ("lm", "vision", "connector", "lm_qv", "vision_qv", "lm+vision", "all")


def target_regex(placement: str) -> str:
    """Regex for LoraConfig(target_modules=...). Combine parts with '+', e.g. 'lm+connector'.

    'all' = lm + vision + connector.
    """
    if placement == "all":
        parts = ["lm", "vision", "connector"]
    else:
        parts = placement.split("+")
    unknown = [p for p in parts if p not in _PLACEMENT_PATTERNS]
    if unknown:
        raise ValueError(f"Unknown placement part(s) {unknown}; choose from {list(_PLACEMENT_PATTERNS)}")
    return "|".join(f"(?:{_PLACEMENT_PATTERNS[p]})" for p in parts)


def component_of(name: str) -> str:
    if "vision_model" in name:
        return "vision"
    if "connector" in name:
        return "connector"
    if name.startswith("lm_head") or ".lm_head" in name:
        return "lm_head"
    if "text_model" in name:
        return "lm"
    return "other"


def params_by_component(model: nn.Module) -> dict[str, int]:
    """Unique parameter counts per component (tied weights counted once)."""
    counts: dict[str, int] = {}
    seen: set[int] = set()
    for name, p in model.named_parameters(remove_duplicate=False):
        if id(p) in seen:
            continue
        seen.add(id(p))
        c = component_of(name)
        counts[c] = counts.get(c, 0) + p.numel()
    counts["total"] = sum(v for k, v in counts.items())
    return counts


def matched_linear_modules(model: nn.Module, placement: str) -> list[tuple[str, tuple[int, int]]]:
    """(name, (out_features, in_features)) of every nn.Linear that the placement targets."""
    pattern = re.compile(target_regex(placement))
    out = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and pattern.fullmatch(name):
            out.append((name, (mod.out_features, mod.in_features)))
    return out


def load_smolvlm(model_id: str = DEFAULT_MODEL_ID, dtype: torch.dtype = torch.float32, device: str | None = None):
    """Load model + processor.

    Keep the weights in fp32 and train with fp16 autocast (Trainer fp16=True): AMP needs
    fp32 master weights, and the PiSSA SVD is more accurate in fp32. 256M params in fp32
    is ~1 GB, which fits easily on a T4.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return model, processor


def check_size_rule(model: nn.Module) -> int:
    total = params_by_component(model)["total"]
    if total > MAX_PARAMS:
        raise ValueError(f"Model has {total/1e6:.1f}M params, above the 500M limit in the brief.")
    return total
