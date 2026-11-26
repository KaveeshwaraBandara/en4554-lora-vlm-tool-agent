"""Offline checks on a tiny random SmolVLM (CPU, no downloads).

Run from the repo root:  python -m pytest -q part1_vlm_lora/tests
"""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from part1_vlm_lora.src.model_utils import matched_linear_modules, target_regex
from part1_vlm_lora.src.tiny_model import tiny_smolvlm, tiny_batch

PLACEMENTS = ["lm", "vision", "connector", "all"]


@pytest.fixture(scope="module")
def base():
    return tiny_smolvlm()


@pytest.fixture(scope="module")
def batch(base):
    return tiny_batch(base)


def test_placements_disjoint_and_complete(base):
    lm = {n for n, _ in matched_linear_modules(base, "lm")}
    vis = {n for n, _ in matched_linear_modules(base, "vision")}
    con = {n for n, _ in matched_linear_modules(base, "connector")}
    allm = {n for n, _ in matched_linear_modules(base, "all")}
    assert lm and vis and len(con) == 1
    assert not (lm & vis) and not (lm & con) and not (vis & con)
    assert allm == lm | vis | con
    # 2 layers x 7 matrices (LM), 2 layers x 6 matrices (vision)
    assert len(lm) == 14 and len(vis) == 12
    assert all("lm_head" not in n and "embed" not in n for n in allm)


@pytest.mark.parametrize("init", [True, "pissa"])
@pytest.mark.parametrize("placement", PLACEMENTS)
def test_peft_wraps_exactly_the_targets_and_preserves_output(base, batch, placement, init):
    model = copy.deepcopy(base)
    with torch.no_grad():
        ref = model(**batch).logits

    cfg = LoraConfig(r=4, lora_alpha=4, lora_dropout=0.0, target_modules=target_regex(placement),
                     init_lora_weights=init)
    pm = get_peft_model(model, cfg)

    wrapped = {n.removeprefix("base_model.model.") for n, m in pm.named_modules() if hasattr(m, "lora_A")}
    expected = {n for n, _ in matched_linear_modules(base, placement)}
    assert wrapped == expected

    # Both LoRA (B=0) and PiSSA (residual W - BA kept in the base) leave the output
    # unchanged at step 0, so any difference in training is due to the init subspace.
    with torch.no_grad():
        out = pm(**batch).logits
    assert torch.allclose(ref, out, atol=1e-4), (ref - out).abs().max()

    # Only adapter params train.
    trainable = [n for n, p in pm.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)


def test_pissa_adapter_spans_principal_subspace(base):
    """PiSSA's B@A should equal the rank-r truncated SVD of the original weight."""
    model = copy.deepcopy(base)
    name = "model.text_model.layers.0.self_attn.q_proj"
    W = dict(model.named_modules())[name].weight.detach().clone()
    r = 4
    pm = get_peft_model(model, LoraConfig(r=r, lora_alpha=r, target_modules=[name.rsplit(".", 1)[-1]],
                                          layers_to_transform=[0], layers_pattern="layers",
                                          init_lora_weights="pissa"))
    mod = dict(pm.named_modules())["base_model.model." + name]
    BA = (mod.lora_B["default"].weight @ mod.lora_A["default"].weight) * mod.scaling["default"]
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    W_r = U[:, :r] @ torch.diag(S[:r]) @ Vh[:r]
    assert torch.allclose(BA, W_r, atol=1e-4)
    assert torch.allclose(mod.base_layer.weight + BA, W, atol=1e-4)
