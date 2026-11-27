"""Offline checks for the adapter methods on a tiny random SmolVLM (CPU)."""

import copy

import pytest
import torch
from peft import LoraConfig, get_peft_model

from part1_vlm_lora.src.adapters import apply_method, grad_norms_by_group, svd_init_
from part1_vlm_lora.src.model_utils import target_regex
from part1_vlm_lora.src.tiny_model import TINY_IMAGE_TOKEN_ID, tiny_batch, tiny_smolvlm


@pytest.fixture(scope="module")
def base():
    return tiny_smolvlm()


@pytest.fixture(scope="module")
def batch(base):
    b = tiny_batch(base)
    labels = b["input_ids"].clone()
    labels[labels == TINY_IMAGE_TOKEN_ID] = -100
    b["labels"] = labels
    return b


def _lora_products(pm):
    return {n: (m.lora_B["default"].weight @ m.lora_A["default"].weight * m.scaling["default"]).detach()
            for n, m in pm.named_modules() if hasattr(m, "lora_A")}


def test_custom_principal_matches_peft_pissa(base):
    ref = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=4, target_modules=target_regex("all"),
                                                         init_lora_weights="pissa"))
    mine = get_peft_model(copy.deepcopy(base), LoraConfig(r=4, lora_alpha=4, target_modules=target_regex("all"),
                                                          init_lora_weights=False))
    assert svd_init_(mine, "principal") == len(_lora_products(ref))
    pr, pm_ = _lora_products(ref), _lora_products(mine)
    for n in pr:
        assert torch.allclose(pr[n], pm_[n], atol=1e-5), n
    rw = {n: p for n, p in ref.named_parameters() if "base_layer.weight" in n}
    mw = {n: p for n, p in mine.named_parameters() if "base_layer.weight" in n}
    for n in rw:
        assert torch.allclose(rw[n], mw[n], atol=1e-5), n


def test_minor_uses_smallest_singular_triplets(base):
    name = "model.text_model.layers.1.mlp.down_proj"
    W = dict(base.named_modules())[name].weight.detach().clone()
    pm, _ = apply_method(copy.deepcopy(base), "minor", "lm", rank=4)
    mod = dict(pm.named_modules())["base_model.model." + name]
    BA = mod.lora_B["default"].weight @ mod.lora_A["default"].weight
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    assert torch.allclose(BA, U[:, -4:] @ torch.diag(S[-4:]) @ Vh[-4:], atol=1e-5)
    # minor components carry far less energy than principal ones
    assert BA.norm() < 0.5 * (U[:, :4] @ torch.diag(S[:4]) @ Vh[:4]).norm()


@pytest.mark.parametrize("method", ["lora", "pissa", "medium", "minor", "full"])
def test_methods_preserve_output_and_train(base, batch, method):
    model = copy.deepcopy(base)
    with torch.no_grad():
        ref = model(**batch).logits
    model, info = apply_method(model, method, "all", rank=4)
    assert info["trainable_params"] > 0
    assert set(info["trainable_by_component"]) == {"lm", "vision", "connector"}
    with torch.no_grad():
        assert torch.allclose(ref, model(**batch).logits, atol=1e-4)

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    losses = []
    for _ in range(5):
        loss = model(**batch).loss
        loss.backward()
        if len(losses) == 0:
            g = grad_norms_by_group(model)
            assert "grad_norm_lm" in g and "grad_norm_vision" in g
        opt.step()
        opt.zero_grad()
        losses.append(float(loss))
    assert losses[-1] < losses[0]


def test_generate_through_peft_wrapper(base, batch):
    model, _ = apply_method(copy.deepcopy(base), "pissa", "lm", rank=4)
    assert model.config.image_token_id == TINY_IMAGE_TOKEN_ID
    inp = {k: v for k, v in batch.items() if k != "labels"}
    out = model.generate(**inp, max_new_tokens=3, do_sample=False)
    assert out.shape[1] == inp["input_ids"].shape[1] + 3
