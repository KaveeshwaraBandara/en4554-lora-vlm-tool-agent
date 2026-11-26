# Part 1 – Does PiSSA's claim transfer to a vision-language model?

**Paper:** PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models (Meng, Wang & Zhang, NeurIPS 2024).
**Model:** `HuggingFaceTB/SmolVLM-256M-Instruct` (SigLIP-base/16-512 vision encoder + pixel-shuffle connector + SmolLM2-135M).
**PEFT:** `LoraConfig(init_lora_weights="pissa")` vs the default LoRA init, at equal rank and equal trainable parameters.

## Plan

1. **Claim:** from the paper, write down the central claim and the figure/table behind the convergence claim (report items 1–2).
2. **Dataset choice:** zero-shot SmolVLM on ~200 examples each of the candidate datasets, and pick the one with the most headroom. SmolVLM's training mix (The Cauldron, Docmatix) overlaps several candidates.
3. **Main experiment:** LoRA vs PiSSA, adapters on the language model, 2 seeds. Log loss and gradient norm every step, and evaluate at fixed step budgets.
4. **Vision-specific test:** spectral concentration (top-r energy) per module for vision / connector / LM, plus the PiSSA-minus-LoRA gain for each adapter placement.
5. **Report** (≤ 3 pages).

## Metrics tied to the claim

- Training loss vs step, and steps needed to reach a fixed loss threshold.
- Gradient norm over the first steps (PiSSA's explanation for faster convergence).
- Task metric at fixed step budgets (not only at the end).

## Adapter placements

Defined in `src/model_utils.py`; checked by `tests/test_offline.py`.

| placement | targets |
|---|---|
| `lm` | text_model q/k/v/o + gate/up/down |
| `vision` | vision_model q/k/v/out + fc1/fc2 |
| `connector` | modality_projection.proj |
| `all` | all of the above |
