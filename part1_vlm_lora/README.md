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

## Dataset decision (step 2, done)

Zero-shot, 200 validation examples each, T4 fp16 (`results/headroom_summary.csv`):

| dataset | default (2048 px) | 1024 px | input tokens (2048 → 1024) |
|---|---|---|---|
| TextVQA (VQA acc) | 51.1 | **51.0** | 955 → 359 |
| DocVQA (ANLS) | 50.1 | 48.4 | 1092 → 361 |
| OK-VQA (VQA acc) | 34.5 | 35.1 | 930 → 362 |
| VQAv2 (VQA acc) | 71.7 | 70.9 | 913 → 360 |

**Chosen: TextVQA at 1024 px.** The pre-set rule ("most headroom first") would pick OK-VQA. We deviate because OK-VQA's gap is mostly missing world knowledge, which a 135M LM cannot acquire from ~2k examples, and the task barely depends on the vision encoder, which would make the placement test (step 4) uninformative. TextVQA needs fine visual reading, and 1024 px loses nothing (51.0 vs 51.1) at 2.7× shorter sequences.

## Paper settings we follow (PiSSA §5)

`lora_alpha = r`, dropout 0, AdamW without weight decay, cosine schedule, warmup ratio 0.03, adapters on all linear layers, lr 2e-5.
We also run lr 2e-4 (a typical LoRA learning rate for small models) to check whether PiSSA's advantage is only an effective-learning-rate effect.
Deviations: batch 8 instead of 128, and fp16 autocast instead of fp32 (T4).

## Run groups (`run_grid.sh`)

| group | runs |
|---|---|
| `main` | zero-shot; {LoRA, PiSSA} × lr {2e-4, 2e-5} × seed {42, 43}; LM adapters, r=16, 300 steps |
| `reference` | full-rank update of the same LM matrices (lr 2e-5); minor and medium SVD init (PiSSA App. A ablation) |
| `vision LR` | {LoRA, PiSSA} × placement {vision, connector, all} |

## Observation: answer-format phase (diagnostic, step 3)

At step 0 the answer loss is ~9.8 for every method and precision (fp32/fp16, eval/train, base/LoRA/PiSSA are identical).
SmolVLM's natural answers are capitalised with a full stop ("Chamberlain."), while TextVQA ground truth is lowercase without one. Under teacher forcing every example is expensive (5.9–14.8 nats/token), even though zero-shot accuracy, which normalises case and punctuation, is 51%.
The loss drops to ~1 within ~5 steps as the model adopts the target format. Convergence is therefore analysed in two phases: **format** (steps ≲ 10) and **content** (after).
The fp16 loss scale starts at 2^12, so no early steps are lost to overflow skips.

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
