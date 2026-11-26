# en4554-lora-vlm-tool-agent

EN4554 Deep Learning for Vision – Assignment 1.

- **Part 1:** does a LoRA-variant claim from NLP transfer to a ≤500M vision-language model? We test **PiSSA** (Meng et al., NeurIPS 2024) on **SmolVLM-256M-Instruct**.
- **Part 2:** LoRA-finetune **Qwen2.5-0.5B-Instruct** for tool calling inside a minimal agent loop, and measure what changes.

Everything runs on a free Colab T4 (fp16, fixed seeds).

## Repository layout

```
common/                   shared helpers: seeding, output dirs (Drive on Colab), JSONL logging
part1_vlm_lora/
  src/                    model utils, adapter placement, training/eval, spectral analysis
  tests/                  offline CPU tests on a tiny random SmolVLM (no downloads)
  configs/                run configs
  notebooks/              Colab entry points (00_setup reproduces the environment)
  results/                CSVs + figures used in the report
  report/                 LaTeX report (≤ 3 pages)
part2_tool_agent/
  data/                   generated train/test JSONL
  writeup/                ≤ 1 page write-up
```

## Quickstart on Colab

1. Open `part1_vlm_lora/notebooks/00_setup.ipynb` in Colab and choose **Runtime → Change runtime type → T4 GPU**.
2. Run all cells. The notebook clones this repo, installs `requirements.txt`, mounts Google Drive and runs the model inspection.

Checkpoints and logs go to `MyDrive/en4554-a1/` so a disconnect does not lose work.

## Running locally

```bash
# install torch >= 2.6 for your CUDA version first
pip install -r requirements.txt
python -m pytest -q part1_vlm_lora/tests          # offline sanity checks (CPU)
python -m part1_vlm_lora.src.inspect_model         # parameter breakdown + adapter targets
```

Outputs go to `./outputs/` (set `EN4554_OUT` to change this). Runs use **fp16 even on GPUs that support bf16**, so results match the T4 setup.

## Reproducibility

- Seeds: 42 (primary) and 43 (second seed). Every run logs its seed, config and package versions.
- Precision: fp32 master weights with fp16 autocast (`fp16=True`). bf16 is rejected by `common.assert_fp16_only`.
- Package versions are pinned in `requirements.txt`.

## Status

| Step | Status |
|---|---|
| Phase 0: scaffold, utils, adapter targeting tests | ✅ |
| Part 1: dataset headroom check | ⏳ |
| Part 1: LoRA vs PiSSA training + eval | ⏳ |
| Part 1: vision-specific analysis (placement × spectrum) | ⏳ |
| Part 1: report | ⏳ |
| Part 2: tools, agent loop, data | ⏳ |
| Part 2: baseline vs LoRA eval + failure analysis | ⏳ |
| Part 2: write-up | ⏳ |
