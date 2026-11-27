# en4554-lora-vlm-tool-agent

EN4554 Deep Learning for Vision – Assignment 1.

- **Part 1 (`part1_vlm_lora/`):** does PiSSA's claim, faster convergence than LoRA (Meng et al., NeurIPS 2024), transfer to a ≤500M vision-language model? Tested on **SmolVLM-256M-Instruct** with **TextVQA**. Report: [`part1_vlm_lora/report/report.pdf`](part1_vlm_lora/report/report.pdf).
- **Part 2 (`part2_tool_agent/`):** LoRA-finetuning a small LLM for tool calling (added by a teammate).

Everything runs on a free Colab T4 (fp16 autocast, fixed seeds).

## Part 1 at a glance

2,000 train / 500 eval questions from the TextVQA **validation** split (disjoint by image), images at 1024 px. LM adapters (all 7 projections × 30 layers), r = 16, α = r, 300 steps, batch 8. Mean of seeds 42 and 43. Accuracy is VQA accuracy on the 500 eval questions.

| | acc @50 | acc @150 | acc @300 | held-out answer loss @300 |
|---|---|---|---|---|
| zero-shot | 49.6 | | | 8.06 |
| LoRA, lr 2e-5 (paper lr) | 48.2 | 51.1 | 51.3 | 0.948 |
| **PiSSA, lr 2e-5 (paper lr)** | **51.9** | **53.2** | **54.6** | **0.717** |
| LoRA, lr 2e-4 | 51.8 | 54.1 | 55.7 | 0.687 |
| PiSSA, lr 2e-4 | 52.2 | 51.9 | 56.1 | 0.688 |

At the paper's learning rate PiSSA converges much faster and ends 3.3 points higher. With a 10× larger learning rate, LoRA reaches the same held-out loss and accuracy, so in this setting the advantage behaves like a larger effective step size. See the report for figures, paired statistics and the vision-specific analysis.

## Repository layout

```
common/                      seeding, output dirs (Google Drive on Colab), JSONL logging
part1_vlm_lora/
  src/                       data, metrics, adapters (LoRA / PiSSA / minor / medium / full),
                             training, spectral analysis, report figures
  tests/                     offline CPU tests on a tiny random SmolVLM (no downloads)
  notebooks/                 00 setup · 01 dataset headroom · 02 training · 03 analysis (CPU)
  run_grid.sh                run groups: reproduce · main · bonus · reference
  results/                   logs, summary tables and figures behind the report
  report/                    LaTeX source + report.pdf (≤ 3 pages)
part2_tool_agent/            Part 2 (teammate)
```

## Reproduce the main result (Colab, ~45 min on a T4)

1. Open `part1_vlm_lora/notebooks/02_train.ipynb` from GitHub in Colab and select **Runtime → T4 GPU**.
2. Run the setup cell. It clones the repo, installs `requirements.txt` and mounts Google Drive for logs.
3. Replace the grid cell's command with `!bash part1_vlm_lora/run_grid.sh reproduce`. This runs zero-shot, then LoRA vs PiSSA at lr 2e-5, seed 42.
4. Run the "Quick look" cells for the table and curves.

The full grid (`main`: 2 learning rates × 2 seeds) takes ~2.7 h, more than one free-tier daily allowance, and restarts cleanly: finished runs are skipped. `python -m part1_vlm_lora.src.report_figures` rebuilds the report figures and statistics from the logs in `part1_vlm_lora/results/runs/`.

## Running locally

```bash
# install torch >= 2.6 for your CUDA version first
pip install -r requirements.txt
python -m pytest -q part1_vlm_lora/tests                  # offline checks (CPU)
bash part1_vlm_lora/run_grid.sh reproduce                 # outputs -> ./outputs (or $EN4554_OUT)
```

Runs use **fp16 even on GPUs that support bf16**, so results match the T4.

## Reproducibility

- Seeds 42 and 43. The train/eval split is fixed by `--data-seed 42`, and LoRA and PiSSA with the same seed see identical batches.
- fp32 master weights with fp16 autocast and dynamic loss scaling (initial scale 2^12, no skipped steps).
- Package versions are pinned in `requirements.txt`. Every run logs its config, seed and versions.
