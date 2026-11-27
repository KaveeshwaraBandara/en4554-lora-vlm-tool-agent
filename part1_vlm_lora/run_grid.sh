#!/usr/bin/env bash
# Part 1 run grid. Finished runs (done.json present) are skipped, so this is safe to
# re-run after a crash or disconnect. Run from the repo root, e.g. inside tmux:
#   bash part1_vlm_lora/run_grid.sh main        # LoRA vs PiSSA, LM adapters, 2 LRs x 2 seeds + zero-shot
#   bash part1_vlm_lora/run_grid.sh reference   # full fine-tuning of the same matrices + minor/medium ablation
#   bash part1_vlm_lora/run_grid.sh vision LR   # placement test at the chosen LR (e.g. 2e-4)
set -euo pipefail
WORKERS="${WORKERS:-2}"            # set WORKERS=8 on the workstation
T="python -m part1_vlm_lora.src.train --num-workers ${WORKERS}"
group="${1:-main}"

case "$group" in
  main)
    $T --method none --max-steps 0
    for seed in 42 43; do
      for lr in 2e-4 2e-5; do
        for m in lora pissa; do
          $T --method "$m" --placement lm --lr "$lr" --seed "$seed"
        done
      done
    done ;;
  reference)
    LR="${2:-2e-4}"
    $T --method full --placement lm --lr 2e-5 --seed 42
    for m in minor medium; do
      $T --method "$m" --placement lm --lr "$LR" --seed 42
    done ;;
  vision)
    LR="${2:?give the LR chosen from the main grid, e.g. 2e-4}"
    for m in lora pissa; do
      # backprop through SigLIP for 5 tiles/image is memory-heavy: micro-batch 2 x 4 accumulation = 8
      $T --method "$m" --placement vision    --lr "$LR" --seed 42 --micro-batch 2 --grad-accum 4
      $T --method "$m" --placement connector --lr "$LR" --seed 42
      $T --method "$m" --placement all       --lr "$LR" --seed 42 --micro-batch 2 --grad-accum 4
    done ;;
  *) echo "unknown group: $group"; exit 1 ;;
esac
