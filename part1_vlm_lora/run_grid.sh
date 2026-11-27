#!/usr/bin/env bash
# Part 1 run grid. Finished runs (done.json present) are skipped, so this is safe to
# re-run after a crash or disconnect. Run from the repo root, e.g. inside tmux:
#   bash part1_vlm_lora/run_grid.sh reproduce   # main result only: zero-shot + LoRA/PiSSA lr 2e-5 s42 (~45 min)
#   bash part1_vlm_lora/run_grid.sh main        # LoRA vs PiSSA, LM adapters, 2 LRs x 2 seeds + zero-shot
#   bash part1_vlm_lora/run_grid.sh reference   # full fine-tuning of the same matrices + minor/medium ablation
#   bash part1_vlm_lora/run_grid.sh bonus       # vision-tower placement + minor-init ablation at lr 2e-5
#   bash part1_vlm_lora/run_grid.sh vision LR   # full placement grid (more GPU than the free tier allows)
set -euo pipefail
WORKERS="${WORKERS:-2}"            # set WORKERS=8 on the workstation
T="python -m part1_vlm_lora.src.train --num-workers ${WORKERS}"
group="${1:-main}"

case "$group" in
  reproduce)
    # Minimal reproduction of the main result (~45 min on a free T4):
    # zero-shot, then LoRA vs PiSSA at the paper's lr (2e-5), seed 42.
    $T --method none --max-steps 0
    $T --method lora  --placement lm --lr 2e-5 --seed 42
    $T --method pissa --placement lm --lr 2e-5 --seed 42 ;;
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
  bonus)
    # Day-2 runs at the paper's lr (2e-5), where PiSSA's effect is clearest; seed 42.
    #  1-2: vision-specific test, adapters in the SigLIP encoder only (backprop through the
    #       vision tower: micro-batch 2 x 4 accumulation = effective 8, same as the LM runs)
    #  3:   PiSSA's own ablation (App. A): adapters from the *minor* singular components
    $T --method pissa --placement vision --lr 2e-5 --seed 42 --micro-batch 2 --grad-accum 4
    $T --method lora  --placement vision --lr 2e-5 --seed 42 --micro-batch 2 --grad-accum 4
    $T --method minor --placement lm     --lr 2e-5 --seed 42 ;;
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
