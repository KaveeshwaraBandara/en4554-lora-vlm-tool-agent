"""Print SmolVLM's parameter breakdown and what each adapter placement would target.

Usage (from the repo root):
    python -m part1_vlm_lora.src.inspect_model                 # real model (downloads ~1 GB)
    python -m part1_vlm_lora.src.inspect_model --tiny          # offline sanity check
    python -m part1_vlm_lora.src.inspect_model --rank 16 --save part1_vlm_lora/results/model_inventory.json
"""

import argparse
from collections import Counter

from common import save_json

from .model_utils import (
    DEFAULT_MODEL_ID,
    PLACEMENTS,
    check_size_rule,
    matched_linear_modules,
    params_by_component,
)


def lora_param_count(shapes: list[tuple[int, int]], r: int) -> int:
    # LoRA/PiSSA adds A (r x in) and B (out x r) per targeted matrix.
    return sum(r * (o + i) for o, i in shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--tiny", action="store_true", help="use a tiny random model (no download)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--save", default=None, help="optional path for a JSON inventory")
    args = ap.parse_args()

    if args.tiny:
        from .tiny_model import tiny_smolvlm

        model, name = tiny_smolvlm(), "tiny-random-smolvlm"
    else:
        from .model_utils import load_smolvlm

        model, _ = load_smolvlm(args.model_id, device="cpu")
        name = args.model_id

    counts = params_by_component(model)
    print(f"\n== {name} ==")
    for k in ("vision", "connector", "lm", "lm_head", "other", "total"):
        if k in counts:
            print(f"  {k:<10} {counts[k] / 1e6:8.2f} M")
    if not args.tiny:
        check_size_rule(model)
        print(f"  size rule (<= 500M): OK")

    inventory = {"model": name, "params": counts, "rank": args.rank, "placements": {}}
    print(f"\n== Adapter placements (r={args.rank}) ==")
    for pl in PLACEMENTS:
        mods = matched_linear_modules(model, pl)
        shapes = [s for _, s in mods]
        kinds = Counter(n.rsplit(".", 1)[-1] for n, _ in mods)
        n_train = lora_param_count(shapes, args.rank)
        print(f"  {pl:<10} {len(mods):4d} matrices | {n_train / 1e6:6.2f} M trainable | {dict(kinds)}")
        inventory["placements"][pl] = {
            "n_matrices": len(mods),
            "trainable_params": n_train,
            "by_type": dict(kinds),
            "shapes": sorted(set(shapes)),
        }

    if args.save:
        save_json(inventory, args.save)
        print(f"\nSaved inventory to {args.save}")


if __name__ == "__main__":
    main()
