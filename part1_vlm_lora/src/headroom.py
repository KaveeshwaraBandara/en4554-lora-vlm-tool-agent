"""Zero-shot SmolVLM on candidate datasets, to choose the fine-tuning task.

Decision rule (fixed before looking at results):
  1. prefer the dataset with the most headroom (lowest zero-shot score),
  2. among close candidates, prefer one whose answers depend on reading fine visual
     detail (TextVQA / DocVQA): the vision-placement test in step 4 only means something
     if the vision encoder has something to learn,
  3. it must fit the T4 budget: mean sequence length at the chosen resolution has to
     allow ~2k training examples x a few hundred steps in well under an hour.

Usage (repo root):
  python -m part1_vlm_lora.src.headroom --n 200 --longest-edge 0 1024
"""

import argparse
import csv
import statistics as st
from pathlib import Path

from common import JsonlLogger, get_out_dir, gpu_summary, set_seed

from .data import DATASETS, load_subset
from .metrics import score
from .model_utils import DEFAULT_MODEL_ID, load_smolvlm
from .vqa import generate_answers, set_resolution


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--longest-edge", type=int, nargs="+", default=[0],
                    help="image resize targets; 0 = processor default")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    args = ap.parse_args()

    set_seed(args.seed)
    print(gpu_summary())
    out_dir = get_out_dir("part1", "headroom")
    model, processor = load_smolvlm(args.model_id)
    model.eval()
    default_size = dict(processor.image_processor.size)
    print("processor default size:", default_size)

    summary = []
    with JsonlLogger(out_dir / "predictions.jsonl", append=False) as log:
        for name in args.datasets:
            spec = DATASETS[name]
            print(f"\n== {name} ({spec.hf_id}, {spec.split}) : loading {args.n} examples ...")
            ds = load_subset(name, args.n, seed=args.seed)
            for le in args.longest_edge:
                processor.image_processor.size = dict(default_size)
                set_resolution(processor, le or None)
                res = generate_answers(model, processor, ds, batch_size=args.batch_size,
                                       max_new_tokens=args.max_new_tokens)
                scores = []
                for ex, r in zip(ds, res):
                    s = score(spec.metric, r["pred"], ex["answers"])
                    scores.append(s)
                    log.log({"dataset": name, "longest_edge": le or default_size.get("longest_edge"),
                             "id": ex["id"], "question": ex["question"], "answers": ex["answers"],
                             **r, "score": s})
                row = {
                    "dataset": name,
                    "metric": spec.metric,
                    "longest_edge": le or default_size.get("longest_edge"),
                    "n": len(scores),
                    "zero_shot": round(100 * st.mean(scores), 2),
                    "mean_input_tokens": round(st.mean(r["n_input_tokens"] for r in res), 1),
                    "mean_image_tokens": round(st.mean(r["n_image_tokens"] for r in res), 1),
                    "ms_per_example": round(1000 * st.mean(r["seconds"] for r in res), 1),
                    "empty_pred_rate": round(sum(not r["pred"] for r in res) / len(res), 3),
                }
                summary.append(row)
                print(row)

    csv_path = out_dir / "headroom_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    # small copy inside the repo so it can be committed
    repo_copy = Path(__file__).resolve().parents[1] / "results" / "headroom_summary.csv"
    repo_copy.write_text(csv_path.read_text())

    print("\n| dataset | metric | longest_edge | zero-shot | input tok | image tok | ms/ex |")
    print("|---|---|---|---|---|---|---|")
    for r in summary:
        print(f"| {r['dataset']} | {r['metric']} | {r['longest_edge']} | {r['zero_shot']} | "
              f"{r['mean_input_tokens']} | {r['mean_image_tokens']} | {r['ms_per_example']} |")
    print(f"\nSaved: {csv_path}\n       {out_dir / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
