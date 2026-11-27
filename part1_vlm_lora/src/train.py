"""Fine-tune SmolVLM with LoRA / PiSSA / minor / medium / full and log what the PiSSA
claim is about: training loss, gradient norm, held-out loss and accuracy over steps.

Precision: fp32 master weights, fp16 autocast, dynamic loss scaling (T4 has no bf16).
Data order depends only on --seed, so LoRA and PiSSA runs with the same seed see the
same batches in the same order (paired comparison).

Examples (repo root):
  python -m part1_vlm_lora.src.train --method pissa --placement lm --lr 2e-4 --seed 42
  python -m part1_vlm_lora.src.train --method none --max-steps 0          # zero-shot on the eval split
  python -m part1_vlm_lora.src.train --method lora --quick                # 2-minute pipeline check
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import math
import time

import torch
from torch.utils.data import DataLoader, RandomSampler

from common import JsonlLogger, get_out_dir, gpu_summary, load_json, save_json, set_seed

from .adapters import METHODS, apply_method, grad_norms_by_group
from .data import DATASETS, majority_answer, make_splits
from .metrics import score
from .model_utils import DEFAULT_MODEL_ID, load_smolvlm
from .vqa import build_messages, generate_answers, set_resolution


# ----------------------------------------------------------------------------- data

class VQACollator:
    """Chat-formatted (image, question) -> answer with loss only on the answer tokens.

    The answer span is found by tokenising the text that the assistant turn adds after the
    generation prompt and matching it at the end of each sequence. If the match fails
    (tokeniser merges across the boundary), the prompt is processed separately to get the
    exact prefix length. `batch["n_fallback"]` counts how often that happened.
    """

    def __init__(self, processor):
        self.p = processor

    def __call__(self, examples):
        tok = self.p.tokenizer
        tok.padding_side = "right"
        fulls, prompts, suffix_ids, images = [], [], [], []
        for ex in examples:
            target = majority_answer(ex["answers"])
            prompt = self.p.apply_chat_template(build_messages(ex["question"]), add_generation_prompt=True)
            full = self.p.apply_chat_template(build_messages(ex["question"], target))
            if not full.startswith(prompt):
                raise RuntimeError("chat template: full conversation does not start with the generation prompt")
            prompts.append(prompt)
            fulls.append(full)
            suffix_ids.append(tok(full[len(prompt):], add_special_tokens=False)["input_ids"])
            images.append([ex["image"]])

        enc = self.p(text=fulls, images=images, return_tensors="pt", padding=True)
        ids = enc["input_ids"]
        labels = torch.full_like(ids, -100)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        n_fallback = 0
        for i, sids in enumerate(suffix_ids):
            L, k = lengths[i], len(sids)
            if ids[i, L - k:L].tolist() == sids:
                start = L - k
            else:  # exact but slower: length of the processed prompt (same image expansion)
                n_fallback += 1
                start = self.p(text=[prompts[i]], images=[images[i]], return_tensors="pt")["input_ids"].shape[1]
            labels[i, start:L] = ids[i, start:L]
        enc["labels"] = labels
        enc["n_fallback"] = torch.tensor(n_fallback)
        return enc


def _to_device(batch, device):
    batch = dict(batch)
    batch.pop("n_fallback", None)
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


@torch.no_grad()
def eval_loss(model, collator, dataset, batch_size: int, device) -> float:
    """Token-weighted mean NLL on the answer tokens (teacher forcing)."""
    model.eval()
    total, n_tok = 0.0, 0
    for s in range(0, len(dataset), batch_size):
        batch = collator([dataset[i] for i in range(s, min(s + batch_size, len(dataset)))])
        batch = _to_device(batch, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(**batch)
        k = int((batch["labels"][:, 1:] != -100).sum())
        total += float(out.loss) * k
        n_tok += k
    model.train()
    return total / max(n_tok, 1)


def gen_eval(model, processor, dataset, metric, batch_size, out_path=None, step=None):
    model.eval()
    res = generate_answers(model, processor, dataset, batch_size=batch_size)
    scores = [score(metric, r["pred"], ex["answers"]) for r, ex in zip(res, dataset)]
    if out_path is not None:
        with JsonlLogger(out_path, append=False) as log:
            for ex, r, s in zip(dataset, res, scores):
                log.log({"step": step, "id": ex["id"], "question": ex["question"],
                         "answers": ex["answers"], "pred": r["pred"], "score": s})
    model.train()
    return 100.0 * sum(scores) / len(scores)


# ----------------------------------------------------------------------------- main

def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=list(METHODS) + ["none"], required=True)
    ap.add_argument("--placement", default="lm")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="textvqa", choices=list(DATASETS))
    ap.add_argument("--longest-edge", type=int, default=1024)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--data-seed", type=int, default=42, help="fixed across runs: same train/eval split")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-loss-every", type=int, default=50)
    ap.add_argument("--eval-loss-n", type=int, default=256)
    ap.add_argument("--eval-loss-extra", type=int, nargs="*", default=[5, 10, 25],
                    help="extra held-out loss points to resolve the early (answer-format) phase")
    ap.add_argument("--init-loss-scale", type=float, default=2.0 ** 12,
                    help="fp16 loss scale at step 1; 2^12 avoids the overflow-skipped first steps seen at 2^14")
    ap.add_argument("--gen-eval-steps", type=int, nargs="*", default=[50, 150, 300])
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--save-adapter", action="store_true")
    ap.add_argument("--force", action="store_true", help="re-run even if done.json exists")
    ap.add_argument("--quick", action="store_true", help="tiny run to check the pipeline")
    a = ap.parse_args(argv)
    if a.quick:
        a.n_train, a.n_eval, a.max_steps = 64, 32, 10
        a.eval_loss_every, a.eval_loss_n, a.gen_eval_steps, a.eval_loss_extra = 5, 32, [10], []
    return a


def run_name_of(a) -> str:
    if a.run_name:
        return a.run_name
    if a.method == "none":
        return f"{a.dataset}_zeroshot_e{a.longest_edge}"
    q = "_quick" if a.quick else ""
    return f"{a.dataset}_{a.method}_{a.placement}_r{a.rank}_lr{a.lr:g}_s{a.seed}{q}"


def main(argv=None):
    a = parse_args(argv)
    name = run_name_of(a)
    out = get_out_dir("part1", "runs", name)
    if (out / "done.json").exists() and not a.force:
        print(f"[skip] {name} already done: {load_json(out / 'done.json')}")
        return

    set_seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(gpu_summary())
    metric = DATASETS[a.dataset].metric

    train_ds, eval_ds = make_splits(a.dataset, a.n_train, a.n_eval, seed=a.data_seed)
    eval_loss_ds = eval_ds.select(range(min(a.eval_loss_n, len(eval_ds))))

    model, processor = load_smolvlm(a.model_id, device=str(device))
    set_resolution(processor, a.longest_edge)
    info = {}
    if a.method != "none":
        t0 = time.time()
        model, info = apply_method(model, a.method, a.placement, a.rank)
        info["init_seconds"] = round(time.time() - t0, 2)
        model.to(device)
    collator = VQACollator(processor)

    config = {**vars(a), "run_name": name, "metric": metric, "gpu": gpu_summary(), **info,
              "versions": {p: md.version(p) for p in ("torch", "transformers", "peft", "datasets")},
              "n_train_images": len(set(train_ds["group"])), "n_eval_images": len(set(eval_ds["group"]))}
    save_json(config, out / "config.json")
    print({k: config[k] for k in ("run_name", "trainable_params", "n_matrices") if k in config})

    eval_log = JsonlLogger(out / "eval_log.jsonl", append=False)

    if a.method == "none" or a.max_steps == 0:
        acc = gen_eval(model, processor, eval_ds, metric, a.gen_batch, out / "preds_step0.jsonl", 0)
        el = eval_loss(model, collator, eval_loss_ds, a.micro_batch, device)
        eval_log.log({"step": 0, "accuracy": acc, "eval_loss": el})
        save_json({"step": 0, "accuracy": acc, "eval_loss": el}, out / "done.json")
        print(f"zero-shot: {metric}={acc:.2f}  eval_loss={el:.4f}")
        return

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    warmup = max(1, math.ceil(a.warmup_ratio * a.max_steps))

    def lr_lambda(step):  # linear warmup, cosine decay to 0 (paper: cosine, warmup 0.03)
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, a.max_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", init_scale=a.init_loss_scale, enabled=device.type == "cuda")

    g = torch.Generator().manual_seed(a.seed)
    loader = DataLoader(train_ds, batch_size=a.micro_batch, sampler=RandomSampler(train_ds, generator=g),
                        collate_fn=collator, num_workers=a.num_workers, drop_last=True,
                        persistent_workers=a.num_workers > 0)
    it, epoch = iter(loader), 0

    def next_batch():
        nonlocal it, epoch
        try:
            return next(it)
        except StopIteration:
            epoch += 1
            it = iter(loader)
            return next(it)

    gen_steps = set(a.gen_eval_steps)
    train_log = JsonlLogger(out / "train_log.jsonl", append=False)
    # step-0 held-out loss: identical for every method (adapter init leaves outputs unchanged)
    el0 = eval_loss(model, collator, eval_loss_ds, a.micro_batch, device)
    eval_log.log({"step": 0, "eval_loss": el0})
    print(f"  eval @ 0: eval_loss={el0:.4f}")
    model.train()
    t_start = time.time()
    first_batch_checked = False
    for step in range(1, a.max_steps + 1):
        t0 = time.time()
        loss_sum, n_fallback, n_label = 0.0, 0, 0
        for _ in range(a.grad_accum):
            batch = next_batch()
            n_fallback += int(batch["n_fallback"])
            n_label += int((batch["labels"][:, 1:] != -100).sum())
            if not first_batch_checked:
                lab = batch["labels"][0]
                tgt = processor.tokenizer.decode(lab[lab != -100])
                print(f"[check] supervised tokens of example 0: {tgt!r}")
                first_batch_checked = True
            batch = _to_device(batch, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                loss = model(**batch).loss / a.grad_accum
            scaler.scale(loss).backward()
            loss_sum += float(loss.detach())

        scaler.unscale_(opt)
        comp = grad_norms_by_group(model)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, a.max_grad_norm))
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        lr_now = sched.get_last_lr()[0]
        sched.step()
        opt.zero_grad(set_to_none=True)

        rec = {"step": step, "epoch": epoch, "loss": loss_sum, "grad_norm": gnorm, **comp, "lr": lr_now,
               "loss_scale": scale_before, "skipped": bool(skipped), "n_fallback": n_fallback, "n_label_tokens": n_label,
               "step_time": round(time.time() - t0, 3)}
        train_log.log(rec)
        if step % 10 == 0 or step <= 10:
            print(f"step {step:4d} loss {loss_sum:.4f} gnorm {gnorm:.3f} lr {lr_now:.2e} tok {n_label:3d} "
                  f"{rec['step_time']:.2f}s/step{' SKIPPED' if skipped else ''}")

        ev = {}
        if step % a.eval_loss_every == 0 or step == a.max_steps or step in a.eval_loss_extra:
            ev["eval_loss"] = eval_loss(model, collator, eval_loss_ds, a.micro_batch, device)
        if step in gen_steps:
            ev["accuracy"] = gen_eval(model, processor, eval_ds, metric, a.gen_batch,
                                      out / f"preds_step{step}.jsonl", step)
        if ev:
            eval_log.log({"step": step, **ev})
            print(f"  eval @ {step}: {ev}")

    train_log.close()
    eval_log.close()
    if a.save_adapter and a.method != "full":
        model.save_pretrained(out / "adapter")
    done = {"steps": a.max_steps, "minutes": round((time.time() - t_start) / 60, 2),
            "peak_mem_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if device.type == "cuda" else None}
    save_json(done, out / "done.json")
    print("done:", done)


if __name__ == "__main__":
    main()
