"""Diagnose the step-1 training loss (~9.8) vs held-out loss (~1.3) seen in the quick run.

Rebuilds the exact first training batch (same split, same seeded sampler) and reports:
  - batch loss for the base model in eval/train mode, fp32/fp16-autocast, with/without grad
  - three repeated train-mode fp16 forwards (does only the first call differ?)
  - the same batch through LoRA- and PiSSA-wrapped models (should equal the base)
  - per-example answer loss and target length (does one long answer dominate?)
Takes ~2 minutes on a T4.

    python -m part1_vlm_lora.src.diagnose
"""

import copy

import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

from common import get_out_dir, set_seed

from .adapters import apply_method
from .data import majority_answer, make_splits
from .model_utils import load_smolvlm
from .train import VQACollator, _to_device
from .vqa import set_resolution


def batch_loss(model, batch, train_mode: bool, amp: bool, grad: bool = False, keep_logits: bool = False):
    model.train(train_mode)
    torch.cuda.reset_peak_memory_stats()
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(**batch)
    loss = float(out.loss.detach())
    finite = bool(torch.isfinite(out.logits).all())
    logits = out.logits.detach().float() if keep_logits else None
    del out
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    torch.cuda.empty_cache()
    return loss, finite, peak, logits


def case(label, fn):
    """Run one check; print one line, never crash the whole diagnostic."""
    try:
        loss, finite, peak, _ = fn()
        print(f"  {label:42} loss={loss:.4f}  logits_finite={finite}  peak={peak:.1f} GiB")
        return loss
    except Exception as e:  # noqa: BLE001
        print(f"  {label:42} FAILED: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        torch.cuda.empty_cache()
        return None


def per_example(logits, labels):
    lg, lb = logits[:, :-1], labels[:, 1:]
    nll = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100, reduction="none")
    nll = nll.view(lb.shape)
    mask = lb != -100
    return (nll * mask).sum(1) / mask.sum(1).clamp(min=1), mask.sum(1)


def main():
    set_seed(42)
    train_ds, _ = make_splits("textvqa", 64, 32, seed=42)  # the --quick split (cached)
    model, processor = load_smolvlm()
    for p in model.parameters():          # as in training: base weights frozen
        p.requires_grad_(False)
    set_resolution(processor, 1024)
    coll = VQACollator(processor)
    idx = list(RandomSampler(train_ds, generator=torch.Generator().manual_seed(42)))[:8]
    exs = [train_ds[i] for i in idx]
    batch = _to_device(coll(exs), model.device)

    print("== 1. batch loss, base model (no grad) ==")
    for train_mode in (False, True):
        for amp in (False, True):
            case(f"base  mode={'train' if train_mode else 'eval'} fp16={amp}",
                 lambda tm=train_mode, a=amp: batch_loss(model, batch, tm, a))
    reps = [case(f"base  train fp16 repeat {i + 1}", lambda: batch_loss(model, batch, True, True)) for i in range(3)]

    print("== 2. same batch through wrapped models (train mode, fp16) ==")
    for m in ("lora", "pissa"):
        try:
            pm, _ = apply_method(copy.deepcopy(model), m, "lm", 16)
            pm.to(model.device)
            case(f"{m:6} no grad", lambda: batch_loss(pm, batch, True, True))
            case(f"{m:6} grad on (= training step 1)", lambda: batch_loss(pm, batch, True, True, grad=True))
            del pm
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: FAILED to build: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
        torch.cuda.empty_cache()

    print("== 3. per-example answer loss (base, eval mode, fp16) ==")
    try:
        _, _, _, logits = batch_loss(model, batch, False, True, keep_logits=True)
        pe, ntok = per_example(logits, batch["labels"])
        for ex, l, n in zip(exs, pe.tolist(), ntok.tolist()):
            print(f"  loss={l:7.3f}  tokens={n:3d}  target={majority_answer(ex['answers'])!r:40}  "
                  f"q={ex['question'][:50]!r}")
        print(f"  token-weighted mean = {float((pe * ntok).sum() / ntok.sum()):.4f}   "
              f"example mean = {float(pe.mean()):.4f}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED: {type(e).__name__}: {str(e).splitlines()[0][:160]}")

    print("== 4. quick run log (first 10 steps) ==")
    log = get_out_dir("part1", "runs", "textvqa_pissa_lm_r16_lr0.0002_s42_quick") / "train_log.jsonl"
    if log.exists():
        import pandas as pd

        df = pd.read_json(log, lines=True)
        cols = [c for c in ("step", "loss", "grad_norm", "skipped", "loss_scale", "n_label_tokens") if c in df]
        print(df[cols].head(10).to_string(index=False))
    else:
        print("  (quick run log not found)")


if __name__ == "__main__":
    main()
