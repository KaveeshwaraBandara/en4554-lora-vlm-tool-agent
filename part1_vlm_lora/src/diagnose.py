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


def batch_loss(model, batch, train_mode: bool, amp: bool, grad: bool = False):
    model.train(train_mode)
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(**batch)
    finite = bool(torch.isfinite(out.logits).all())
    return float(out.loss.detach()), finite, out.logits.detach().float()


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
    set_resolution(processor, 1024)
    coll = VQACollator(processor)
    idx = list(RandomSampler(train_ds, generator=torch.Generator().manual_seed(42)))[:8]
    exs = [train_ds[i] for i in idx]
    batch = _to_device(coll(exs), model.device)

    print("== batch loss, base model ==")
    for train_mode in (False, True):
        for amp in (False, True):
            l, fin, _ = batch_loss(model, batch, train_mode, amp)
            print(f"  mode={'train' if train_mode else 'eval ':5} amp_fp16={amp!s:5}  loss={l:.4f}  logits_finite={fin}")
    l, fin, _ = batch_loss(model, batch, True, True, grad=True)
    print(f"  mode=train amp_fp16=True  grad=on  loss={l:.4f}  logits_finite={fin}")
    reps = [round(batch_loss(model, batch, True, True)[0], 4) for _ in range(3)]
    print(f"  3 repeated train/fp16 forwards: {reps}")

    print("== same batch through wrapped models (train mode, fp16) ==")
    for m in ("lora", "pissa"):
        pm, _ = apply_method(copy.deepcopy(model), m, "lm", 16)
        pm.to(model.device)
        l, fin, _ = batch_loss(pm, batch, True, True)
        print(f"  {m:6} loss={l:.4f}  logits_finite={fin}")
        del pm
        torch.cuda.empty_cache()

    print("== per-example answer loss (base, eval, fp32) ==")
    _, _, logits = batch_loss(model, batch, False, False)
    pe, ntok = per_example(logits, batch["labels"])
    for ex, l, n in zip(exs, pe.tolist(), ntok.tolist()):
        print(f"  loss={l:7.3f}  tokens={n:3d}  target={majority_answer(ex['answers'])!r:40}  q={ex['question'][:50]!r}")
    print(f"  token-weighted mean = {float((pe * ntok).sum() / ntok.sum()):.4f}   example mean = {float(pe.mean()):.4f}")

    print("== quick run log (first 10 steps) ==")
    log = get_out_dir("part1", "runs", "textvqa_pissa_lm_r16_lr0.0002_s42_quick") / "train_log.jsonl"
    if log.exists():
        import pandas as pd

        df = pd.read_json(log, lines=True)
        cols = [c for c in ("step", "loss", "grad_norm", "skipped", "loss_scale", "n_label_tokens") if c in df]
        print(df[cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
