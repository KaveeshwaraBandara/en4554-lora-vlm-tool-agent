"""Spectral analysis of the weights PiSSA initialises from (CPU only, no training).

For every matrix targeted by the `lm`, `vision` and `connector` placements:
  top_r_energy    sum_{i<=r} s_i^2 / sum s_i^2   (share of the weight PiSSA's adapter holds at init)
  minor_r_energy  same for the r smallest singular values (what the `minor` ablation holds)
  eff_rank_frac   entropy effective rank (Roy & Vetterli, 2007) / min(m, n)
  sqrt_sigma_top  mean sqrt(s_i), i <= r: the row norm of PiSSA's A (and column norm of B)
  update_scale    (||A_pissa||_F + ||B_pissa||_F) / ||A_lora||_F, a first-order proxy for how much
                  larger a PiSSA update dW = dB A + B dA is than LoRA's (B = 0, so dW = dB A) for the
                  same Adam step. LoRA's A is PEFT's Kaiming-uniform(a=sqrt(5)) init, whose rows
                  have expected norm 1/sqrt(3), so ||A_lora||_F = sqrt(r/3).

Two questions for the report:
  1. Does the update-scale ratio explain why PiSSA at lr 2e-5 behaves like LoRA at ~10x lr?
  2. Vision-specific: are SigLIP's spectra flatter than SmolLM2's (lower top-r energy)? If so,
     PiSSA's init captures less of each vision matrix, and its edge should shrink when the
     adapters sit in the vision tower.

    python -m part1_vlm_lora.src.spectral --rank 16
"""

from __future__ import annotations

import argparse
import math
import re

import torch

from common import get_out_dir

from .model_utils import DEFAULT_MODEL_ID, matched_linear_modules

LORA_A_ROW_NORM = 1 / math.sqrt(3)


def spectral_stats(W: torch.Tensor, r: int) -> dict:
    s = torch.linalg.svdvals(W.detach().float())
    e = s ** 2
    tot = e.sum()
    p = e / tot
    eff_rank = float(torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum()))
    a_pissa = float(torch.sqrt(s[:r].sum()))  # ||A||_F = ||B||_F = sqrt(sum_{i<=r} s_i)
    return {
        "rows": W.shape[0], "cols": W.shape[1],
        "top_r_energy": float(e[:r].sum() / tot),
        "minor_r_energy": float(e[-r:].sum() / tot),
        "eff_rank_frac": eff_rank / min(W.shape),
        "sigma1": float(s[0]), "sigma_r": float(s[r - 1]),
        "sqrt_sigma_top": float(torch.sqrt(s[:r]).mean()),
        "update_scale": 2 * a_pissa / (LORA_A_ROW_NORM * math.sqrt(r)),
    }


def main():
    import pandas as pd

    from .model_utils import load_smolvlm

    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    a = ap.parse_args()

    model, _ = load_smolvlm(a.model_id, device="cpu")
    mods = dict(model.named_modules())
    rows = []
    for comp in ("lm", "vision", "connector"):
        for name, _ in matched_linear_modules(model, comp):
            m = re.search(r"layers\.(\d+)\.", name)
            rows.append({"component": comp, "name": name, "layer": int(m.group(1)) if m else -1,
                         "type": name.rsplit(".", 1)[-1], **spectral_stats(mods[name].weight, a.rank)})
    df = pd.DataFrame(rows)
    out = get_out_dir("part1", "spectral")
    df.to_csv(out / f"spectral_r{a.rank}.csv", index=False)

    cols = ["top_r_energy", "minor_r_energy", "eff_rank_frac", "sqrt_sigma_top", "update_scale"]
    summ = df.groupby("component")[cols].agg(["mean", "median"]).round(4)
    by_type = df.groupby(["component", "type"])[["top_r_energy", "update_scale"]].mean().round(4)
    print(f"\n== per component (r={a.rank}, {len(df)} matrices) ==\n{summ.to_string()}")
    print(f"\n== per component and matrix type ==\n{by_type.to_string()}")
    summ.to_csv(out / f"spectral_summary_r{a.rank}.csv")

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for comp, c in (("lm", "tab:blue"), ("vision", "tab:red")):
        d = df[df.component == comp]
        g = d.groupby("layer")
        axes[0].plot(g.top_r_energy.mean().index / max(1, d.layer.max()), g.top_r_energy.mean().values,
                     marker="o", ms=3, color=c, label=comp)
        axes[1].plot(g.update_scale.mean().index / max(1, d.layer.max()), g.update_scale.mean().values,
                     marker="o", ms=3, color=c, label=comp)
    axes[0].set_ylabel(f"top-{a.rank} energy share")
    axes[1].set_ylabel("PiSSA / LoRA update-scale proxy")
    for ax in axes:
        ax.set_xlabel("relative depth")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"spectral_r{a.rank}.png", dpi=150)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
