"""Load run logs and make quick summaries/plots (report figures come from here later)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import get_out_dir, load_json


def load_runs(root: str | Path | None = None, pattern: str = "*") -> dict[str, dict]:
    root = Path(root) if root else get_out_dir("part1", "runs")
    runs = {}
    for d in sorted(root.glob(pattern)):
        if not (d / "config.json").exists():
            continue
        r = {"config": load_json(d / "config.json"), "dir": d,
             "done": load_json(d / "done.json") if (d / "done.json").exists() else None}
        for key, fname in (("train", "train_log.jsonl"), ("eval", "eval_log.jsonl")):
            f = d / fname
            r[key] = pd.read_json(f, lines=True) if f.exists() and f.stat().st_size else pd.DataFrame()
        runs[d.name] = r
    return runs


def summary_table(runs: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, r in runs.items():
        c, ev, tr = r["config"], r["eval"], r["train"]
        row = {"run": name, "method": c["method"], "placement": c.get("placement"), "lr": c.get("lr"),
               "seed": c.get("seed"), "trainable_M": round(c.get("trainable_params", 0) / 1e6, 2),
               "done": r["done"] is not None}
        if not ev.empty:
            for _, e in ev.iterrows():
                if pd.notna(e.get("accuracy")):
                    row[f"acc@{int(e['step'])}"] = round(e["accuracy"], 2)
            el = ev.dropna(subset=["eval_loss"]) if "eval_loss" in ev else ev.iloc[0:0]
            if len(el):
                row["final_eval_loss"] = round(el.iloc[-1]["eval_loss"], 4)
        if not tr.empty:
            row["loss@first50"] = round(tr[tr.step <= 50]["loss"].mean(), 4)
            row["loss@last50"] = round(tr.tail(50)["loss"].mean(), 4)
            row["skipped_steps"] = int(tr["skipped"].sum())
            row["sec/step"] = round(tr["step_time"].median(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(runs: dict[str, dict], placement: str = "lm", ema: float = 0.9, max_gn_step: int = 150):
    import matplotlib.pyplot as plt

    sel = {n: r for n, r in runs.items() if r["config"]["method"] != "none"
           and r["config"].get("placement") == placement and not r["train"].empty}
    lrs = sorted({r["config"]["lr"] for r in sel.values()}, reverse=True)
    if not lrs:
        print("no runs to plot")
        return None
    colors = {"lora": "tab:blue", "pissa": "tab:orange", "full": "tab:gray", "minor": "tab:green", "medium": "tab:purple"}
    fig, axes = plt.subplots(len(lrs), 3, figsize=(15, 3.6 * len(lrs)), squeeze=False)
    for i, lr in enumerate(lrs):
        for name, r in sel.items():
            c = r["config"]
            if c["lr"] != lr:
                continue
            tr, ev = r["train"], r["eval"]
            ls = "-" if c["seed"] == 42 else "--"
            kw = dict(color=colors.get(c["method"], "k"), ls=ls, lw=1.3, label=f"{c['method']} s{c['seed']}")
            axes[i, 0].plot(tr.step, tr.loss.ewm(alpha=1 - ema).mean(), **kw)
            early = tr[tr.step <= max_gn_step]
            axes[i, 1].plot(early.step, early.grad_norm, **kw)
            if "eval_loss" in ev:
                e = ev.dropna(subset=["eval_loss"])
                axes[i, 2].plot(e.step, e.eval_loss, marker="o", ms=3, **kw)
        axes[i, 0].set_title(f"train loss (EMA {ema}) · lr={lr:g}")
        axes[i, 1].set_title(f"grad norm, first {max_gn_step} steps · lr={lr:g}")
        axes[i, 2].set_title(f"held-out answer loss · lr={lr:g}")
        for ax in axes[i]:
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
        axes[i, 0].legend(fontsize=8)
    fig.tight_layout()
    return fig
