"""Figures, tables and paired statistics for the Part 1 report, built from the run logs.

    python -m part1_vlm_lora.src.report_figures [--runs part1_vlm_lora/results/runs]

Writes to part1_vlm_lora/results/{figures,tables}/:
  fig_convergence.pdf   held-out answer loss, training loss, gradient norm (LoRA vs PiSSA, 2 lrs)
  fig_vision.pdf        spectral statistics (LM vs vision tower) [+ vision-placement runs if present]
  main_table.tex/.csv   accuracy at fixed step budgets, held-out loss, steps to held-out loss <= 1
  paired.tex/.csv       paired bootstrap of accuracy differences (same 500 eval questions)
  numbers.json          every number quoted in the report text
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import load_runs

RESULTS = Path(__file__).resolve().parents[1] / "results"
COLORS = {"lora": "#1f77b4", "pissa": "#d62728", "minor": "#2ca02c", "full": "#7f7f7f"}
LR_STYLE = {2e-5: "-", 2e-4: "--"}
LABEL = {"lora": "LoRA", "pissa": "PiSSA", "minor": "minor-SVD init", "full": "full update"}
THRESH = 1.0
STYLE = {"font.size": 8, "axes.titlesize": 8.5, "legend.fontsize": 7, "font.family": "STIXGeneral",
         "mathtext.fontset": "stix", "pdf.fonttype": 42}


def _lr_key(lr: float) -> float:
    return float(f"{lr:.0e}")


def index_runs(runs: dict) -> pd.DataFrame:
    rows = []
    for name, r in runs.items():
        c = r["config"]
        if c["method"] == "none" or name.endswith("_quick"):
            continue
        rows.append({"name": name, "method": c["method"], "placement": c.get("placement"),
                     "lr": _lr_key(c["lr"]), "seed": c["seed"], "done": r["done"] is not None})
    return pd.DataFrame(rows)


def eval_series(r, col):
    ev = r["eval"]
    if ev.empty or col not in ev:
        return pd.Series(dtype=float)
    e = ev.dropna(subset=[col])
    return e.groupby("step")[col].last()


def preds(run_dir: Path, step: int) -> pd.Series:
    f = run_dir / f"preds_step{step}.jsonl"
    if not f.exists():
        return pd.Series(dtype=float)
    d = pd.read_json(f, lines=True)
    return d.set_index(d["id"].astype(str))["score"]


def paired_bootstrap(diff: np.ndarray, n_boot: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return float(diff.mean()), float(lo), float(hi), float(min(1.0, p))


def method_scores(runs, idx, method, lr, placement, step):
    """Per-question accuracy averaged over the available seeds."""
    sel = idx[(idx.method == method) & (idx.lr == lr) & (idx.placement == placement)]
    series = [preds(runs[n]["dir"], step) for n in sel.name]
    series = [s for s in series if len(s)]
    if not series:
        return None, 0
    return pd.concat(series, axis=1).mean(axis=1), len(series)


def steps_to(series: pd.Series, thr: float):
    hit = series[series <= thr]
    return int(hit.index.min()) if len(hit) else None


# --------------------------------------------------------------------------- figures

def fig_convergence(runs, idx, out: Path):
    import matplotlib.pyplot as plt

    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.15))
    lm = idx[(idx.placement == "lm") & idx.method.isin(["lora", "pissa"])]
    for (method, lr), g in lm.groupby(["method", "lr"]):
        kw = dict(color=COLORS[method], ls=LR_STYLE.get(lr, ":"), lw=1.2)
        lab = f"{LABEL[method]}, lr {lr:.0e}".replace("e-05", "e-5").replace("e-04", "e-4")
        # (a) held-out loss: mean over seeds, band = seed range
        S = pd.concat([eval_series(runs[n], "eval_loss") for n in g.name], axis=1)
        x = S.index.values
        axes[0].plot(np.maximum(x, 1), S.mean(axis=1), marker="o", ms=2.2, label=lab, **kw)
        axes[0].fill_between(np.maximum(x, 1), S.min(axis=1), S.max(axis=1), color=COLORS[method], alpha=0.12, lw=0)
        # (b) training loss (EMA), (c) grad norm (first 100 steps)
        tr = [runs[n]["train"].set_index("step") for n in g.name]
        ema = pd.concat([t["loss"].ewm(alpha=0.1).mean() for t in tr], axis=1).mean(axis=1)
        axes[1].plot(ema.index, ema.values, **kw)
        gn = pd.concat([t["grad_norm"].rolling(5, min_periods=1).median() for t in tr], axis=1).mean(axis=1)
        gn = gn[gn.index <= 100]
        axes[2].plot(gn.index, gn.values, **kw)
    axes[0].set_xscale("log")
    axes[0].axhline(THRESH, color="k", lw=0.6, ls=":")
    axes[0].set(title="(a) held-out answer loss", xlabel="step (log)", ylabel="NLL / token")
    axes[1].set(title="(b) training loss (EMA)", xlabel="step", ylim=(0, 4))
    axes[2].set(title="(c) gradient norm (pre-clip)", xlabel="step", yscale="log")
    axes[0].legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_convergence.pdf")
    fig.savefig(out / "fig_convergence.png", dpi=200)
    plt.close(fig)


def fig_vision(runs, idx, spectral_csv: Path | None, out: Path):
    import matplotlib.pyplot as plt

    vis = idx[(idx.placement == "vision") & idx.done]
    have_spec = spectral_csv is not None and spectral_csv.exists()
    n = int(have_spec) * 2 + int(len(vis) > 0)
    if n == 0:
        return None
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(1, n, figsize=(2.35 * n, 2.1), squeeze=False)
    axes = list(axes[0])
    k = 0
    if have_spec:
        sp = pd.read_csv(spectral_csv)
        for comp, c in (("lm", "#1f77b4"), ("vision", "#d62728")):
            d = sp[sp.component == comp]
            g = d.groupby("layer")
            depth = g.top_r_energy.mean().index / max(1, d.layer.max())
            axes[0].plot(depth, g.top_r_energy.mean().values, marker="o", ms=2, color=c,
                         label="SmolLM2 (LM)" if comp == "lm" else "SigLIP (vision)")
            axes[1].plot(depth, g.update_scale.mean().values, marker="o", ms=2, color=c)
        con = sp[sp.component == "connector"]
        if len(con):
            axes[0].scatter([1.0], con.top_r_energy, marker="*", s=40, color="k", label="connector", zorder=3)
            axes[1].scatter([1.0], con.update_scale, marker="*", s=40, color="k", zorder=3)
        axes[0].set(title="(a) energy in top-16 singular dirs", xlabel="relative depth")
        axes[1].set(title="(b) PiSSA / LoRA update-scale proxy", xlabel="relative depth")
        axes[0].legend(frameon=False)
        k = 2
    if len(vis):
        ax = axes[k]
        for method, g in vis.groupby("method"):
            for n_ in g.name:
                s = eval_series(runs[n_], "eval_loss")
                ax.plot(np.maximum(s.index.values, 1), s.values, marker="o", ms=2.2, color=COLORS[method],
                        label=f"{LABEL[method]} (vision tower)")
        for method in ("lora", "pissa"):
            g = idx[(idx.placement == "lm") & (idx.method == method) & (idx.lr == 2e-5) & (idx.seed == 42)]
            for n_ in g.name:
                s = eval_series(runs[n_], "eval_loss")
                ax.plot(np.maximum(s.index.values, 1), s.values, color=COLORS[method], ls=":", lw=1,
                        label=f"{LABEL[method]} (LM)")
        ax.set(xscale="log", title="(c) held-out loss, lr 2e-5", xlabel="step (log)")
        ax.legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout(pad=0.4)
    fig.savefig(out / "fig_vision.pdf")
    fig.savefig(out / "fig_vision.png", dpi=200)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- tables + numbers

def main_table(runs, idx, zeroshot):
    rows = []
    for (method, lr, placement), g in idx[idx.done].groupby(["method", "lr", "placement"]):
        accs = {s: [] for s in (50, 150, 300)}
        losses, s2t = [], []
        for n in g.name:
            acc = eval_series(runs[n], "accuracy")
            for s in accs:
                if s in acc.index:
                    accs[s].append(acc[s])
            el = eval_series(runs[n], "eval_loss")
            if len(el):
                losses.append(el.iloc[-1])
                s2t.append(steps_to(el, THRESH))
        row = {"method": method, "lr": lr, "placement": placement, "seeds": len(g)}
        for s, v in accs.items():
            row[f"acc{s}"] = float(np.mean(v)) if v else None
            row[f"acc{s}_half_range"] = float((max(v) - min(v)) / 2) if len(v) > 1 else None
        row["eval_loss"] = float(np.mean(losses)) if losses else None
        row["steps_to_loss_le_1"] = s2t
        tr = [runs[n]["train"] for n in g.name]
        row["train_loss_last50"] = float(np.mean([t.tail(50)["loss"].mean() for t in tr]))
        row["grad_norm_med_10_100"] = float(np.mean([t[(t.step >= 10) & (t.step <= 100)]["grad_norm"].median()
                                                     for t in tr]))
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["placement", "lr", "method"])
    if zeroshot is not None:
        df.attrs["zeroshot"] = zeroshot
    return df


def paired_table(runs, idx):
    comps = [  # (label, (method, lr, placement) A, B): diff = A - B
        ("PiSSA - LoRA, lr 2e-5", ("pissa", 2e-5, "lm"), ("lora", 2e-5, "lm")),
        ("PiSSA - LoRA, lr 2e-4", ("pissa", 2e-4, "lm"), ("lora", 2e-4, "lm")),
        ("PiSSA 2e-5 - LoRA 2e-4", ("pissa", 2e-5, "lm"), ("lora", 2e-4, "lm")),
        ("PiSSA - LoRA, vision tower, 2e-5", ("pissa", 2e-5, "vision"), ("lora", 2e-5, "vision")),
        ("minor - LoRA, lr 2e-5", ("minor", 2e-5, "lm"), ("lora", 2e-5, "lm")),
    ]
    rows = []
    for label, a, b in comps:
        for step in (50, 150, 300):
            sa, na = method_scores(runs, idx, *a, step)
            sb, nb = method_scores(runs, idx, *b, step)
            if sa is None or sb is None:
                continue
            common = sa.index.intersection(sb.index)
            d = (sa[common] - sb[common]).to_numpy() * 100
            m, lo, hi, p = paired_bootstrap(d)
            rows.append({"comparison": label, "step": step, "n_questions": len(common), "seeds": min(na, nb),
                         "diff": m, "ci_low": lo, "ci_high": hi, "p": p})
    return pd.DataFrame(rows)


def _fmt(x, nd=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def write_latex(df: pd.DataFrame, paired: pd.DataFrame, zeroshot, out: Path):
    lm = df[df.placement == "lm"]
    lines = [r"\begin{tabular}{llcccccc}", r"\toprule",
             r"Method & lr & acc@50 & acc@150 & acc@300 & held-out loss & steps to loss$\le$1 & train loss \\",
             r"\midrule"]
    if zeroshot is not None:
        lines.append(rf"Zero-shot & -- & \multicolumn{{3}}{{c}}{{{zeroshot['acc']:.1f}}} & {zeroshot['loss']:.2f} & -- & -- \\")
    for _, r in lm.iterrows():
        name = LABEL.get(r.method, r.method)
        lr = f"{r.lr:.0e}".replace("e-05", "e-5").replace("e-04", "e-4")
        acc = [(_fmt(r[f'acc{s}']) + (rf"\,$\pm${_fmt(r[f'acc{s}_half_range'])}" if r[f'acc{s}_half_range'] is not None else ""))
               for s in (50, 150, 300)]
        s2t = "/".join(str(x) if x is not None else "$>$300" for x in r.steps_to_loss_le_1)
        bold = r.method == "pissa" and abs(r.lr - 2e-5) < 1e-9
        cells = [name, lr, *acc, _fmt(r.eval_loss, 3), s2t, _fmt(r.train_loss_last50, 2)]
        if bold:
            cells = [rf"\textbf{{{c}}}" for c in cells]
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "main_table.tex").write_text("\n".join(lines) + "\n")

    pl = [r"\begin{tabular}{lccc}", r"\toprule",
          r"Accuracy difference (points) & step 50 & step 150 & step 300 \\", r"\midrule"]
    for label, g in paired.groupby("comparison", sort=False):
        cells = []
        for s in (50, 150, 300):
            r = g[g.step == s]
            if len(r):
                r = r.iloc[0]
                star = "$^{*}$" if r.ci_low > 0 or r.ci_high < 0 else ""
                cells.append(f"${r['diff']:+.1f}$ [${r.ci_low:+.1f}$, ${r.ci_high:+.1f}$]{star}")
            else:
                cells.append("--")
        pl.append(label.replace(" - ", " $-$ ") + " & " + " & ".join(cells) + r" \\")
    pl += [r"\bottomrule", r"\end{tabular}"]
    (out / "paired.tex").write_text("\n".join(pl) + "\n")


def write_macros(df: pd.DataFrame, paired: pd.DataFrame, zeroshot, spectral: dict | None, out: Path):
    """Numbers for the report text as \\R{key} (keys printed to macros.tex). Undefined -> bold ??."""
    m: dict[str, str] = {}
    if zeroshot:
        m["zs-acc"], m["zs-loss"] = f"{zeroshot['acc']:.1f}", f"{zeroshot['loss']:.2f}"
    for _, r in df.iterrows():
        k = f"{r.method}-{r.placement}-{r.lr:.0e}".replace("e-05", "e-5").replace("e-04", "e-4")
        for s_ in (50, 150, 300):
            if r[f"acc{s_}"] is not None and not pd.isna(r[f"acc{s_}"]):
                m[f"{k}-acc{s_}"] = f"{r[f'acc{s_}']:.1f}"
        if r.eval_loss is not None:
            m[f"{k}-loss"] = f"{r.eval_loss:.3f}"
        m[f"{k}-train"] = f"{r.train_loss_last50:.2f}"
        m[f"{k}-gnorm"] = f"{r.grad_norm_med_10_100:.1f}"
        m[f"{k}-s2t"] = "/".join(str(x) if x is not None else "$>$300" for x in r.steps_to_loss_le_1)
    short = {"PiSSA - LoRA, lr 2e-5": "pl-low", "PiSSA - LoRA, lr 2e-4": "pl-high",
             "PiSSA 2e-5 - LoRA 2e-4": "pl-cross", "PiSSA - LoRA, vision tower, 2e-5": "pl-vision",
             "minor - LoRA, lr 2e-5": "ml-low"}
    for _, r in paired.iterrows():
        k = f"{short.get(r.comparison, r.comparison)}-{int(r.step)}"
        m[f"{k}-diff"], m[f"{k}-lo"], m[f"{k}-hi"] = f"${r['diff']:+.1f}$", f"${r.ci_low:+.1f}$", f"${r.ci_high:+.1f}$"
        m[f"{k}-p"] = f"{r.p:.3f}" if r.p >= 0.001 else "$<$0.001"
    if spectral:
        for comp, d in spectral.items():
            m[f"spec-{comp}-energy"] = f"{100 * d['top_r_energy']:.1f}"
            m[f"spec-{comp}-minor"] = f"{100 * d['minor_r_energy']:.2f}"
            m[f"spec-{comp}-effrank"] = f"{100 * d['eff_rank_frac']:.0f}"
            m[f"spec-{comp}-scale"] = f"{d['update_scale']:.1f}"
    lines = [r"\makeatletter"] + [rf"\@namedef{{r@{k}}}{{{v}}}" for k, v in sorted(m.items())] + [r"\makeatother"]
    (out / "macros.tex").write_text("\n".join(lines) + "\n")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(RESULTS / "runs"))
    ap.add_argument("--spectral", default=str(RESULTS / "spectral" / "spectral_r16.csv"))
    ap.add_argument("--out", default=str(RESULTS))
    a = ap.parse_args(argv)
    figs, tabs = Path(a.out) / "figures", Path(a.out) / "tables"
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    runs = load_runs(a.runs)
    idx = index_runs(runs)
    zs = next((r for n, r in runs.items() if r["config"]["method"] == "none"), None)
    zeroshot = None
    if zs is not None and not zs["eval"].empty:
        e0 = zs["eval"].iloc[0]
        zeroshot = {"acc": float(e0["accuracy"]), "loss": float(e0["eval_loss"])}

    fig_convergence(runs, idx, figs)
    fig_vision(runs, idx, Path(a.spectral), figs)
    df = main_table(runs, idx, zeroshot)
    paired = paired_table(runs, idx)
    df.to_csv(tabs / "main_table.csv", index=False)
    paired.to_csv(tabs / "paired.csv", index=False)
    write_latex(df, paired, zeroshot, tabs)

    numbers = {"zeroshot": zeroshot, "main": df.to_dict(orient="records"), "paired": paired.to_dict(orient="records")}
    spec = Path(a.spectral)
    if spec.exists():
        sp = pd.read_csv(spec)
        numbers["spectral"] = sp.groupby("component")[["top_r_energy", "minor_r_energy", "eff_rank_frac",
                                                       "update_scale"]].mean().round(4).to_dict(orient="index")
    (tabs / "numbers.json").write_text(json.dumps(numbers, indent=2, default=str))
    m = write_macros(df, paired, zeroshot, numbers.get("spectral"), tabs)
    if spec.exists() and "sigma1" in pd.read_csv(spec, nrows=1).columns:  # per-type mean sigma_1 range
        t = pd.read_csv(spec).groupby(["component", "type"]).sigma1.mean()
        extra = {f"spec-{c}-s1min": f"{t[c].min():.1f}" for c in ("lm", "vision")}
        extra |= {f"spec-{c}-s1max": f"{t[c].max():.1f}" for c in ("lm", "vision")}
        with open(tabs / "macros.tex", "a") as f:
            f.write("\\makeatletter\n" + "".join(f"\\@namedef{{r@{k}}}{{{v}}}\n" for k, v in extra.items())
                    + "\\makeatother\n")
    pd.set_option("display.width", 200)
    print(df.drop(columns=[c for c in df.columns if c.endswith("half_range")]).to_string(index=False))
    print(paired.round(3).to_string(index=False))
    if "spectral" in numbers:
        print(json.dumps(numbers["spectral"], indent=1))


if __name__ == "__main__":
    main()
