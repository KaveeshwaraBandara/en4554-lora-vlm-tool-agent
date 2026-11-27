"""Report pipeline on synthetic logs with the same layout as real runs (no GPU)."""

import json

import numpy as np
import pandas as pd

from common import JsonlLogger, save_json
from part1_vlm_lora.src import report_figures as R


def _fake_run(root, method, lr, seed, placement="lm", speed=1.0, final=0.55):
    name = f"textvqa_{method}_{placement}_r16_lr{lr:g}_s{seed}"
    d = root / name
    d.mkdir(parents=True)
    save_json({"method": method, "placement": placement, "lr": lr, "seed": seed, "trainable_params": 4884480}, d / "config.json")
    rng = np.random.default_rng(seed)
    with JsonlLogger(d / "train_log.jsonl", append=False) as L:
        for st in range(1, 301):
            L.log({"step": st, "loss": 0.6 + 8 * np.exp(-st * speed / 20) + rng.random() * 0.1,
                   "grad_norm": (30 if method == "pissa" else 5) * (1 + rng.random()), "skipped": False, "step_time": 1.3})
    with JsonlLogger(d / "eval_log.jsonl", append=False) as L:
        for st in (0, 5, 10, 25, 50, 100, 150, 200, 250, 300):
            rec = {"step": st, "eval_loss": 0.7 + 7.4 * np.exp(-st * speed / 15)}
            if st in (50, 150, 300):
                rec["accuracy"] = 100 * (final - 0.05 * np.exp(-st / 100))
            L.log(rec)
    for st in (50, 150, 300):
        with JsonlLogger(d / f"preds_step{st}.jsonl", append=False) as L:
            p = final - 0.05 * np.exp(-st / 100)
            for i in range(500):
                L.log({"step": st, "id": str(i), "score": float(rng.random() < p)})
    save_json({"steps": 300}, d / "done.json")


def test_report_pipeline(tmp_path):
    runs = tmp_path / "runs"
    for seed in (42, 43):
        _fake_run(runs, "lora", 2e-5, seed, speed=0.3, final=0.51)
        _fake_run(runs, "pissa", 2e-5, seed, speed=1.5, final=0.55)
        _fake_run(runs, "lora", 2e-4, seed, speed=1.5, final=0.56)
        _fake_run(runs, "pissa", 2e-4, seed, speed=2.0, final=0.56)
    _fake_run(runs, "lora", 2e-5, 42, placement="vision", speed=0.4)
    _fake_run(runs, "pissa", 2e-5, 42, placement="vision", speed=0.8)
    zs = runs / "textvqa_zeroshot_e1024"
    zs.mkdir()
    save_json({"method": "none", "placement": "lm", "lr": 2e-4, "seed": 42}, zs / "config.json")
    with JsonlLogger(zs / "eval_log.jsonl", append=False) as L:
        L.log({"step": 0, "accuracy": 49.62, "eval_loss": 8.06})
    sp = pd.DataFrame([{"component": c, "layer": l, "type": "q_proj", "top_r_energy": e, "minor_r_energy": 0.001,
                        "eff_rank_frac": 0.8, "update_scale": u}
                       for c, n, e, u in (("lm", 30, 0.3, 12.0), ("vision", 12, 0.15, 6.0)) for l in range(n)]
                      + [{"component": "connector", "layer": -1, "type": "proj", "top_r_energy": 0.2,
                          "minor_r_energy": 0.0, "eff_rank_frac": 0.7, "update_scale": 9.0}])
    sp.to_csv(tmp_path / "spectral.csv", index=False)

    out = tmp_path / "out"
    R.main(["--runs", str(runs), "--spectral", str(tmp_path / "spectral.csv"), "--out", str(out)])
    for f in ("figures/fig_convergence.pdf", "figures/fig_vision.pdf", "tables/main_table.tex",
              "tables/paired.tex", "tables/numbers.json"):
        assert (out / f).exists(), f
    tex = (out / "tables" / "main_table.tex").read_text()
    assert "Zero-shot" in tex and r"\textbf{PiSSA}" in tex
    paired = pd.read_csv(out / "tables" / "paired.csv")
    row = paired[(paired.comparison == "PiSSA - LoRA, lr 2e-5") & (paired.step == 300)].iloc[0]
    assert row.n_questions == 500 and row.seeds == 2 and row.ci_low < row["diff"] < row.ci_high
    nums = json.loads((out / "tables" / "numbers.json").read_text())
    assert nums["zeroshot"]["acc"] == 49.62 and "vision" in nums["spectral"]
