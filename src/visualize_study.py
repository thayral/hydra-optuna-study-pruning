#!/usr/bin/env python3
"""
visualize_study.py

Visualizes an existing Optuna study (loaded from storage) to highlight the effect of pruning.

Implements 3 ideas:
1) Optuna built-in plots (optimization history, intermediate values, param importances)
2) Quantitative summary of trial states + pruning rate
3) Custom plot: histogram of epochs/steps actually executed per trial

Usage:
  python visualize_study.py \
    --study-name mnist_pruning_demo \
    --storage sqlite:///optuna.db \
    --outdir reports

Notes:
- Optuna's interactive plots are saved as HTML (requires plotly, which optuna usually installs extras for).
- The histogram is saved as PNG (matplotlib).
- If your study has no intermediate_values, the pruning-related plots will be sparse.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import optuna


def safe_makedirs(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_study(study_name: str, storage: str) -> optuna.study.Study:
    return optuna.load_study(study_name=study_name, storage=storage)


def summarize_trials(study: optuna.study.Study) -> str:
    states = Counter(t.state for t in study.trials)

    total = len(study.trials)
    pruned = states.get(optuna.trial.TrialState.PRUNED, 0)
    complete = states.get(optuna.trial.TrialState.COMPLETE, 0)
    fail = states.get(optuna.trial.TrialState.FAIL, 0)
    running = states.get(optuna.trial.TrialState.RUNNING, 0)
    waiting = states.get(optuna.trial.TrialState.WAITING, 0)

    lines = []
    lines.append("=== Trial states ===")
    for k, v in sorted(states.items(), key=lambda kv: str(kv[0])):
        lines.append(f"  {k}: {v}")

    if total > 0:
        lines.append("")
        lines.append(f"Total trials: {total}")
        lines.append(f"Completed:   {complete} ({100*complete/total:.1f}%)")
        lines.append(f"Pruned:      {pruned} ({100*pruned/total:.1f}%)")
        if fail:
            lines.append(f"Failed:      {fail} ({100*fail/total:.1f}%)")
        if running:
            lines.append(f"Running:     {running} ({100*running/total:.1f}%)")
        if waiting:
            lines.append(f"Waiting:     {waiting} ({100*waiting/total:.1f}%)")

    # Best trial info (if available)
    try:
        bt = study.best_trial
        lines.append("")
        lines.append("=== Best trial ===")
        lines.append(f"  number: {bt.number}")
        lines.append(f"  value:  {bt.value}")
        lines.append(f"  params: {bt.params}")
    except Exception:
        # Multi-objective or no completed trials
        pass

    return "\n".join(lines)


def steps_executed_per_trial(study: optuna.study.Study) -> List[int]:
    """
    Uses intermediate_values keys to estimate how many steps/epochs were executed.
    Convention: step indices start at 0 in most loops, so max_step + 1 ≈ steps executed.
    """
    steps = []
    for t in study.trials:
        if t.intermediate_values:
            max_step = max(t.intermediate_values.keys())
            steps.append(int(max_step) +1)
        else:
            steps.append(0)
    return steps






def save_optuna_plots_html(study: optuna.study.Study, outdir: Path) -> List[Tuple[str, Path]]:
    """
    Saves the key Optuna pruning-relevant plots as HTML:
      - optimization history
      - intermediate values (the most relevant for pruning)
      - param importances (nice bonus)
    """
    # Import here so script still runs even if plotly isn't installed
    import optuna.visualization as vis

    outputs: List[Tuple[str, Path]] = []

    figs = [
        ("optimization_history", vis.plot_optimization_history(study)),
        ("intermediate_values", vis.plot_intermediate_values(study)),
    ]

    # Param importances can fail if there's not enough completed data.
    try:
        figs.append(("param_importances", vis.plot_param_importances(study)))
    except Exception:
        pass

    for name, fig in figs:
        path = outdir / f"{name}.html"
        fig.write_html(str(path))
        outputs.append((name, path))

    return outputs


def save_steps_histogram_png(steps: List[int], outdir: Path) -> Path:
    import matplotlib.pyplot as plt

    if not steps:
        steps = [0]

    max_step = max(steps)

    # Half-integer bin edges so integer values are centered on bars
    bins = [x - 0.5 for x in range(0, max_step + 2)]  # -0.5, 0.5, ..., max_step+0.5

    plt.figure()
    plt.hist(steps, bins=bins, align="mid")
    plt.xlabel("Steps/Epochs executed (estimated from intermediate_values)")
    plt.ylabel("Number of trials")
    plt.title("Distribution of training length per trial (pruning effect)")

    # Put tick marks at integer step values
    plt.xticks(range(0, max_step + 1))

    path = outdir / "steps_executed_hist.png"
    plt.savefig(path, bbox_inches="tight", dpi=200)
    plt.close()
    return path

def write_summary_txt(summary: str, outdir: Path) -> Path:
    p = outdir / "summary.txt"
    p.write_text(summary, encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-name", default="mnist_pruning_demo_cnn_acc_1")
    ap.add_argument(
        "--storage",
        default="sqlite:///optuna.db",
        help='Optuna storage URL, e.g. "sqlite:///optuna.db"',
    )
    ap.add_argument("--outdir", default="reports", help="Output directory for reports")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    safe_makedirs(outdir)

    print(f"Loading study: {args.study_name}")
    print(f"Storage:      {args.storage}")
    study = load_study(study_name=args.study_name, storage=args.storage)

    # 2) Quantitative summary
    summary = summarize_trials(study)
    print("\n" + summary)
    summary_path = write_summary_txt(summary, outdir)
    print(f"\nWrote summary: {summary_path}")

    # 1) Optuna built-in plots (HTML)
    try:
        plot_paths = save_optuna_plots_html(study, outdir)
        for name, p in plot_paths:
            print(f"Wrote plot: {name:>22} -> {p}")
    except Exception as e:
        print("\nCould not write Optuna HTML plots.")
        print("Reason:", repr(e))
        print("Tip: ensure plotly is installed: pip install plotly")

    # 3) Custom histogram of executed steps/epochs (PNG)
    steps = steps_executed_per_trial(study)
    hist_path = save_steps_histogram_png(steps, outdir)
    print(f"Wrote histogram:            -> {hist_path}")

    print("\nDone. Open the HTML files in your browser; include screenshots in your README.")


if __name__ == "__main__":
    main()