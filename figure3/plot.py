import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import PercentFormatter

ARTIFACTS = Path(os.environ.get("FGN_ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts")).resolve()

CONFIG_STYLE = {
    "fgn_cg5_lifted": dict(
        label="FGN", color="C1", linestyle="-", linewidth=2.2
    ),
    "sgn_cg5": dict(
        label="SGN", color="C2", linestyle="-", linewidth=2.2
    ),
    "sgdm_step": dict(
        label="SGD-M (reference)", color="0.42", linestyle="--", linewidth=1.7
    ),
}

FIGURES = (
    ("test_accuracy", "Test accuracy", "figure3_test_accuracy"),
    ("test_loss", "Test cross-entropy", "figure3_test_cross_entropy"),
    ("train_loss", "Train-monitor cross-entropy", "figure3_train_cross_entropy"),
)


def load_runs(root: Path, configs=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the runs/dataset/method/config/seed layout."""
    traces, summaries = [], []
    for path in sorted(root.glob("*/*/*/seed*/summary.json")):
        if not path.parent.name[4:].isdigit():
            continue
        if configs is not None and path.parent.parent.name not in configs:
            continue
        summary = json.loads(path.read_text())
        if summary.get("status") != "completed":
            continue
        metrics_path = path.parent / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)
        trace = pd.read_csv(metrics_path)
        trace = trace.loc[trace["event"] == "eval"].copy()
        if trace.empty:
            raise ValueError(f"no evaluation rows: {metrics_path}")
        trace["dataset"] = summary["dataset"]
        trace["optimizer"] = summary["optimizer"]
        trace["config_id"] = path.parent.parent.name
        trace["seed"] = summary["seed"]
        traces.append(trace)
        summaries.append({
            **summary,
            "config_id": path.parent.parent.name,
            "run_dir": str(path.parent),
        })

    if not traces:
        raise FileNotFoundError(f"no completed experiment runs under {root}")
    runs = pd.DataFrame(summaries)
    frame = pd.concat(traces, ignore_index=True)
    identity = ["dataset", "optimizer", "config_id", "seed"]
    if runs.duplicated(identity).any() or frame.duplicated(identity + ["step"]).any():
        raise ValueError("duplicate run or evaluation step")
    if configs is not None:
        missing = set(configs) - set(runs["config_id"])
        if missing:
            raise ValueError(f"Selected configurations have no completed runs: {sorted(missing)}")
    for dataset, rows in runs.groupby("dataset"):
        seeds = {name: tuple(sorted(group["seed"]))
                 for name, group in rows.groupby("config_id")}
        if len(set(seeds.values())) > 1:
            raise ValueError(f"Methods use different seed sets for {dataset}: {seeds}")
        if rows["steps_completed"].nunique() != 1:
            raise ValueError(f"Methods use different training horizons for {dataset}")
        traces_for_dataset = frame.loc[frame["dataset"] == dataset]
        schedules = [tuple(group["step"].sort_values())
                     for _, group in traces_for_dataset.groupby(["config_id", "seed"])]
        if len(set(schedules)) != 1:
            raise ValueError(f"Inconsistent evaluation steps across methods/seeds for {dataset}")
    return frame, runs


def plot_metric(frame: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    for config_id, style in CONFIG_STYLE.items():
        method = frame.loc[frame["config_id"] == config_id].dropna(subset=["epoch", metric])
        if method.empty:
            continue
        stats = (
            method.groupby("step")
            .agg(epoch=("epoch", "first"), mean=(metric, "mean"), std=(metric, "std"), n=(metric, "count"))
            .sort_index()
        )
        x = stats["epoch"].to_numpy(dtype=float)
        mean = stats["mean"].to_numpy(dtype=float)
        ax.plot(x, mean, **style)
        if int(stats["n"].min()) >= 2:
            std = stats["std"].to_numpy(dtype=float)
            ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=0.20, linewidth=0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    if metric == "test_accuracy":
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlim(left=0)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="best", frameon=False)
    for suffix in (".pdf", ".png"):
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ResNet-20 / CIFAR-100 curves.")

    parser.add_argument("--input-dir", type=Path, default=ARTIFACTS / "figure3" / "runs")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "figure3" / "plots")
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIG_STYLE), default=list(CONFIG_STYLE))

    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "pdf.fonttype": 42,
    })
    traces, summaries = load_runs(args.input_dir, args.configs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries.to_csv(args.output_dir / "final_results.csv", index=False)

    for dataset in sorted(traces["dataset"].unique()):
        out_dir = args.output_dir / dataset
        out_dir.mkdir(parents=True, exist_ok=True)
        subset = traces.loc[
            (traces["dataset"] == dataset)
            & traces["config_id"].isin(CONFIG_STYLE)
            ]
        for metric, label, filename in FIGURES:
            plot_metric(subset, metric, label, out_dir / filename)
    print(f"Saved CIFAR-100 plots to {args.output_dir}")


if __name__ == "__main__":
    main()
