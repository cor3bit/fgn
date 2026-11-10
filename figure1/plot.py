import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import artifact_root, safe_mean, safe_sem, safe_std, setup_plot_style
from configs import METHOD_LABELS_1A, METHOD_ORDER_1A

PANEL_TITLES = {
    "1a": "(a) Practical bottleneck",
    "1b": "(b) Trace consequence",
    "1c": "(c) Damped-step consequence",
}

# Explicit colors keep semantics stable across panels
COLOR_FULL = "C2"  # Full softmax GGN / SGN
COLOR_FGN = "C1"  # FGN / retained true-vs-rest
COLOR_DROPPED = "C0"  # Dropped competitor covariance (blue)
COLOR_BASELINE = "0.42"  # Adam reference


def parse_args():
    p = argparse.ArgumentParser(description="Plot Figure 1 panel artifacts.")
    p.add_argument("--panel", type=str, default="all", choices=["1a", "1b", "1c", "triptych", "all"])
    p.add_argument("--runs-csv", type=Path, default=artifact_root() / "raw/1a/runs.csv")
    p.add_argument("--panel-1b-csv", type=Path, default=artifact_root() / "raw/1b/decomposition_traces.csv")
    p.add_argument("--panel-1c-csv", type=Path, default=artifact_root() / "raw/1c/decrease_ratios.csv")
    p.add_argument("--out-dir-derived", type=Path, default=artifact_root() / "derived")
    p.add_argument("--out-dir-plots", type=Path, default=artifact_root() / "plots")
    p.add_argument("--band", type=str, default="none", choices=["none", "std", "sem"])
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def load_csv(path: Path, context: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context} CSV: {path}")
    return pd.read_csv(path)


def aggregate_1a(df_runs: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df_runs,
        ["method_id", "seed", "n_classes", "n_params", "step_time_median_ms", "is_unstable"],
        "1a runs",
    )

    if df_runs.empty:
        raise ValueError("1a runs CSV is empty")
    unknown = set(df_runs["method_id"]) - set(METHOD_ORDER_1A)
    if unknown:
        raise ValueError(f"Unknown timing methods: {sorted(unknown)}")
    if df_runs.duplicated(["method_id", "n_classes", "seed"]).any():
        raise ValueError("Duplicate method/class/seed records in timing data")
    if df_runs["is_unstable"].astype(str).str.lower().isin(["true", "1"]).any():
        raise ValueError("Timing data contain unstable runs; inspect runs.csv before plotting")
    times = pd.to_numeric(df_runs["step_time_median_ms"], errors="coerce")
    if not np.all(np.isfinite(times) & (times > 0.0)):
        raise ValueError("Timing data must contain finite, positive step times")
    seed_sets = {tuple(sorted(g["seed"].unique())) for _, g in df_runs.groupby(["method_id", "n_classes"])}
    class_sets = {tuple(sorted(g["n_classes"].unique())) for _, g in df_runs.groupby("method_id")}
    if len(seed_sets) != 1 or len(class_sets) != 1:
        raise ValueError("Incomplete timing grid: methods/classes must use the same seeds and class counts")
    protocol_columns = [
        "feat_dim", "batch_size", "n_samples", "learning_rate", "weight_decay", "lam0",
        "cg_maxiter", "cg_tol", "cg_stabilise_every", "cg_warm_start", "cg_lifted",
        "warmup_steps", "timed_steps", "measure_every", "device", "somax_version", "jax_version",
    ]
    for method, runs in df_runs.groupby("method_id"):
        for column in protocol_columns:
            if column in runs and runs[column].nunique(dropna=False) != 1:
                raise ValueError(f"Mixed {column} settings for {method}; use separate artifacts directories")

    return (
        df_runs.groupby(["method_id", "n_classes"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            n_params_mean=("n_params", safe_mean),
            step_ms_mean=("step_time_median_ms", safe_mean),
            step_ms_std=("step_time_median_ms", safe_std),
            step_ms_sem=("step_time_median_ms", safe_sem),
            unstable_frac=("is_unstable", safe_mean),
        )
        .sort_values(["method_id", "n_classes"])
    )


def summarize_1b(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "n_classes", "p_star", "p_dagger", "xi", "trace_full",
            "trace_retained", "trace_dropped", "abs_sum_error", "abs_dropped_error",
        ],
        "1b traces",
    )
    return pd.DataFrame([
        {
            "n_classes": int(df["n_classes"].iloc[0]),
            "p_star": float(df["p_star"].iloc[0]),
            "p_dagger": float(df["p_dagger"].iloc[0]),
            "n_points": int(len(df)),
            "xi_min": float(df["xi"].min()),
            "xi_max": float(df["xi"].max()),
            "max_abs_sum_error": float(pd.to_numeric(df["abs_sum_error"], errors="coerce").max()),
            "max_abs_dropped_error": float(pd.to_numeric(df["abs_dropped_error"], errors="coerce").max()),
        }
    ])


def summarize_1c(df: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        df,
        [
            "lambda_regime_label", "lambda_regime_order", "lambda_scale", "lambda",
            "lambda_reference_trace_H_fgn_over_dim", "xi", "full_model_decrease_ratio",
            "full_model_decrease_fgn", "full_model_decrease_ggn", "min_eig_damped_fgn",
            "min_eig_damped_ggn",
        ],
        "1c full-GGN predicted-decrease ratios",
    )
    rows = []
    for label, g in df.groupby("lambda_regime_label", sort=False):
        g = g.sort_values("xi")
        ratios = pd.to_numeric(g["full_model_decrease_ratio"], errors="coerce")
        rows.append({
            "lambda_regime_label": str(label),
            "lambda_regime_order": int(g["lambda_regime_order"].iloc[0]),
            "lambda_scale": float(g["lambda_scale"].iloc[0]),
            "lambda_reference_trace_H_fgn_over_dim": float(g["lambda_reference_trace_H_fgn_over_dim"].iloc[0]),
            "lambda": float(g["lambda"].iloc[0]),
            "n_points": int(len(g)),
            "xi_min": float(g["xi"].min()),
            "xi_max": float(g["xi"].max()),
            "min_full_model_decrease_ratio": float(ratios.min()),
            "max_full_model_decrease_ratio": float(ratios.max()),
            "ratio_nonincreasing": bool(np.all(np.diff(ratios.to_numpy(dtype=float)) <= 1e-10)),
            "min_full_model_decrease_fgn": float(pd.to_numeric(g["full_model_decrease_fgn"], errors="coerce").min()),
            "min_full_model_decrease_ggn": float(pd.to_numeric(g["full_model_decrease_ggn"], errors="coerce").min()),
            "min_eig_damped_fgn": float(pd.to_numeric(g["min_eig_damped_fgn"], errors="coerce").min()),
            "min_eig_damped_ggn": float(pd.to_numeric(g["min_eig_damped_ggn"], errors="coerce").min()),
        })
    return pd.DataFrame(rows).sort_values("lambda_regime_order")


def format_class_tick(x, _pos=None):
    x = float(x)
    if x <= 0.0:
        return ""
    k = round(math.log2(x))
    if abs(x - (2 ** k)) < 1e-9:
        return rf"$2^{{{k}}}$"
    return str(int(x))


def clean_regime_label(label: str) -> str:
    label = str(label).strip()
    if label:
        return label
    return "damping"


def set_panel_title(ax, panel_id: str):
    ax.set_title(PANEL_TITLES[panel_id], loc="center", pad=14, fontsize=12.5)


def draw_1a_on_ax(ax, agg: pd.DataFrame, band: str, show_legend: bool = True):
    for method_id in METHOD_ORDER_1A:
        g = agg[agg["method_id"] == method_id].sort_values("n_classes")
        if g.empty:
            continue

        x = g["n_classes"].to_numpy(dtype=float)
        y = g["step_ms_mean"].to_numpy(dtype=float)
        label = METHOD_LABELS_1A[method_id]

        style = {"marker": "o", "label": label}
        if method_id == "adam_ref":
            style.update({"linestyle": "--", "linewidth": 1.55, "color": COLOR_BASELINE, "alpha": 0.95})
        elif method_id == "fgn":
            style.update({"linewidth": 2.1, "color": COLOR_FGN})
        elif method_id == "sgn":
            style.update({"linewidth": 2.1, "color": COLOR_FULL})
        else:
            style.update({"linewidth": 2.1})

        line, = ax.plot(x, y, **style)

        if band != "none":
            y_band = g["step_ms_std"].to_numpy(dtype=float) if band == "std" else g["step_ms_sem"].to_numpy(dtype=float)
            y_band = np.nan_to_num(y_band, nan=0.0)
            if np.any(y_band > 0.0):
                ax.fill_between(x, y - y_band, y + y_band, alpha=0.11, color=line.get_color(), linewidth=0)

    xticks = sorted(agg["n_classes"].unique().tolist())
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks)
    ax.set_xticklabels([format_class_tick(x) for x in xticks])
    ax.set_xlabel(r"Number of classes $C$")
    ax.set_ylabel("Step time (ms)")
    ax.grid(True)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)
    set_panel_title(ax, "1a")


def draw_1b_on_ax(ax, df: pd.DataFrame, show_legend: bool = True):
    require_columns(
        df,
        ["xi", "trace_full", "trace_retained", "trace_dropped"],
        "1b traces",
    )
    g = df.sort_values("xi")
    x = g["xi"].to_numpy(dtype=float)
    full = g["trace_full"].to_numpy(dtype=float)
    retained = g["trace_retained"].to_numpy(dtype=float)
    dropped = g["trace_dropped"].to_numpy(dtype=float)

    ax.plot(x, full, label="full softmax trace", color=COLOR_FULL)
    ax.plot(x, retained, label="retained true-vs-rest", color=COLOR_FGN)
    ax.plot(x, dropped, label=r"dropped $p_\dagger\xi$", linestyle="--", color=COLOR_DROPPED)
    ax.fill_between(x, retained, full, color=COLOR_DROPPED, alpha=0.10, linewidth=0)

    if len(x) > 3:
        idx = int(0.68 * (len(x) - 1))
        ax.text(
            x[idx],
            0.5 * (full[idx] + retained[idx]),
            "dropped trace",
            fontsize=8.5,
            color="0.25",
            alpha=0.78,
            ha="center",
            va="center",
        )

    ax.set_xlabel(r"Competitor dispersion $\xi$")
    ax.set_ylabel("Trace")
    ax.grid(True)
    if show_legend:
        ax.legend(loc="upper left", frameon=False)
    set_panel_title(ax, "1b")


def draw_1c_on_ax(ax, df: pd.DataFrame, show_legend: bool = True):
    require_columns(
        df,
        [
            "xi", "full_model_decrease_ratio", "lambda_regime_label",
            "lambda_regime_order",
        ],
        "1c full-GGN predicted-decrease ratios",
    )
    df = df.sort_values(["lambda_regime_order", "xi"])

    for label, g in df.groupby("lambda_regime_label", sort=False):
        g = g.sort_values("xi")
        x = g["xi"].to_numpy(dtype=float)
        y = g["full_model_decrease_ratio"].to_numpy(dtype=float)
        ax.plot(
            x,
            y,
            marker="o",
            markevery=max(1, len(x) // 8),
            label=clean_regime_label(label),
        )

    y_all = pd.to_numeric(df["full_model_decrease_ratio"], errors="coerce").to_numpy(dtype=float)
    y_min = float(np.nanmin(y_all))
    y_max = float(np.nanmax(y_all))
    ax.axhline(1.0, linestyle=":", linewidth=1.0, alpha=0.45)
    ax.set_ylim(y_min - 0.04, min(1.02, y_max + 0.025))
    ax.set_xlabel(r"Competitor dispersion $\xi$")
    ax.set_ylabel("Predicted decrease ratio")
    ax.grid(True)
    if show_legend:
        ax.legend(loc="lower left", frameon=False)
    set_panel_title(ax, "1c")


def plot_1a(agg: pd.DataFrame, out_dir_plots: Path, band: str):
    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.6), constrained_layout=True)
    draw_1a_on_ax(ax=ax, agg=agg, band=band, show_legend=True)

    out_dir_plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir_plots / "figure1_1a.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir_plots / "figure1_1a.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_1b(df: pd.DataFrame, out_dir_plots: Path):
    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.3), constrained_layout=True)
    draw_1b_on_ax(ax=ax, df=df, show_legend=True)

    out_dir_plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir_plots / "figure1_1b.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir_plots / "figure1_1b.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_1c(df: pd.DataFrame, out_dir_plots: Path):
    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.4, 4.3), constrained_layout=True)
    draw_1c_on_ax(ax=ax, df=df, show_legend=True)

    out_dir_plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir_plots / "figure1_1c.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir_plots / "figure1_1c.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_triptych(agg_1a: pd.DataFrame, df_1b: pd.DataFrame, df_1c: pd.DataFrame, out_dir_plots: Path, band: str):
    setup_plot_style()
    fig, axs = plt.subplots(1, 3, figsize=(16.2, 4.35), constrained_layout=True)

    draw_1a_on_ax(ax=axs[0], agg=agg_1a, band=band, show_legend=True)
    draw_1b_on_ax(ax=axs[1], df=df_1b, show_legend=True)
    draw_1c_on_ax(ax=axs[2], df=df_1c, show_legend=True)

    out_dir_plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir_plots / "figure1_triptych.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir_plots / "figure1_triptych.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    need_1a = args.panel in {"1a", "triptych", "all"}
    need_1b = args.panel in {"1b", "triptych", "all"}
    need_1c = args.panel in {"1c", "triptych", "all"}

    agg_1a = None
    df_1b = None
    df_1c = None

    args.out_dir_derived.mkdir(parents=True, exist_ok=True)
    args.out_dir_plots.mkdir(parents=True, exist_ok=True)

    if need_1a:
        df_runs = load_csv(args.runs_csv, "1a runs")
        agg_1a = aggregate_1a(df_runs)
        out = args.out_dir_derived / "1a"
        out.mkdir(parents=True, exist_ok=True)
        agg_1a.to_csv(out / "scaling_summary.csv", index=False)

    if need_1b:
        df_1b = load_csv(args.panel_1b_csv, "1b traces")
        out = args.out_dir_derived / "1b"
        out.mkdir(parents=True, exist_ok=True)
        summarize_1b(df_1b).to_csv(out / "decomposition_summary.csv", index=False)

    if need_1c:
        df_1c = load_csv(args.panel_1c_csv, "1c full-GGN predicted-decrease ratios")
        out = args.out_dir_derived / "1c"
        out.mkdir(parents=True, exist_ok=True)
        summarize_1c(df_1c).to_csv(out / "decrease_ratio_summary.csv", index=False)

    if args.panel in {"1a", "all"}:
        assert agg_1a is not None
        plot_1a(agg=agg_1a, out_dir_plots=args.out_dir_plots, band=args.band)

    if args.panel in {"1b", "all"}:
        assert df_1b is not None
        plot_1b(df=df_1b, out_dir_plots=args.out_dir_plots)

    if args.panel in {"1c", "all"}:
        assert df_1c is not None
        plot_1c(df=df_1c, out_dir_plots=args.out_dir_plots)

    if args.panel in {"triptych", "all"}:
        assert agg_1a is not None and df_1b is not None and df_1c is not None
        plot_triptych(
            agg_1a=agg_1a,
            df_1b=df_1b,
            df_1c=df_1c,
            out_dir_plots=args.out_dir_plots,
            band=args.band,
        )


if __name__ == "__main__":
    main()
