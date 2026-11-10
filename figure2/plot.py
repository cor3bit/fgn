import os

os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from common import artifact_root, setup_plot_style
from configs import CONFIG_COLUMNS_2, METHOD_LABELS_2, METHOD_ORDER_2

COLOR_ADAM = "0.42"
COLOR_FGN = "C1"
COLOR_SGN = "C2"

METHOD_COLORS_2 = {
    "adam": COLOR_ADAM,
    "fgn": COLOR_FGN,
    "sgn": COLOR_SGN,
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot Figure 2 frozen-feature head-optimization artifacts.")
    p.add_argument("--runs-csv", type=Path, default=artifact_root() / "figure2" / "raw" / "runs.csv")
    p.add_argument("--traces-eval-csv", type=Path, default=artifact_root() / "figure2" / "raw" / "traces_eval.csv")
    p.add_argument("--out-dir-derived", type=Path, default=artifact_root() / "figure2" / "derived")
    p.add_argument("--out-dir-plots", type=Path, default=artifact_root() / "figure2" / "plots")
    p.add_argument("--dataset-id", type=str, default=None)
    p.add_argument("--encoder-name", type=str, default=None)
    p.add_argument("--common-points", type=int, default=240)
    p.add_argument("--band", type=str, default="std", choices=["none", "std", "sem"])
    p.add_argument(
        "--band-alpha",
        type=float,
        default=0.24,
        help="Opacity for variability bands.",
    )
    p.add_argument(
        "--acc-ymin",
        type=float,
        default=37.0,
        help="Accuracy y-axis lower bound in percent. Use NaN for automatic limits.",
    )
    p.add_argument(
        "--acc-ymax",
        type=float,
        default=45.3,
        help="Accuracy y-axis upper bound in percent. Use NaN for automatic limits.",
    )
    p.add_argument(
        "--time-to-acc-thresholds",
        nargs="*",
        type=float,
        default=[42.0, 43.0, 44.0],
        help="Accuracy thresholds, in percent, for time-to-accuracy summaries.",
    )
    p.add_argument("--time-limit-s", type=float, default=None)
    p.add_argument("--yscale", type=str, default="linear", choices=["linear", "log"])
    p.add_argument("--optimizer-order", nargs="*", default=METHOD_ORDER_2)
    return p.parse_args()


def require_columns(df: pd.DataFrame, cols: list[str], context: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def load_csv(path: Path, context: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {context} CSV: {path}")
    return pd.read_csv(path)


def method_label(method_id: str) -> str:
    return METHOD_LABELS_2.get(str(method_id), str(method_id))


def method_sort_key(method_id: str):
    order = {m: i for i, m in enumerate(METHOD_ORDER_2)}
    s = str(method_id)
    return (order.get(s, 999), s)


def get_method_order(df: pd.DataFrame, user_order: list[str] | None) -> list[str]:
    methods = list(df["method_id"].dropna().unique())
    if user_order:
        present = [m for m in user_order if m in methods]
        rest = [m for m in methods if m not in present]
        return present + sorted(rest, key=method_sort_key)
    return sorted(methods, key=method_sort_key)


def sanitise_tag(x: str) -> str:
    return str(x).replace("/", "_").replace(" ", "_").replace(":", "_")


def dataset_tag_from_frame(df: pd.DataFrame) -> str:
    if "dataset_id" not in df.columns:
        return "unknown_dataset"
    vals = sorted(str(x) for x in df["dataset_id"].dropna().unique())
    if len(vals) == 0:
        return "unknown_dataset"
    if len(vals) == 1:
        return sanitise_tag(vals[0])
    return "multi_dataset"


def choose_time_formatter(max_time_s: float):
    formatter = FuncFormatter(lambda x, _pos: f"{x:.3g}" if max_time_s < 0.1 else f"{x:.1f}")
    return formatter, "Wall time (s)"


def interpolate_trace(times: np.ndarray, values: np.ndarray, common_time: np.ndarray) -> np.ndarray:
    """Interpolate one run onto a shared time grid.

    The caller must choose common_time so that it lies inside this run's observed
    time range. We check this explicitly to avoid silent flat extrapolation by
    np.interp at the right boundary.
    """
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    common_time = np.asarray(common_time, dtype=float)

    order = np.argsort(times)
    times = times[order]
    values = values[order]

    finite = np.isfinite(times) & np.isfinite(values)
    times = times[finite]
    values = values[finite]

    if len(times) == 0:
        return np.full_like(common_time, np.nan, dtype=float)

    if times[0] > 0.0:
        times = np.insert(times, 0, 0.0)
        values = np.insert(values, 0, values[0])

    uniq_t, uniq_idx = np.unique(times, return_index=True)
    uniq_v = values[uniq_idx]

    if len(uniq_t) < 2:
        if np.allclose(common_time, uniq_t[0], rtol=0.0, atol=1e-12):
            return np.full_like(common_time, uniq_v[0], dtype=float)
        raise ValueError(
            "Cannot interpolate a run with fewer than two unique time points "
            f"onto grid [{common_time[0]}, {common_time[-1]}]."
        )

    lo = float(common_time[0])
    hi = float(common_time[-1])
    t_min = float(uniq_t[0])
    t_max = float(uniq_t[-1])
    eps = 1e-9 * max(1.0, abs(t_max))

    if lo < t_min - eps or hi > t_max + eps:
        raise ValueError(
            "common_time lies outside run support: "
            f"grid=[{lo:.9g}, {hi:.9g}], run=[{t_min:.9g}, {t_max:.9g}]. "
            "Do not truncate the per-run trace before interpolation."
        )

    return np.interp(common_time, uniq_t, uniq_v)


def config_columns(df_runs: pd.DataFrame) -> list[str]:
    return [c for c in CONFIG_COLUMNS_2 if c in df_runs.columns]


def assert_single_config_per_method(df_runs: pd.DataFrame):
    cfg_cols = config_columns(df_runs)
    if not cfg_cols:
        return
    counts = (
        df_runs[cfg_cols]
        .drop_duplicates()
        .groupby("method_id", dropna=False)
        .size()
    )
    bad = counts[counts > 1]
    if len(bad):
        raise ValueError(
            "Final plot input contains multiple configurations per method. "
            "Use a clean artifact directory with fixed METHOD_KWARGS_2 settings. "
            "Offending counts: "
            f"{bad.to_dict()}"
        )


def build_curves(selected_runs: pd.DataFrame, df_eval: pd.DataFrame, common_points: int, time_limit_s: float | None):
    """Build mean/band curves on one shared wall-clock grid.

    Protocol:
      1. Gather all selected runs for each method.
      2. Compute each method's local supported horizon as the shortest run
         duration across its seeds.
      3. Compute a single global horizon shared across methods.
      4. Interpolate every run onto the same common_time grid.
      5. Compute mean/std/sem across seeds at each shared time point.

    Important: we do not pre-filter a run to wall_clock_s <= global_limit before
    interpolation. Keeping the first point after global_limit lets linear
    interpolation evaluate correctly at the boundary instead of silently
    plateauing at the last earlier evaluation point.
    """
    require_columns(df_eval, ["run_name", "method_id", "seed", "wall_clock_s", "val_loss", "val_acc"], "eval traces")
    require_columns(selected_runs, ["run_name", "method_id", "seed"], "selected runs")

    selected_names = set(selected_runs["run_name"].tolist())
    df_eval = df_eval[df_eval["run_name"].isin(selected_names)].copy()
    if set(df_eval["run_name"].unique()) != selected_names:
        raise ValueError("Every selected run must have evaluation traces")
    if df_eval.duplicated(["run_name", "step"]).any():
        raise ValueError("Duplicate evaluation records")
    if not np.isfinite(df_eval[["wall_clock_s", "val_loss", "val_acc"]].to_numpy(dtype=float)).all():
        raise ValueError("Evaluation traces must be finite")

    per_method: dict[str, dict] = {}
    local_limits: list[float] = []

    # ------------------------------------------------------------------
    # Pass 1: collect traces and compute local method horizons.
    # ------------------------------------------------------------------
    for method_id in get_method_order(selected_runs, METHOD_ORDER_2):
        method_runs = selected_runs[selected_runs["method_id"] == method_id].copy()
        run_names = method_runs["run_name"].tolist()
        trace_subset = df_eval[df_eval["run_name"].isin(run_names)].copy()

        if trace_subset.empty:
            print(f"Figure2 {method_id}: no eval traces for selected config")
            continue

        run_frames = []
        run_max_times = []

        for run_name, g in trace_subset.groupby("run_name", sort=False):
            g = g.sort_values("wall_clock_s").copy()
            if g.empty:
                continue

            times = pd.to_numeric(g["wall_clock_s"], errors="coerce").to_numpy(dtype=float)
            times = times[np.isfinite(times)]
            if len(times) == 0:
                continue

            run_max_t = float(times.max())
            effective_max_t = min(run_max_t, float(time_limit_s)) if time_limit_s is not None else run_max_t

            if effective_max_t <= 0.0:
                continue

            run_max_times.append(effective_max_t)
            run_frames.append((run_name, g))

        if run_frames:
            local_limit = float(min(run_max_times))
            per_method[method_id] = {
                "runs": method_runs,
                "frames": run_frames,
                "local_limit": local_limit,
            }
            local_limits.append(local_limit)

    if not per_method:
        return pd.DataFrame(), pd.DataFrame()

    global_limit = float(min(local_limits))
    if time_limit_s is not None:
        global_limit = min(global_limit, float(time_limit_s))

    if global_limit <= 0.0:
        raise ValueError(f"Nonpositive global comparison horizon: {global_limit}")

    common_time = np.linspace(0.0, global_limit, int(common_points))

    print(
        "Figure2 global comparison horizon: "
        f"{global_limit:.3f}s "
        f"(local limits: "
        + ", ".join(f"{m}={payload['local_limit']:.3f}s" for m, payload in per_method.items())
        + ")"
    )

    curve_rows = []
    summary_rows = []

    # ------------------------------------------------------------------
    # Pass 2: interpolate every run on the same global grid.
    # ------------------------------------------------------------------
    for method_id in get_method_order(selected_runs, METHOD_ORDER_2):
        if method_id not in per_method:
            continue

        payload = per_method[method_id]
        loss_interp = []
        acc_interp = []

        for run_name, g in payload["frames"]:
            # Do NOT filter to <= global_limit here: if global_limit falls
            # between two evaluation points, the later point is needed for
            # correct linear interpolation.
            times = g["wall_clock_s"].to_numpy(dtype=float)
            losses = g["val_loss"].to_numpy(dtype=float)
            accs = g["val_acc"].to_numpy(dtype=float)

            loss_interp.append(interpolate_trace(times, losses, common_time))
            acc_interp.append(interpolate_trace(times, accs, common_time))

        if not loss_interp:
            print(f"Figure2 {method_id}: interpolation produced no runs")
            continue

        L = np.stack(loss_interp, axis=1)
        A = np.stack(acc_interp, axis=1)

        loss_mean = np.nanmean(L, axis=1)
        loss_std = np.nanstd(L, axis=1)
        loss_sem = loss_std / np.sqrt(L.shape[1])

        acc_mean = np.nanmean(A, axis=1)
        acc_std = np.nanstd(A, axis=1)
        acc_sem = acc_std / np.sqrt(A.shape[1])

        first_run = payload["runs"].iloc[0]

        for t, lm, ls, le, am, ast, ase in zip(
                common_time,
                loss_mean,
                loss_std,
                loss_sem,
                acc_mean,
                acc_std,
                acc_sem,
        ):
            curve_rows.append({
                "dataset_id": first_run.get("dataset_id", None),
                "encoder_name": first_run.get("encoder_name", None),
                "method_id": method_id,
                "method_label": method_label(method_id),
                "wall_clock_s": float(t),
                "val_loss_mean": float(lm),
                "val_loss_std": float(ls),
                "val_loss_sem": float(le),
                "val_acc_mean": float(am),
                "val_acc_std": float(ast),
                "val_acc_sem": float(ase),
                "n_seeds": int(L.shape[1]),
                "local_time_limit_s": float(payload["local_limit"]),
                "global_time_limit_s": float(global_limit),
                "curve_time_grid": "global_shared_wall_clock",
            })

        summary_rows.append({
            "dataset_id": first_run.get("dataset_id", None),
            "encoder_name": first_run.get("encoder_name", None),
            "method_id": method_id,
            "method_label": method_label(method_id),
            "n_seeds": int(L.shape[1]),
            "local_time_limit_s": float(payload["local_limit"]),
            "global_time_limit_s": float(global_limit),
            "curve_time_grid": "global_shared_wall_clock",
            "final_curve_val_loss_mean": float(loss_mean[-1]),
            "final_curve_val_loss_std": float(loss_std[-1]),
            "final_curve_val_acc_mean": float(acc_mean[-1]),
            "final_curve_val_acc_std": float(acc_std[-1]),
        })

        print(
            f"Figure2 {method_id}: "
            f"acc@{global_limit:.2f}s={acc_mean[-1]:.4f} +- {acc_std[-1]:.4f} | "
            f"loss={loss_mean[-1]:.4f} +- {loss_std[-1]:.4f} | "
            f"seeds={L.shape[1]} | local_limit={payload['local_limit']:.2f}s"
        )

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def compute_loss_limits(curves_df: pd.DataFrame, band: str):
    vals = curves_df["val_loss_mean"].to_numpy(dtype=float)
    if band != "none":
        spread_col = "val_loss_std" if band == "std" else "val_loss_sem"
        spread = curves_df[spread_col].to_numpy(dtype=float)
        vals = np.concatenate([vals - spread, vals + spread])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    lo = max(0.0, float(np.nanmin(vals)) - 0.04 * max(1.0, float(np.nanmax(vals))))
    hi = float(np.nanmax(vals)) + 0.05 * max(1.0, float(np.nanmax(vals)))
    return (lo, hi if hi > lo else lo + 1.0)


def compute_acc_limits(curves_df: pd.DataFrame, band: str):
    vals = curves_df["val_acc_mean"].to_numpy(dtype=float)
    if band != "none":
        spread_col = "val_acc_std" if band == "std" else "val_acc_sem"
        spread = curves_df[spread_col].to_numpy(dtype=float)
        vals = np.concatenate([vals - spread, vals + spread])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 0.0, 1.0
    lo = max(0.0, float(np.nanmin(vals)) - 0.03)
    hi = min(1.0, float(np.nanmax(vals)) + 0.03)
    lo = max(0.0, 0.05 * np.floor(lo / 0.05))
    hi = min(1.0, 0.05 * np.ceil(hi / 0.05))
    return (float(lo), float(hi if hi > lo else min(1.0, lo + 0.1)))


def _accuracy_formatter(y, _pos):
    return f"{100.0 * y:.0f}"


def _time_to_threshold(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """First crossing time of y >= threshold, using linear interpolation.

    x is wall time in seconds. y is accuracy as a fraction, e.g. 0.44.
    Returns NaN when the threshold is not reached.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) == 0:
        return float("nan")

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    hits = np.nonzero(y >= float(threshold))[0]
    if len(hits) == 0:
        return float("nan")

    j = int(hits[0])
    if j == 0:
        return float(x[0])

    x0, x1 = float(x[j - 1]), float(x[j])
    y0, y1 = float(y[j - 1]), float(y[j])

    if y1 == y0:
        return float(x1)

    alpha = (float(threshold) - y0) / (y1 - y0)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    return float(x0 + alpha * (x1 - x0))


def build_time_to_acc_summary(curves_df: pd.DataFrame, thresholds_pct: list[float]) -> pd.DataFrame:
    rows = []
    if curves_df.empty:
        return pd.DataFrame(rows)

    for method_id in get_method_order(curves_df, METHOD_ORDER_2):
        g = curves_df[curves_df["method_id"] == method_id].sort_values("wall_clock_s")
        if g.empty:
            continue

        x = g["wall_clock_s"].to_numpy(dtype=float)
        y = g["val_acc_mean"].to_numpy(dtype=float)

        final_acc = float(y[-1]) if len(y) else float("nan")
        global_limit = (
            float(g["global_time_limit_s"].iloc[0])
            if "global_time_limit_s" in g.columns and len(g)
            else float(np.nanmax(x))
        )

        for threshold_pct in thresholds_pct:
            threshold = float(threshold_pct) / 100.0
            t = _time_to_threshold(x, y, threshold)
            rows.append(
                {
                    "method_id": method_id,
                    "method_label": g["method_label"].iloc[0],
                    "threshold_acc_pct": float(threshold_pct),
                    "threshold_acc": threshold,
                    "time_to_threshold_s": t,
                    "reached": bool(np.isfinite(t)),
                    "final_curve_acc": final_acc,
                    "final_curve_acc_pct": 100.0 * final_acc,
                    "global_time_limit_s": global_limit,
                }
            )

    return pd.DataFrame(rows)


def print_time_to_acc_summary(summary_df: pd.DataFrame):
    if summary_df.empty:
        print("Figure2 time-to-accuracy summary: no rows")
        return

    print("\n" + "=" * 88)
    print("Figure2 time-to-accuracy summary from mean curves")
    print("=" * 88)

    for method_id in get_method_order(summary_df, METHOD_ORDER_2):
        g = summary_df[summary_df["method_id"] == method_id]
        if g.empty:
            continue

        label = g["method_label"].iloc[0]
        parts = []
        for _, row in g.sort_values("threshold_acc_pct").iterrows():
            thr = float(row["threshold_acc_pct"])
            if bool(row["reached"]):
                parts.append(f"{thr:.0f}%: {float(row['time_to_threshold_s']):.3f}s")
            else:
                parts.append(f"{thr:.0f}%: not reached")
        final_acc = float(g["final_curve_acc_pct"].iloc[0])
        print(f"{label}: " + " | ".join(parts) + f" | final={final_acc:.2f}%")


def _plot_metric_curves(
        curves_df,
        out_dir_plots,
        optimizer_order,
        band,
        yscale,
        metric,
        out_stem,
        band_alpha: float,
        acc_ymin: float | None,
        acc_ymax: float | None,
):
    if curves_df.empty:
        raise ValueError("No Figure 2 curve data available to plot.")
    if metric == "acc":
        mean_col, std_col, sem_col, ylabel = "val_acc_mean", "val_acc_std", "val_acc_sem", "Test accuracy"
    elif metric == "loss":
        mean_col, std_col, sem_col, ylabel = "val_loss_mean", "val_loss_std", "val_loss_sem", "Test cross-entropy"
    else:
        raise ValueError(metric)
    setup_plot_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 4.6), constrained_layout=True)
    for method_id in get_method_order(curves_df, optimizer_order):
        g = curves_df[curves_df["method_id"] == method_id].sort_values("wall_clock_s")
        if g.empty:
            continue
        x = g["wall_clock_s"].to_numpy(dtype=float)
        y = g[mean_col].to_numpy(dtype=float)
        color = METHOD_COLORS_2.get(method_id, None)
        is_reference = method_id == "adam"
        line, = ax.plot(
            x, y,
            label=g["method_label"].iloc[0],
            color=color,
            linestyle="--" if is_reference else "-",
            linewidth=1.55 if is_reference else 2.1,
            alpha=0.95 if is_reference else 1.0,
        )

        if band != "none":
            spread_col = std_col if band == "std" else sem_col
            spread = np.nan_to_num(g[spread_col].to_numpy(dtype=float), nan=0.0)
            if np.any(spread > 0.0):
                ax.fill_between(x, y - spread, y + spread, alpha=float(band_alpha), color=line.get_color(), linewidth=0)
    formatter, xlabel = choose_time_formatter(float(curves_df["wall_clock_s"].max()))
    ax.xaxis.set_major_formatter(formatter)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if metric == "acc":
        ax.yaxis.set_major_formatter(FuncFormatter(_accuracy_formatter))
        if acc_ymin is not None and acc_ymax is not None:
            ax.set_ylim(float(acc_ymin) / 100.0, float(acc_ymax) / 100.0)
        else:
            ax.set_ylim(*compute_acc_limits(curves_df, band=band))
    else:
        ax.set_yscale(yscale)
        if yscale == "linear":
            ax.set_ylim(*compute_loss_limits(curves_df, band=band))
    ax.grid(True)
    ax.legend(loc="best", frameon=False)
    tag = dataset_tag_from_frame(curves_df)
    out_dir_plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir_plots / f"{out_stem}__{tag}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir_plots / f"{out_stem}__{tag}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_figure2(
        curves_df,
        out_dir_plots,
        optimizer_order,
        band,
        yscale,
        band_alpha: float,
        acc_ymin: float | None,
        acc_ymax: float | None,
):
    _plot_metric_curves(
        curves_df,
        out_dir_plots,
        optimizer_order,
        band,
        yscale,
        "acc",
        "figure2_headopt",
        band_alpha=band_alpha,
        acc_ymin=acc_ymin,
        acc_ymax=acc_ymax,
    )
    _plot_metric_curves(
        curves_df,
        out_dir_plots,
        optimizer_order,
        band,
        yscale,
        "loss",
        "figure2_headopt_loss",
        band_alpha=band_alpha,
        acc_ymin=None,
        acc_ymax=None,
    )


def main():
    args = parse_args()
    args.out_dir_derived.mkdir(parents=True, exist_ok=True)
    args.out_dir_plots.mkdir(parents=True, exist_ok=True)
    df_runs = load_csv(args.runs_csv, "runs")
    df_eval = load_csv(args.traces_eval_csv, "eval traces")
    if args.dataset_id is not None:
        df_runs = df_runs[df_runs["dataset_id"] == args.dataset_id].copy()
        df_eval = df_eval[df_eval["dataset_id"] == args.dataset_id].copy()
    if args.encoder_name is not None:
        df_runs = df_runs[df_runs["encoder_name"] == args.encoder_name].copy()
        df_eval = df_eval[df_eval["encoder_name"] == args.encoder_name].copy()
    if args.common_points < 2:
        raise ValueError("--common-points must be at least 2")
    if df_runs.empty:
        raise ValueError("No Figure 2 runs left after dataset/encoder filtering.")
    selected_runs = df_runs
    if selected_runs["method_id"].isna().any() or selected_runs["seed"].isna().any():
        raise ValueError("Run identity is incomplete")
    if selected_runs.duplicated(["method_id", "seed"]).any():
        raise ValueError("Duplicate method/seed runs; choose a clean artifact directory")
    if "is_unstable" in selected_runs and selected_runs["is_unstable"].astype(str).str.lower().isin(["true", "1"]).any():
        raise ValueError("Unstable runs cannot be included in the main figure")
    if "n_epochs" in selected_runs and "n_batches" in selected_runs:
        expected_steps = selected_runs["n_epochs"] * selected_runs["n_batches"]
        if (selected_runs["steps_completed"] != expected_steps).any():
            raise ValueError("Incomplete runs cannot be included in the main figure")
    if selected_runs["dataset_id"].nunique() != 1 or selected_runs["encoder_name"].nunique() != 1:
        raise ValueError("Choose one dataset and encoder before aggregation")
    seed_sets = selected_runs.groupby("method_id")["seed"].apply(lambda values: frozenset(values))
    if len(set(seed_sets)) != 1:
        raise ValueError("Methods must use the same seed set")
    assert_single_config_per_method(selected_runs)
    curves_df, summary_df = build_curves(selected_runs, df_eval, int(args.common_points), args.time_limit_s)
    tag = dataset_tag_from_frame(curves_df)

    acc_ymin = None if args.acc_ymin is None or not np.isfinite(float(args.acc_ymin)) else float(args.acc_ymin)
    acc_ymax = None if args.acc_ymax is None or not np.isfinite(float(args.acc_ymax)) else float(args.acc_ymax)
    if (acc_ymin is None) != (acc_ymax is None):
        raise ValueError("Set both --acc-ymin and --acc-ymax, or set both to NaN for automatic limits.")
    if acc_ymin is not None and acc_ymax is not None and acc_ymax <= acc_ymin:
        raise ValueError("--acc-ymax must be larger than --acc-ymin")

    time_to_acc_df = build_time_to_acc_summary(
        curves_df,
        thresholds_pct=[float(x) for x in args.time_to_acc_thresholds],
    )
    print_time_to_acc_summary(time_to_acc_df)

    selected_runs.to_csv(args.out_dir_derived / f"figure2_selected_runs__{tag}.csv", index=False)
    curves_df.to_csv(args.out_dir_derived / f"figure2_curve_points__{tag}.csv", index=False)
    summary_df.to_csv(args.out_dir_derived / f"figure2_curve_summary__{tag}.csv", index=False)
    time_to_acc_df.to_csv(args.out_dir_derived / f"figure2_time_to_acc__{tag}.csv", index=False)
    final_summary = selected_runs.groupby("method_id", sort=False).agg(
        n_seeds=("seed", "nunique"),
        final_test_accuracy_mean=("final_val_acc", "mean"),
        final_test_accuracy_std=("final_val_acc", lambda x: x.std(ddof=0)),
        final_test_loss_mean=("final_val_loss", "mean"),
        train_time_s_mean=("train_wall_clock_s", "mean"),
        train_time_s_std=("train_wall_clock_s", lambda x: x.std(ddof=0)),
        step_time_ms_mean=("step_time_mean_ms", "mean"),
    ).reset_index()
    final_summary.to_csv(args.out_dir_derived / f"figure2_final_summary__{tag}.csv", index=False)

    plot_figure2(
        curves_df,
        args.out_dir_plots,
        args.optimizer_order,
        args.band,
        args.yscale,
        band_alpha=float(args.band_alpha),
        acc_ymin=acc_ymin,
        acc_ymax=acc_ymax,
    )
    print(f"[ok] wrote Figure 2 derived artifacts to {args.out_dir_derived} with tag={tag}")
    print(f"[ok] wrote Figure 2 plots to {args.out_dir_plots} with tag={tag}")


if __name__ == "__main__":
    main()
