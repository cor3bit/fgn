import argparse

import numpy as np

from common import artifact_root, write_csv_rows, write_json
from configs import DEFAULTS_1B


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 1b exact trace decomposition data.")
    p.add_argument("--n-classes", type=int, default=DEFAULTS_1B["n_classes"])
    p.add_argument("--p-star", type=float, default=DEFAULTS_1B["p_star"])
    p.add_argument("--n-points", type=int, default=DEFAULTS_1B["n_points"])
    p.add_argument("--artifact-subdir", type=str, default="1b")
    return p.parse_args()


def validate_args(args):
    if args.n_classes < 2:
        raise ValueError("--n-classes must be at least 2")
    if not (0.0 < args.p_star < 1.0):
        raise ValueError("--p-star must lie in (0, 1)")
    if args.n_points < 2:
        raise ValueError("--n-points must be at least 2")


def make_rows(*, n_classes: int, p_star: float, n_points: int, tol: float = 5e-14):
    n_comp = int(n_classes) - 1
    p_star = float(p_star)
    p_dagger = 1.0 - p_star
    xi_max = 1.0 - 1.0 / float(n_comp)

    xis = np.linspace(0.0, xi_max, int(n_points), dtype=np.float64)
    rows = []
    max_sum_err = 0.0
    max_drop_err = 0.0
    max_dense_err = 0.0

    for xi in xis:
        rho_l2_sq = 1.0 - float(xi)

        trace_retained = p_star * p_dagger * (1.0 + rho_l2_sq)
        trace_dropped = p_dagger * float(xi)
        trace_full = 2.0 * p_star * p_dagger + (p_dagger ** 2) * float(xi)
        trace_sum = trace_retained + trace_dropped
        dropped_formula = p_dagger * float(xi)

        alpha = 1.0 - np.sqrt(max(0.0, 1.0 - xi / xi_max)) if xi_max > 0 else 0.0
        rho = np.full(n_comp, alpha / n_comp)
        rho[0] += 1.0 - alpha
        p = np.concatenate([[p_star], p_dagger * rho])
        dense_full = np.trace(np.diag(p) - np.outer(p, p))
        dense_dropped = p_dagger * np.trace(np.diag(rho) - np.outer(rho, rho))
        sum_err = abs(trace_full - trace_sum)
        drop_err = abs(dense_dropped - dropped_formula)
        dense_err = abs(dense_full - trace_full)
        max_sum_err = max(max_sum_err, sum_err)
        max_drop_err = max(max_drop_err, drop_err)
        max_dense_err = max(max_dense_err, dense_err)

        if dense_err > tol:
            raise AssertionError(f"dense full trace check failed at xi={xi}: error={dense_err}")

        if sum_err > tol:
            raise AssertionError(f"trace decomposition failed at xi={xi}: error={sum_err}")
        if drop_err > tol:
            raise AssertionError(f"dropped trace formula failed at xi={xi}: error={drop_err}")

        rows.append({
            "panel": "1b",
            "study": "exact_output_space_trace_decomposition",
            "n_classes": int(n_classes),
            "n_competitors": int(n_comp),
            "p_star": p_star,
            "p_dagger": p_dagger,
            "xi": float(xi),
            "xi_max": float(xi_max),
            "rho_l2_sq": float(rho_l2_sq),
            "trace_full": float(trace_full),
            "trace_retained": float(trace_retained),
            "trace_dropped": float(trace_dropped),
            "trace_sum": float(trace_sum),
            "dropped_formula": float(dropped_formula),
            "abs_sum_error": float(sum_err),
            "abs_dropped_error": float(drop_err),
        })

    metadata = {
        "panel": "1b",
        "study": "exact_output_space_trace_decomposition",
        "construction": "direct sweep over admissible xi using theorem trace formulas",
        "n_classes": int(n_classes),
        "n_competitors": int(n_comp),
        "p_star": p_star,
        "p_dagger": p_dagger,
        "xi_min": 0.0,
        "xi_max": float(xi_max),
        "n_points": int(n_points),
        "formulas": {
            "trace_retained": "p_star * p_dagger * (2 - xi)",
            "trace_dropped": "p_dagger * xi",
            "trace_full": "2 * p_star * p_dagger + p_dagger**2 * xi",
        },
        "validation": {
            "trace_full_equals_retained_plus_dropped_max_abs_error": float(max_sum_err),
            "dense_dropped_trace_equals_p_dagger_xi_max_abs_error": float(max_drop_err),
            "dense_full_trace_equals_formula_max_abs_error": float(max_dense_err),
            "tolerance": float(tol),
        },
    }
    return rows, metadata


def main():
    args = parse_args()
    validate_args(args)

    raw_dir = artifact_root() / "raw" / args.artifact_subdir
    rows, metadata = make_rows(
        n_classes=int(args.n_classes),
        p_star=float(args.p_star),
        n_points=int(args.n_points),
    )

    out_csv = raw_dir / "decomposition_traces.csv"
    out_meta = raw_dir / "metadata.json"
    write_csv_rows(out_csv, rows)
    write_json(out_meta, metadata)

    print(f"Saved {len(rows)} rows to {out_csv}")
    print(f"Saved metadata to {out_meta}")


if __name__ == "__main__":
    main()
