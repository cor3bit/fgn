import argparse

import numpy as np

from common import artifact_root, write_csv_rows, write_json
from configs import DEFAULTS_1C

STUDY_NAME = "full_ggn_predicted_decrease_ratio"
DAMPING_RULE = "lambda = s_lambda * tr(H_fgn(xi_ref=0)) / dim; fixed across xi"


def _default_lambda_regime_spec() -> str:
    regimes = DEFAULTS_1C["lambda_regime_scales"]
    return ",".join(f"{label}:{float(scale):g}" for label, scale in regimes.items())


def parse_lambda_regime_spec(spec: str) -> list[tuple[str, float]]:
    """Parse comma-separated label:scale damping regimes."""
    if spec is None or not str(spec).strip():
        raise ValueError("lambda regime spec must be nonempty")

    regimes: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw_item in str(spec).split(","):
        item = raw_item.strip()
        if not item:
            continue

        if ":" in item:
            label_raw, scale_raw = item.rsplit(":", 1)
            label = label_raw.strip()
            scale_s = scale_raw.strip()
        else:
            scale_s = item
            label = f"s_lambda={float(scale_s):g}"

        if not label:
            raise ValueError(f"empty lambda regime label in spec: {spec!r}")
        if label in seen:
            raise ValueError(f"duplicate lambda regime label: {label!r}")

        scale = float(scale_s)
        if scale <= 0.0 or not np.isfinite(scale):
            raise ValueError(f"lambda regime scale must be positive and finite: {item!r}")

        regimes.append((label, scale))
        seen.add(label)

    if not regimes:
        raise ValueError(f"no lambda regimes parsed from spec: {spec!r}")
    return regimes


def parse_args():
    p = argparse.ArgumentParser(description="Generate Figure 1c exact dense damped-step data.")
    p.add_argument("--n-classes", type=int, default=DEFAULTS_1C["n_classes"])
    p.add_argument("--p-star", type=float, default=DEFAULTS_1C["p_star"])
    p.add_argument("--n-points", type=int, default=DEFAULTS_1C["n_points"])
    p.add_argument("--alpha-min", type=float, default=DEFAULTS_1C["alpha_min"])
    p.add_argument("--eta", type=float, default=DEFAULTS_1C["eta"])
    p.add_argument("--beta", type=float, default=DEFAULTS_1C["beta"])
    p.add_argument(
        "--lambda-regime-scales",
        type=str,
        default=_default_lambda_regime_spec(),
        help=(
            "Comma-separated label:s_lambda regimes. Each absolute lambda is "
            "chosen once as s_lambda * tr(H_fgn(xi_ref=0)) / dim, then held "
            "fixed across the xi sweep. Example: "
            "'low damping:0.01,medium damping:0.1,high damping:1.0'."
        ),
    )
    p.add_argument("--tol", type=float, default=1e-10)
    p.add_argument("--artifact-subdir", type=str, default="1c")
    return p.parse_args()


def validate_args(args):
    if args.n_classes < 3:
        raise ValueError("--n-classes must be at least 3 for a competitor covariance panel")
    if not (0.0 < args.p_star < 1.0):
        raise ValueError("--p-star must lie in (0, 1)")
    if args.n_points < 2:
        raise ValueError("--n-points must be at least 2")
    if not (0.0 < args.alpha_min < 1.0):
        raise ValueError("--alpha-min must lie in (0, 1); positive alpha keeps logits finite")
    if args.beta <= 0.0:
        raise ValueError("--beta must be positive")
    if args.tol <= 0.0:
        raise ValueError("--tol must be positive")
    parse_lambda_regime_spec(args.lambda_regime_scales)


def competitor_spread_values(n_competitors: int) -> np.ndarray:
    """Deterministic t_j with t_0=0 and uniform mean zero.

    The rho(alpha) path mixes a point mass at row 0 with the uniform
    distribution. With t_0=0 and mean(t)=0, the retained competitor mean row is
    constant as xi changes; the projected dropped covariance is what changes.
    """
    k = int(n_competitors)
    if k <= 0:
        raise ValueError("n_competitors must be positive")
    if k == 1:
        return np.zeros((1,), dtype=np.float64)

    n_pairs = (k - 1) // 2
    vals = [0.0]
    if n_pairs > 0:
        for mag in np.linspace(1.0 / n_pairs, 1.0, n_pairs, dtype=np.float64):
            vals.extend([-float(mag), float(mag)])

    while len(vals) < k:
        vals.append(0.0)

    t = np.asarray(vals[:k], dtype=np.float64)
    if abs(float(t[0])) > 1e-14:
        raise AssertionError("internal construction error: t_0 must be zero")
    if abs(float(np.mean(t))) > 1e-14:
        raise AssertionError("internal construction error: uniform mean(t) must be zero")
    return t


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    ez = np.exp(z)
    return ez / np.sum(ez)


def symmetric_part(mat: np.ndarray) -> np.ndarray:
    return 0.5 * (mat + mat.T)


def min_eig_sym(mat: np.ndarray) -> float:
    return float(np.linalg.eigvalsh(symmetric_part(mat))[0])


def full_model_quadratic_value(g: np.ndarray, H_ggn: np.ndarray, lam: float, d: np.ndarray) -> float:
    return float(g @ d + 0.5 * d @ (H_ggn @ d) + 0.5 * float(lam) * d @ d)


def full_model_predicted_decrease(g: np.ndarray, H_ggn: np.ndarray, lam: float, d: np.ndarray) -> float:
    return -full_model_quadratic_value(g, H_ggn, lam, d)


def make_geometry(*, n_classes: int, eta: float, beta: float):
    n_comp = int(n_classes) - 1
    u = np.asarray([1.0, 0.0], dtype=np.float64)
    v = np.asarray([0.0, 1.0], dtype=np.float64)

    t_values = competitor_spread_values(n_comp)
    a_star = -u + float(eta) * v
    a_comp = np.stack([u + float(beta) * t * v for t in t_values], axis=0)
    A = np.vstack([a_star[None, :], a_comp])

    e_star = np.zeros((int(n_classes),), dtype=np.float64)
    e_star[0] = 1.0
    return A, a_star, a_comp, t_values, e_star


def rho_from_alpha(alpha: float, n_competitors: int) -> np.ndarray:
    rho = float(alpha) * np.full((int(n_competitors),), 1.0 / int(n_competitors), dtype=np.float64)
    rho[0] += 1.0 - float(alpha)
    return rho


def reference_retained_scale(*, p_star: float, a_star: np.ndarray, a_comp: np.ndarray) -> tuple[float, np.ndarray]:
    """Return tr(H_fgn(xi_ref=0))/dim and H_fgn(xi_ref=0).

    The reference point is the concentrated competitor distribution rho=e_0,
    whose dispersion xi is zero. The resulting lambda values are held fixed
    across the full xi sweep.
    """
    p_dagger = 1.0 - float(p_star)
    dim = int(a_star.shape[0])
    rho_ref = np.zeros((a_comp.shape[0],), dtype=np.float64)
    rho_ref[0] = 1.0
    q_ref = rho_ref @ a_comp - a_star
    H_fgn_ref = float(p_star) * p_dagger * np.outer(q_ref, q_ref)
    return float(np.trace(H_fgn_ref)) / float(dim), H_fgn_ref


def compute_quantities(
        *,
        p_star: float,
        rho: np.ndarray,
        A: np.ndarray,
        a_star: np.ndarray,
        a_comp: np.ndarray,
        e_star: np.ndarray,
        lam: float,
        tol: float,
):
    p_dagger = 1.0 - float(p_star)
    dim = int(A.shape[1])
    eye = np.eye(dim, dtype=np.float64)

    xi = 1.0 - float(rho @ rho)
    p = np.concatenate([[float(p_star)], p_dagger * rho])
    if abs(float(np.sum(p)) - 1.0) > tol:
        raise AssertionError("probabilities do not sum to one")
    if np.any(p <= 0.0):
        raise AssertionError("probabilities must be strictly positive for finite logits")

    probs_from_bias = softmax(np.log(p))
    softmax_err = float(np.max(np.abs(probs_from_bias - p)))
    if softmax_err > 5e-13:
        raise AssertionError(f"softmax(log p) reconstruction failed: {softmax_err}")

    a_bar = rho @ a_comp
    q = a_bar - a_star
    H_fgn = float(p_star) * p_dagger * np.outer(q, q)

    Q = np.diag(p) - np.outer(p, p)
    H_ggn = A.T @ Q @ A

    cov_rho = np.diag(rho) - np.outer(rho, rho)
    H_drop = p_dagger * (a_comp.T @ cov_rho @ a_comp)

    decomp_err = float(np.max(np.abs(H_ggn - H_fgn - H_drop)))
    if decomp_err > 5e-12:
        raise AssertionError(f"parameter-space decomposition failed: {decomp_err}")

    g = A.T @ (p - e_star)
    g_ref = p_dagger * q
    grad_err = float(np.max(np.abs(g - g_ref)))
    if grad_err > 5e-12:
        raise AssertionError(f"gradient identity failed: {grad_err}")

    sym_fgn = float(np.max(np.abs(H_fgn - H_fgn.T)))
    sym_ggn = float(np.max(np.abs(H_ggn - H_ggn.T)))
    min_hf = min_eig_sym(H_fgn)
    min_hg = min_eig_sym(H_ggn)
    min_df = min_eig_sym(H_fgn + float(lam) * eye)
    min_dg = min_eig_sym(H_ggn + float(lam) * eye)

    if sym_fgn > tol or sym_ggn > tol:
        raise AssertionError("curvature matrix symmetry check failed")
    if min_hf < -tol or min_hg < -tol:
        raise AssertionError(f"PSD check failed: min_eigs=({min_hf}, {min_hg})")
    if min_df <= 0.0 or min_dg <= 0.0:
        raise AssertionError(f"damped SPD check failed: min_eigs=({min_df}, {min_dg})")

    d_fgn = -np.linalg.solve(H_fgn + float(lam) * eye, g)
    d_ggn = -np.linalg.solve(H_ggn + float(lam) * eye, g)

    decrease_fgn = full_model_predicted_decrease(g, H_ggn, float(lam), d_fgn)
    decrease_ggn = full_model_predicted_decrease(g, H_ggn, float(lam), d_ggn)
    if decrease_ggn <= 0.0:
        raise AssertionError(f"full-GGN optimal predicted decrease must be positive: {decrease_ggn}")

    ratio = float(decrease_fgn / max(decrease_ggn, 1e-300))
    gap = float(decrease_ggn - decrease_fgn)
    if not np.isfinite(ratio):
        raise AssertionError("nonfinite full-model predicted-decrease ratio")
    if ratio > 1.0 + 5e-10:
        raise AssertionError(f"full-model decrease ratio exceeds one: {ratio}")

    return {
        "xi": float(xi),
        "a_bar": a_bar,
        "H_fgn": H_fgn,
        "H_ggn": H_ggn,
        "H_drop": H_drop,
        "g": g,
        "d_fgn": d_fgn,
        "d_ggn": d_ggn,
        "full_model_decrease_fgn": float(decrease_fgn),
        "full_model_decrease_ggn": float(decrease_ggn),
        "full_model_decrease_ratio": float(ratio),
        "full_model_optimality_gap": float(gap),
        "min_eig_H_fgn": float(min_hf),
        "min_eig_H_ggn": float(min_hg),
        "min_eig_damped_fgn": float(min_df),
        "min_eig_damped_ggn": float(min_dg),
        "softmax_reconstruction_error": float(softmax_err),
        "decomposition_error": float(decomp_err),
        "gradient_identity_error": float(grad_err),
        "trace_H_fgn": float(np.trace(H_fgn)),
        "trace_H_ggn": float(np.trace(H_ggn)),
        "trace_H_drop": float(np.trace(H_drop)),
    }


def make_rows(
        *,
        n_classes: int,
        p_star: float,
        n_points: int,
        alpha_min: float,
        eta: float,
        beta: float,
        lambda_regimes: list[tuple[str, float]],
        tol: float,
):
    dim = 2
    n_comp = int(n_classes) - 1
    p_star = float(p_star)
    p_dagger = 1.0 - p_star
    xi_max = 1.0 - 1.0 / float(n_comp)

    A, a_star, a_comp, t_values, e_star = make_geometry(
        n_classes=int(n_classes),
        eta=float(eta),
        beta=float(beta),
    )
    lambda_ref_scale, H_fgn_ref = reference_retained_scale(
        p_star=p_star,
        a_star=a_star,
        a_comp=a_comp,
    )
    if lambda_ref_scale <= 0.0 or not np.isfinite(lambda_ref_scale):
        raise AssertionError(f"invalid reference damping scale: {lambda_ref_scale}")

    alphas = np.linspace(float(alpha_min), 1.0, int(n_points), dtype=np.float64)
    rows = []
    validation_by_regime = {}
    maxima = {
        "softmax_reconstruction_error": 0.0,
        "decomposition_error": 0.0,
        "gradient_identity_error": 0.0,
        "max_full_model_optimality_gap": 0.0,
        "min_eig_H_fgn": float("inf"),
        "min_eig_H_ggn": float("inf"),
        "min_eig_damped_fgn": float("inf"),
        "min_eig_damped_ggn": float("inf"),
    }

    for regime_order, (regime_label, regime_scale) in enumerate(lambda_regimes):
        lam = float(regime_scale) * float(lambda_ref_scale)
        regime_ratios = []
        regime_decreases_fgn = []
        regime_decreases_ggn = []
        regime_min_df = []
        regime_min_dg = []

        for alpha in alphas:
            rho = rho_from_alpha(float(alpha), n_comp)
            if np.any(rho <= 0.0):
                raise AssertionError("rho(alpha) must stay strictly positive for finite logits")
            if abs(float(np.sum(rho)) - 1.0) > tol:
                raise AssertionError("rho does not sum to one")

            q = compute_quantities(
                p_star=p_star,
                rho=rho,
                A=A,
                a_star=a_star,
                a_comp=a_comp,
                e_star=e_star,
                lam=lam,
                tol=float(tol),
            )

            ratio = float(q["full_model_decrease_ratio"])
            regime_ratios.append(ratio)
            regime_decreases_fgn.append(float(q["full_model_decrease_fgn"]))
            regime_decreases_ggn.append(float(q["full_model_decrease_ggn"]))
            regime_min_df.append(float(q["min_eig_damped_fgn"]))
            regime_min_dg.append(float(q["min_eig_damped_ggn"]))

            maxima["softmax_reconstruction_error"] = max(
                maxima["softmax_reconstruction_error"], float(q["softmax_reconstruction_error"])
            )
            maxima["decomposition_error"] = max(maxima["decomposition_error"], float(q["decomposition_error"]))
            maxima["gradient_identity_error"] = max(
                maxima["gradient_identity_error"], float(q["gradient_identity_error"])
            )
            maxima["max_full_model_optimality_gap"] = max(
                maxima["max_full_model_optimality_gap"], float(q["full_model_optimality_gap"])
            )
            maxima["min_eig_H_fgn"] = min(maxima["min_eig_H_fgn"], float(q["min_eig_H_fgn"]))
            maxima["min_eig_H_ggn"] = min(maxima["min_eig_H_ggn"], float(q["min_eig_H_ggn"]))
            maxima["min_eig_damped_fgn"] = min(
                maxima["min_eig_damped_fgn"], float(q["min_eig_damped_fgn"])
            )
            maxima["min_eig_damped_ggn"] = min(
                maxima["min_eig_damped_ggn"], float(q["min_eig_damped_ggn"])
            )

            rows.append({
                "panel": "1c",
                "study": STUDY_NAME,
                "n_classes": int(n_classes),
                "n_competitors": int(n_comp),
                "p_star": float(p_star),
                "p_dagger": float(p_dagger),
                "alpha": float(alpha),
                "alpha_min": float(alpha_min),
                "xi": float(q["xi"]),
                "xi_max": float(xi_max),
                "xi_ref": 0.0,
                "eta": float(eta),
                "beta": float(beta),
                "lambda_regime_order": int(regime_order),
                "lambda_regime_label": str(regime_label),
                "lambda_scale": float(regime_scale),
                "lambda_reference_trace_H_fgn_over_dim": float(lambda_ref_scale),
                "lambda": float(lam),
                "damping_rule": DAMPING_RULE,
                "lambda_fixed_across_xi": True,
                "output_drop_trace": float(p_dagger * q["xi"]),
                "param_drop_trace": float(q["trace_H_drop"]),
                "trace_H_fgn_ref": float(np.trace(H_fgn_ref)),
                "trace_H_fgn": float(q["trace_H_fgn"]),
                "trace_H_ggn": float(q["trace_H_ggn"]),
                "trace_H_drop": float(q["trace_H_drop"]),
                "min_eig_H_fgn": float(q["min_eig_H_fgn"]),
                "min_eig_H_ggn": float(q["min_eig_H_ggn"]),
                "min_eig_damped_fgn": float(q["min_eig_damped_fgn"]),
                "min_eig_damped_ggn": float(q["min_eig_damped_ggn"]),
                "decomposition_error": float(q["decomposition_error"]),
                "softmax_reconstruction_error": float(q["softmax_reconstruction_error"]),
                "gradient_identity_error": float(q["gradient_identity_error"]),
                "grad_norm": float(np.linalg.norm(q["g"])),
                "full_model_decrease_fgn": float(q["full_model_decrease_fgn"]),
                "full_model_decrease_ggn": float(q["full_model_decrease_ggn"]),
                "full_model_decrease_ratio": float(q["full_model_decrease_ratio"]),
                "full_model_optimality_gap": float(q["full_model_optimality_gap"]),
            })

        ratios = np.asarray(regime_ratios, dtype=np.float64)
        validation_by_regime[regime_label] = {
            "lambda_scale": float(regime_scale),
            "lambda_reference_trace_H_fgn_over_dim": float(lambda_ref_scale),
            "lambda": float(lam),
            "min_full_model_decrease_ratio": float(np.min(ratios)),
            "max_full_model_decrease_ratio": float(np.max(ratios)),
            "decrease_ratio_nonincreasing": bool(np.all(np.diff(ratios) <= 1e-10)),
            "min_full_model_decrease_fgn": float(np.min(regime_decreases_fgn)),
            "min_full_model_decrease_ggn": float(np.min(regime_decreases_ggn)),
            "min_eig_damped_fgn": float(np.min(regime_min_df)),
            "min_eig_damped_ggn": float(np.min(regime_min_dg)),
        }

    ratios_all = np.asarray([r["full_model_decrease_ratio"] for r in rows], dtype=np.float64)
    metadata = {
        "panel": "1c",
        "study": STUDY_NAME,
        "construction": (
            "2D linear-logit toy at theta=0 with exact dense full-GGN and FGN solves; "
            "both steps are evaluated under the same full-GGN damped quadratic."
        ),
        "n_classes": int(n_classes),
        "n_competitors": int(n_comp),
        "p_star": float(p_star),
        "p_dagger": float(p_dagger),
        "n_points_per_regime": int(n_points),
        "n_regimes": int(len(lambda_regimes)),
        "n_rows": int(len(rows)),
        "alpha_min": float(alpha_min),
        "alpha_max": 1.0,
        "xi_min": float(min(r["xi"] for r in rows)),
        "xi_max": float(xi_max),
        "xi_ref": 0.0,
        "eta": float(eta),
        "beta": float(beta),
        "lambda_reference_trace_H_fgn_over_dim": float(lambda_ref_scale),
        "damping_rule": DAMPING_RULE,
        "lambda_fixed_across_xi": True,
        "row_geometry": {
            "a_star": a_star.tolist(),
            "a_j": "[1, beta * t_j] for competitors",
            "t_values": t_values.tolist(),
            "rho_alpha": "(1-alpha) e_0 + alpha * uniform",
            "rho_reference": "e_0",
            "mean_competitor_row_is_constant_along_sweep": True,
        },
        "formula": {
            "q_full": "g^T d + 0.5 d^T H_G d + 0.5 lambda ||d||^2",
            "decrease": "-q_full(d)",
            "ratio": "decrease_full_model(d_FGN) / decrease_full_model(d_GGN)",
        },
        "lambda_regimes": [
            {
                "label": label,
                "lambda_scale": float(scale),
                "lambda": float(scale) * float(lambda_ref_scale),
            }
            for label, scale in lambda_regimes
        ],
        "validation": {
            **{k: float(v) for k, v in maxima.items()},
            "tolerance": float(tol),
            "min_full_model_decrease_ratio_all": float(np.min(ratios_all)),
            "max_full_model_decrease_ratio_all": float(np.max(ratios_all)),
            "decrease_ratio_in_unit_interval_for_chosen_regimes": bool(
                np.all(ratios_all >= -5e-10) and np.all(ratios_all <= 1.0 + 5e-10)
            ),
            "finite_ratios": bool(np.all(np.isfinite(ratios_all))),
            "by_regime": validation_by_regime,
        },
    }
    return rows, metadata


def main():
    args = parse_args()
    validate_args(args)

    raw_dir = artifact_root() / "raw" / args.artifact_subdir

    lambda_regimes = parse_lambda_regime_spec(args.lambda_regime_scales)
    rows, metadata = make_rows(
        n_classes=int(args.n_classes),
        p_star=float(args.p_star),
        n_points=int(args.n_points),
        alpha_min=float(args.alpha_min),
        eta=float(args.eta),
        beta=float(args.beta),
        lambda_regimes=lambda_regimes,
        tol=float(args.tol),
    )

    out_csv = raw_dir / "decrease_ratios.csv"
    out_meta = raw_dir / "metadata.json"
    write_csv_rows(out_csv, rows)
    write_json(out_meta, metadata)

    print(f"Saved {len(rows)} rows to {out_csv}")
    print(f"Saved metadata to {out_meta}")


if __name__ == "__main__":
    main()
