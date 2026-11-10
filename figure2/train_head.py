import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_COMPILES"] = "0"

import argparse
import csv
import logging
import math
import time
from pathlib import Path
from typing import Any, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import somax
from optax import softmax_cross_entropy_with_integer_labels

from common import (
    artifact_root,
    append_csv_row,
    append_csv_rows,
    get_hardware_info,
    is_finite_tree,
    read_json_string,
    str2bool,
    write_json,
)
from configs import DEFAULTS_2, METHOD_IMPL_2, METHOD_LABELS_2, METHOD_ORDER_2


# CLI


def parse_args():
    p = argparse.ArgumentParser(
        description="Cars196 frozen-feature linear-head optimization benchmark."
    )

    # Data/cache.
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--dataset-id", type=str, default=None)
    p.add_argument("--encoder-name", type=str, default=None)

    # Method.
    p.add_argument("--method-id", type=str, required=True, choices=METHOD_ORDER_2)
    p.add_argument("--seed", type=int, required=True)

    # Experiment.
    p.add_argument("--batch-size", type=int, default=DEFAULTS_2["batch_size"])
    p.add_argument("--n-epochs", type=int, default=DEFAULTS_2["n_epochs"])
    p.add_argument("--eval-every", type=int, default=DEFAULTS_2["eval_every"])
    p.add_argument("--log-every", type=int, default=DEFAULTS_2["log_every"])
    p.add_argument("--target-val-loss", type=float, default=DEFAULTS_2["target_val_loss"])
    p.add_argument("--init-scale", type=float, default=DEFAULTS_2["init_scale"])
    p.add_argument(
        "--stop-on-instability",
        type=str2bool,
        default=DEFAULTS_2["stop_on_instability"],
        nargs="?",
        const=True,
    )

    # First-order / post-direction transform.
    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.0)

    # Somax damping.
    p.add_argument("--lam0", type=float, default=1.0)

    # Somax CG.
    p.add_argument("--cg-maxiter", type=int, default=5)
    p.add_argument("--cg-tol", type=float, default=1e-5)
    p.add_argument("--cg-stabilise-every", type=int, default=0)
    p.add_argument("--cg-warm-start", type=str2bool, default=False, nargs="?", const=True)

    p.add_argument("--gram-mode", type=str2bool, default=False, nargs="?", const=True)

    # Hardware/output.
    p.add_argument("--device", type=str, default=DEFAULTS_2["device"], choices=["gpu", "cpu"])

    args = p.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.method_id == "fgn" and args.cg_warm_start:
        raise ValueError("fgn_ce uses row-space CG; set --cg-warm-start false")
    if args.method_id in {"sgn", "fgn"} and int(args.cg_stabilise_every) != 0:
        raise ValueError("The Cars196 benchmark uses --cg-stabilise-every 0 for SGN/FGN")

    if args.gram_mode and args.method_id != "fgn":
        raise ValueError("--gram-mode is only valid for FGN")

    return args


# Feature cache / data


def load_feature_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"missing feature cache: {path}")

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        required = {"x_train", "y_train"}
        if not required.issubset(keys):
            raise ValueError(f"feature cache {path} missing required arrays: {sorted(required - keys)}")

        metadata = read_json_string(data["metadata_json"]) if "metadata_json" in keys else {}

        if "x_val" in keys and "y_val" in keys:
            x_eval = data["x_val"]
            y_eval = data["y_val"]
            split_names = metadata.get("split_names", {}) if isinstance(metadata, dict) else {}
            eval_split_name = metadata.get("eval_split_name", split_names.get("val", "val"))
        elif "x_test" in keys and "y_test" in keys:
            x_eval = data["x_test"]
            y_eval = data["y_test"]
            eval_split_name = metadata.get("eval_split_name", "test")
        else:
            raise ValueError("feature cache must contain x_val/y_val or x_test/y_test")

        x_train = data["x_train"]
        y_train = data["y_train"]

    x_train = np.asarray(x_train, dtype=np.float32)
    x_eval = np.asarray(x_eval, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).reshape(-1)
    y_eval = np.asarray(y_eval, dtype=np.int32).reshape(-1)

    if x_train.ndim != 2 or x_eval.ndim != 2:
        raise ValueError("x_train and x_val/x_test must have shape [n, d_feat]")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train and y_train have incompatible first dimensions")
    if x_eval.shape[0] != y_eval.shape[0]:
        raise ValueError("x_val/x_test and y_val/y_test have incompatible first dimensions")
    if x_train.shape[1] != x_eval.shape[1]:
        raise ValueError("train and eval feature dimensions differ")

    if len(y_train) == 0 or len(y_eval) == 0:
        raise ValueError("train and test feature arrays must be nonempty")
    if np.min(y_train) < 0 or np.min(y_eval) < 0:
        raise ValueError("class labels must be nonnegative")
    if not np.isfinite(x_train).all() or not np.isfinite(x_eval).all():
        raise ValueError("feature arrays must be finite")
    inferred_classes = int(max(np.max(y_train), np.max(y_eval)) + 1)
    n_classes = int(metadata.get("n_classes", inferred_classes))
    if n_classes < 2:
        raise ValueError("at least two classes are required")
    if inferred_classes > n_classes:
        raise ValueError(f"metadata n_classes={n_classes} is smaller than labels imply {inferred_classes}")

    metadata = dict(metadata)
    metadata.setdefault("feature_dim", int(x_train.shape[1]))
    metadata.setdefault("n_classes", int(n_classes))
    metadata.setdefault("n_train", int(x_train.shape[0]))
    metadata.setdefault("n_eval", int(x_eval.shape[0]))
    metadata["eval_split_name"] = str(eval_split_name)

    return x_train, y_train, x_eval, y_eval, metadata


def shuffle_train_once(x_train: np.ndarray, y_train: np.ndarray, seed: int):
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(int(x_train.shape[0]))
    return x_train[perm], y_train[perm]


def make_train_batches(x_train: np.ndarray, y_train: np.ndarray, batch_size: int):
    n_train = int(x_train.shape[0])
    b = int(batch_size)

    if b <= 0:
        raise ValueError("batch size must be positive")
    if n_train <= 0:
        raise ValueError("training set must be nonempty")

    n_batches = int(np.ceil(n_train / b))
    n_effective = n_batches * b
    n_padded = n_effective - n_train

    if n_padded > 0:
        padding = np.arange(n_padded) % n_train
        x_train = np.concatenate([x_train, x_train[padding]], axis=0)
        y_train = np.concatenate([y_train, y_train[padding]], axis=0)

    feat_dim = int(x_train.shape[1])

    x_batches = jnp.asarray(
        x_train.reshape((n_batches, b, feat_dim)),
        dtype=jnp.float32,
    )
    y_batches = jnp.asarray(
        y_train.reshape((n_batches, b)),
        dtype=jnp.int32,
    )

    return x_batches, y_batches, int(n_train), int(n_batches), int(n_padded)


# Model / metrics


def init_linear_head_params(key, feat_dim: int, n_classes: int, init_scale: float):
    k_w, _ = jax.random.split(key)
    scale = float(init_scale) / math.sqrt(float(feat_dim))
    w = scale * jax.random.normal(k_w, (int(feat_dim), int(n_classes)), dtype=jnp.float32)
    b = jnp.zeros((int(n_classes),), dtype=jnp.float32)
    return {"W": w, "b": b}


def predict_linear_head(params, x):
    return x @ params["W"] + params["b"]


def count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def make_lr_schedule(base_lr: float):
    return lambda step: jnp.asarray(float(base_lr), dtype=jnp.float32)


@jax.jit
def ce_loss(params, xs, ys):
    logits = predict_linear_head(params, xs)
    return jnp.mean(softmax_cross_entropy_with_integer_labels(logits, ys))


@jax.jit
def eval_full(params, xs, ys):
    logits = predict_linear_head(params, xs)
    loss = jnp.mean(softmax_cross_entropy_with_integer_labels(logits, ys))
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == ys)
    return loss, acc


@jax.jit
def delta_stats(params_before, params_after):
    dw = params_after["W"] - params_before["W"]
    db = params_after["b"] - params_before["b"]
    sq = jnp.sum(dw * dw) + jnp.sum(db * db)
    n = params_after["W"].size + params_after["b"].size
    norm = jnp.sqrt(sq)
    rms = jnp.sqrt(sq / jnp.asarray(n, dtype=sq.dtype))
    return norm, rms


# Somax construction / telemetry


def method_label(args) -> str:
    return METHOD_LABELS_2.get(args.method_id, args.method_id)


def method_impl(args) -> str:
    return METHOD_IMPL_2[args.method_id]


def somax_cadence_hit(global_step: int, every_k: int) -> bool:
    return every_k > 0 and ((int(global_step) - 1) % int(every_k) == 0)


def make_somax_method(args, learning_rate):
    impl = method_impl(args)
    kwargs = {
        "predict_fn": predict_linear_head,
        "x_key": "x",
        "y_key": "y",
        "learning_rate": learning_rate,
        "weight_decay": (float(args.weight_decay) if float(args.weight_decay) > 0.0 else None),
        "lam_policy": "const",
        "lam0": float(args.lam0),
        "tol": float(args.cg_tol),
        "maxiter": int(args.cg_maxiter),
        "warm_start": bool(args.cg_warm_start),
        "stabilise_every": int(args.cg_stabilise_every),
        "record_loss_every_k": int(args.log_every),
        "record_gain_ratio_every_k": -1,
        "record_lam": True,
        "record_cg_stats": True,
    }

    if args.method_id == "fgn":
        kwargs["gram_mode"] = "linear_head" if args.gram_mode else "none"
        kwargs["linear_head_has_bias"] = True
        kwargs["lifted"] = False

    return somax.make(impl, **kwargs)


FLOAT_INFO_KEYS = (
    "loss_before",
    "loss_after",
    "lam_used",
    "lam_next",
    "lam_factor",
    "gain_ratio",
    "pred_dec",
    "act_dec",
    "cg_resid",
    "delta_rms",
)

INT_INFO_KEYS = (
    "step",
    "tr_action",
    "cg_iters",
    "cg_maxiter",
)

BOOL_INFO_KEYS = (
    "gain_ratio_valid",
    "step_accepted",
    "cg_converged",
    "lam_clipped",
)


def read_float(info: dict[str, Any], key: str) -> Optional[float]:
    value = info.get(key)
    if value is None:
        return None
    x = float(np.asarray(value))
    if not math.isfinite(x):
        return None
    return x


def read_int(info: dict[str, Any], key: str) -> Optional[int]:
    value = info.get(key)
    if value is None:
        return None
    x = int(np.asarray(value))
    if x < 0:
        return None
    return x


def read_bool(info: dict[str, Any], key: str) -> Optional[bool]:
    value = info.get(key)
    if value is None:
        return None
    return bool(np.asarray(value))


def extract_ctrl_info(info: Optional[dict[str, Any]]) -> dict[str, Any]:
    base = {} if info is None else dict(info)
    out = {}
    for key in FLOAT_INFO_KEYS:
        out[f"info_{key}"] = read_float(base, key)
    for key in INT_INFO_KEYS:
        out[f"info_{key}"] = read_int(base, key)
    for key in BOOL_INFO_KEYS:
        out[f"info_{key}"] = read_bool(base, key)
    out["info_rho"] = out.get("info_gain_ratio")
    return out


def empty_ctrl_info() -> dict[str, Any]:
    out = {}
    for key in FLOAT_INFO_KEYS:
        out[f"info_{key}"] = None
    for key in INT_INFO_KEYS:
        out[f"info_{key}"] = None
    for key in BOOL_INFO_KEYS:
        out[f"info_{key}"] = None
    out["info_rho"] = None
    return out


def should_emit_somax_ctrl_row(global_step: int, args) -> bool:
    if args.method_id == "adam":
        return False
    return somax_cadence_hit(global_step, int(args.log_every))


# Main


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.device == "cpu":
        jax.config.update("jax_platform_name", "cpu")
    elif jax.default_backend() != "gpu":
        raise RuntimeError("GPU requested, but JAX is not using a GPU backend")

    seed = int(args.seed)
    np.random.seed(seed)

    raw_dir = artifact_root() / "figure2" / "raw"
    runs_csv = raw_dir / "runs.csv"
    if runs_csv.exists():
        with runs_csv.open(newline="") as handle:
            if any(row["method_id"] == args.method_id and int(row["seed"]) == seed for row in csv.DictReader(handle)):
                raise FileExistsError(f"Method {args.method_id}, seed {seed} already exists in {runs_csv}")
    traces_eval_csv = raw_dir / "traces_eval.csv"
    traces_ctrl_csv = raw_dir / "traces_ctrl.csv"

    # Data.
    x_train_np, y_train_np, x_eval_np, y_eval_np, metadata = load_feature_cache(args.feature_cache)
    feat_dim = int(x_train_np.shape[1])
    n_classes = int(metadata["n_classes"])
    dataset_id = args.dataset_id or str(metadata.get("dataset_id", "unknown_dataset"))
    encoder_name = args.encoder_name or str(metadata.get("encoder_name", "unknown_encoder"))
    eval_split_name = str(metadata.get("eval_split_name", "val"))

    # Always shuffle once before static batching. Cars196/HF caches are class-sorted.
    x_train_np, y_train_np = shuffle_train_once(x_train_np, y_train_np, seed=seed)

    x_batches, y_batches, n_train, n_batches, n_padded = make_train_batches(
        x_train_np,
        y_train_np,
        batch_size=int(args.batch_size),
    )

    x_eval = jnp.asarray(x_eval_np, dtype=jnp.float32)
    y_eval = jnp.asarray(y_eval_np, dtype=jnp.int32)
    n_eval = int(y_eval_np.shape[0])

    total_steps = int(args.n_epochs) * int(n_batches)
    if total_steps <= 0:
        raise ValueError("total training steps must be positive")

    # Params.
    rng = jax.random.PRNGKey(seed)
    rng, init_key, compile_key = jax.random.split(rng, 3)
    params0 = init_linear_head_params(
        init_key,
        feat_dim=feat_dim,
        n_classes=n_classes,
        init_scale=float(args.init_scale),
    )
    n_params = count_params(params0)

    lr_fn = make_lr_schedule(float(args.learning_rate))
    impl = method_impl(args)
    label = method_label(args)
    is_fom = args.method_id == "adam"

    # Optimizer.
    if is_fom:
        if float(args.weight_decay) != 0.0:
            raise ValueError("The Cars196 Adam baseline uses --weight-decay 0.0")
        tx = optax.adam(learning_rate=lr_fn)
        opt_state0 = tx.init(params0)
        som_state0 = None

        @jax.jit
        def train_step_optax(params, opt_state, xs, ys):
            loss, grads = jax.value_and_grad(ce_loss)(params, xs, ys)
            updates, opt_state_new = tx.update(grads, opt_state, params)
            params_new = optax.apply_updates(params, updates)
            return params_new, opt_state_new, loss

    else:
        som = make_somax_method(args, lr_fn)
        som_state0 = som.init(params0)
        opt_state0 = None

        @jax.jit
        def train_step_somax(params, som_state, xs, ys, rng_key):
            batch = {"x": xs, "y": ys}
            params_new, som_state_new, info = som.step(
                params=params,
                batch=batch,
                state=som_state,
                rng=rng_key,
            )
            return params_new, som_state_new, info

    computer_name, cpu_info, ram_info, gpu_info = get_hardware_info()
    run_name = (
        f"figure2__{dataset_id}__{encoder_name}__{args.method_id}__impl_{impl}"
        f"__lr{float(args.learning_rate):g}__lam{float(args.lam0):g}"
        f"__b{int(args.batch_size)}__s{seed}__{int(time.time())}"
    )

    logging.info(
        "Starting Cars196 head benchmark | dataset=%s | encoder=%s | method=%s | impl=%s | D=%d | C=%d | B=%d | seed=%d",
        dataset_id,
        encoder_name,
        args.method_id,
        impl,
        feat_dim,
        n_classes,
        int(args.batch_size),
        seed,
    )

    # Compile warmup: compile step/eval on copies, then reset to identical
    # initial params/state before the timed optimization trace.
    warm_x = x_batches[0]
    warm_y = y_batches[0]
    compile_t0 = time.perf_counter()
    if is_fom:
        p_warm, o_warm, loss_warm = train_step_optax(params0, opt_state0, warm_x, warm_y)
        eval_warm = eval_full(params0, x_eval, y_eval)
        jax.block_until_ready(p_warm)
        jax.block_until_ready(o_warm)
        jax.block_until_ready(loss_warm)
        jax.block_until_ready(eval_warm)
    else:
        p_warm, s_warm, info_warm = train_step_somax(params0, som_state0, warm_x, warm_y, compile_key)
        eval_warm = eval_full(params0, x_eval, y_eval)
        jax.block_until_ready(p_warm)
        jax.block_until_ready(s_warm)
        jax.block_until_ready(info_warm)
        jax.block_until_ready(eval_warm)
    # Compile the loop RNG operation before timing; the warmed values are discarded.
    jax.block_until_ready(jax.random.split(rng))
    compile_time_s = time.perf_counter() - compile_t0

    params = params0
    opt_state = opt_state0
    som_state = som_state0

    # Initial eval after compilation, not counted in optimization time.
    eval0_loss, eval0_acc = eval_full(params, x_eval, y_eval)
    eval0_loss = float(jax.device_get(eval0_loss))
    eval0_acc = float(jax.device_get(eval0_acc))

    target_configured = math.isfinite(float(args.target_val_loss))
    reached_target = target_configured and (eval0_loss <= float(args.target_val_loss))
    time_to_target_s = 0.0 if reached_target else np.nan

    eval_rows = [
        {
            "run_name": run_name,
            "panel": "2",
            "dataset_id": dataset_id,
            "encoder_name": encoder_name,
            "eval_split_name": eval_split_name,
            "method_id": args.method_id,
            "method_label": label,
            "impl_name": impl,
            "seed": seed,
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "damping_policy": "const",
            "lam0": float(args.lam0),
            "step": 0,
            "wall_clock_s": 0.0,
            "elapsed_wall_clock_s": 0.0,
            "eval_time_accum_s": 0.0,
            "val_loss": eval0_loss,
            "val_acc": eval0_acc,
            "train_loss_last": np.nan,
        }
    ]
    ctrl_rows = []
    best_val_loss = eval0_loss
    best_val_acc = eval0_acc
    last_val_loss = eval0_loss
    last_val_acc = eval0_acc
    last_train_loss = np.nan
    last_ctrl_info = empty_ctrl_info()

    unstable = False
    unstable_reason = ""
    step_times_ms = []
    step_times_ms_steady = []
    steps_completed = 0
    last_eval_step = 0

    # Timers. wall_clock_s in eval traces is synchronized training time only:
    # no feature extraction, no compilation, no validation pass.
    elapsed_start = time.perf_counter()
    segment_start = elapsed_start
    segment_steps = 0
    train_time_accum_s = 0.0
    eval_time_accum_s = 0.0

    def close_train_segment() -> tuple[float, int]:
        nonlocal segment_start, segment_steps, train_time_accum_s
        now = time.perf_counter()
        seg_s = now - segment_start
        seg_steps = int(segment_steps)
        if seg_steps > 0:
            train_time_accum_s += seg_s
        segment_start = time.perf_counter()
        segment_steps = 0
        return seg_s, seg_steps

    for step in range(total_steps):
        batch_idx = step % n_batches
        batch_x = x_batches[batch_idx]
        batch_y = y_batches[batch_idx]
        params_before = params

        if is_fom:
            params, opt_state, train_val = train_step_optax(params, opt_state, batch_x, batch_y)
            info = None
        else:
            rng, sub = jax.random.split(rng)
            params, som_state, info = train_step_somax(params, som_state, batch_x, batch_y, sub)
            train_val = None

        global_step = step + 1
        steps_completed = global_step
        segment_steps += 1

        need_speed_log = global_step % int(args.log_every) == 0
        need_eval = global_step % int(args.eval_every) == 0
        expect_ctrl = should_emit_somax_ctrl_row(global_step, args)
        boundary = need_speed_log or need_eval or expect_ctrl

        if boundary:
            if is_fom:
                jax.block_until_ready((params, opt_state, train_val))
            else:
                jax.block_until_ready((params, som_state, info))
            seg_s, seg_steps = close_train_segment()
        else:
            seg_s, seg_steps = 0.0, 0

        step_norm = None
        step_rms = None
        if boundary:
            dn, dr = delta_stats(params_before, params)
            dn, dr = jax.device_get((dn, dr))
            step_norm = float(dn)
            step_rms = float(dr)

        ctrl_row = None

        if is_fom and need_speed_log:
            current_loss = float(jax.device_get(train_val))
            last_train_loss = current_loss
            if not math.isfinite(current_loss):
                unstable = True
                unstable_reason = "nonfinite_train_loss"

        if (not is_fom) and expect_ctrl:
            info_host = dict(jax.device_get(info))
            ctrl_info = extract_ctrl_info(info_host)
            last_ctrl_info = ctrl_info
            loss_before = ctrl_info["info_loss_before"]
            if loss_before is not None:
                last_train_loss = loss_before
            ctrl_row = {
                "run_name": run_name,
                "panel": "2",
                "dataset_id": dataset_id,
                "encoder_name": encoder_name,
                "method_id": args.method_id,
                "method_label": label,
                "impl_name": impl,
                "seed": seed,
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "damping_policy": "const",
                "lam0": float(args.lam0),
                "step": global_step,
                "wall_clock_s": float(train_time_accum_s),
                "train_loss": last_train_loss,
                "val_loss_last": last_val_loss,
                "val_acc_last": last_val_acc,
                "step_norm": step_norm,
                "step_rms": step_rms,
                "step_ms": None,
                "steps_per_sec": None,
                "steps_in_window": None,
                "lr": float(jax.device_get(lr_fn(global_step))),
            }
            ctrl_row.update(ctrl_info)

        if need_speed_log:
            if seg_steps <= 0:
                mean_step_ms = np.nan
                steps_per_sec = np.nan
            else:
                mean_step_ms = 1e3 * seg_s / float(seg_steps)
                steps_per_sec = float(seg_steps) / max(seg_s, 1e-12)
            step_times_ms.append(mean_step_ms)
            if len(step_times_ms) > 1:
                step_times_ms_steady.append(mean_step_ms)

            if ctrl_row is None:
                ctrl_row = {
                    "run_name": run_name,
                    "panel": "2",
                    "dataset_id": dataset_id,
                    "encoder_name": encoder_name,
                    "method_id": args.method_id,
                    "method_label": label,
                    "impl_name": impl,
                    "seed": seed,
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.learning_rate),
                    "damping_policy": "const",
                    "lam0": float(args.lam0),
                    "step": global_step,
                    "wall_clock_s": float(train_time_accum_s),
                    "train_loss": last_train_loss,
                    "val_loss_last": last_val_loss,
                    "val_acc_last": last_val_acc,
                    "step_norm": step_norm,
                    "step_rms": step_rms,
                    "step_ms": None,
                    "steps_per_sec": None,
                    "steps_in_window": None,
                    "lr": float(jax.device_get(lr_fn(global_step))),
                }
                ctrl_row.update(empty_ctrl_info())

            ctrl_row["step_ms"] = float(mean_step_ms) if math.isfinite(mean_step_ms) else np.nan
            ctrl_row["steps_per_sec"] = float(steps_per_sec) if math.isfinite(steps_per_sec) else np.nan
            ctrl_row["steps_in_window"] = int(seg_steps)

            logging.info(
                "speed | step=%d | train_time=%.3f | step_ms=%.3f | sps=%.3f | train_loss=%s",
                global_step,
                train_time_accum_s,
                mean_step_ms,
                steps_per_sec,
                f"{last_train_loss:.6f}" if math.isfinite(last_train_loss) else "nan",
            )

            if not is_finite_tree(params):
                unstable = True
                unstable_reason = unstable_reason or "nonfinite_params"

            if args.stop_on_instability and unstable:
                if ctrl_row is not None:
                    ctrl_rows.append(ctrl_row)
                logging.warning("Stopping early due to instability at step=%d: %s", global_step, unstable_reason)
                break

        if ctrl_row is not None:
            ctrl_rows.append(ctrl_row)

        if need_eval:
            eval_t0 = time.perf_counter()
            val_loss, val_acc = eval_full(params, x_eval, y_eval)
            val_loss = float(jax.device_get(val_loss))
            val_acc = float(jax.device_get(val_acc))
            eval_time_accum_s += time.perf_counter() - eval_t0

            last_val_loss = val_loss
            last_val_acc = val_acc
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            last_eval_step = global_step

            elapsed_wall_clock_s = time.perf_counter() - elapsed_start
            eval_rows.append(
                {
                    "run_name": run_name,
                    "panel": "2",
                    "dataset_id": dataset_id,
                    "encoder_name": encoder_name,
                    "eval_split_name": eval_split_name,
                    "method_id": args.method_id,
                    "method_label": label,
                    "impl_name": impl,
                    "seed": seed,
                    "batch_size": int(args.batch_size),
                    "learning_rate": float(args.learning_rate),
                    "damping_policy": "const",
                    "lam0": float(args.lam0),
                    "step": global_step,
                    "wall_clock_s": float(train_time_accum_s),
                    "elapsed_wall_clock_s": float(elapsed_wall_clock_s),
                    "eval_time_accum_s": float(eval_time_accum_s),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "train_loss_last": last_train_loss,
                }
            )

            logging.info(
                "eval  | step=%d | train_time=%.3f | val_loss=%.6f | val_acc=%.4f",
                global_step,
                train_time_accum_s,
                val_loss,
                val_acc,
            )

            if target_configured and (not reached_target) and val_loss <= float(args.target_val_loss):
                reached_target = True
                time_to_target_s = float(train_time_accum_s)

        if boundary:
            # Exclude validation and host logging from the next
            # synchronized optimization-time segment.
            segment_start = time.perf_counter()

    # Close any pending training segment and do a final eval if cadence missed.
    if steps_completed > 0 and segment_steps > 0:
        if is_fom:
            jax.block_until_ready((params, opt_state, train_val))
        else:
            jax.block_until_ready((params, som_state, info))
        close_train_segment()

    if steps_completed > 0 and last_eval_step != steps_completed:
        eval_t0 = time.perf_counter()
        val_loss, val_acc = eval_full(params, x_eval, y_eval)
        val_loss = float(jax.device_get(val_loss))
        val_acc = float(jax.device_get(val_acc))
        eval_time_accum_s += time.perf_counter() - eval_t0
        last_val_loss = val_loss
        last_val_acc = val_acc
        best_val_loss = min(best_val_loss, val_loss)
        best_val_acc = max(best_val_acc, val_acc)
        if target_configured and (not reached_target) and val_loss <= float(args.target_val_loss):
            reached_target = True
            time_to_target_s = float(train_time_accum_s)

        eval_rows.append(
            {
                "run_name": run_name,
                "panel": "2",
                "dataset_id": dataset_id,
                "encoder_name": encoder_name,
                "eval_split_name": eval_split_name,
                "method_id": args.method_id,
                "method_label": label,
                "impl_name": impl,
                "seed": seed,
                "batch_size": int(args.batch_size),
                "learning_rate": float(args.learning_rate),
                "damping_policy": "const",
                "lam0": float(args.lam0),
                "step": steps_completed,
                "wall_clock_s": float(train_time_accum_s),
                "elapsed_wall_clock_s": float(time.perf_counter() - elapsed_start),
                "eval_time_accum_s": float(eval_time_accum_s),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "train_loss_last": last_train_loss,
            }
        )

    step_times_ms = np.asarray(step_times_ms, dtype=np.float64)
    step_times_ms_steady = np.asarray(step_times_ms_steady, dtype=np.float64)

    def collect_ctrl_series(key: str):
        return [row[key] for row in ctrl_rows if row.get(key) is not None]

    if not math.isfinite(last_val_loss) or not math.isfinite(last_val_acc):
        unstable = True
        unstable_reason = unstable_reason or "nonfinite_final_eval"

    cg_iters_series = collect_ctrl_series("info_cg_iters")
    cg_resid_series = collect_ctrl_series("info_cg_resid")
    lam_used_series = collect_ctrl_series("info_lam_used")
    gain_ratio_series = collect_ctrl_series("info_gain_ratio")
    cg_converged_series = collect_ctrl_series("info_cg_converged")

    summary_row = {
        "run_name": run_name,
        "panel": "2",
        "study": "figure2_frozen_feature_linear_head_optimization",
        "feature_cache": str(args.feature_cache),
        "dataset_id": dataset_id,
        "encoder_name": encoder_name,
        "eval_split_name": eval_split_name,
        "method_id": args.method_id,
        "method_label": label,
        "impl_name": impl,
        "seed": seed,
        "feat_dim": feat_dim,
        "n_classes": n_classes,
        "batch_size": int(args.batch_size),
        "n_train": int(n_train),
        "n_train_raw": int(x_train_np.shape[0]),
        "n_padded": int(n_padded),
        "n_effective_train_per_epoch": int(n_batches * int(args.batch_size)),
        "n_eval": int(n_eval),
        "n_batches": int(n_batches),
        "n_params": int(n_params),
        "steps_completed": int(steps_completed),
        "n_epochs": int(args.n_epochs),
        "eval_every": int(args.eval_every),
        "log_every": int(args.log_every),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "init_scale": float(args.init_scale),
        "damping_policy": "const",
        "lam0": float(args.lam0),
        "cg_maxiter": int(args.cg_maxiter),
        "cg_tol": float(args.cg_tol),
        "cg_stabilise_every": int(args.cg_stabilise_every),
        "cg_warm_start": bool(args.cg_warm_start),
        "gram_mode": bool(args.gram_mode),
        "shuffle_train": True,
        "target_val_loss": float(args.target_val_loss),
        "initial_val_loss": float(eval0_loss),
        "initial_val_acc": float(eval0_acc),
        "final_val_loss": float(last_val_loss),
        "final_val_acc": float(last_val_acc),
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "final_train_loss": float(last_train_loss) if math.isfinite(last_train_loss) else np.nan,
        "time_to_target_s": float(time_to_target_s) if math.isfinite(time_to_target_s) else np.nan,
        "train_wall_clock_s": float(train_time_accum_s),
        "elapsed_wall_clock_s": float(time.perf_counter() - elapsed_start),
        "eval_time_accum_s": float(eval_time_accum_s),
        "compile_time_s": float(compile_time_s),
        "step_time_mean_ms": float(np.mean(step_times_ms)) if len(step_times_ms) else np.nan,
        "step_time_median_ms": float(np.median(step_times_ms)) if len(step_times_ms) else np.nan,
        "step_time_std_ms": float(np.std(step_times_ms, ddof=0)) if len(step_times_ms) else np.nan,
        "step_time_mean_ms_steady": float(np.mean(step_times_ms_steady)) if len(step_times_ms_steady) else np.nan,
        "step_time_median_ms_steady": float(np.median(step_times_ms_steady)) if len(step_times_ms_steady) else np.nan,
        "step_time_std_ms_steady": float(np.std(step_times_ms_steady, ddof=0)) if len(step_times_ms_steady) else np.nan,
        "mean_cg_iters": float(np.mean(cg_iters_series)) if cg_iters_series else np.nan,
        "median_cg_iters": float(np.median(cg_iters_series)) if cg_iters_series else np.nan,
        "mean_cg_relative_residual": float(np.mean(cg_resid_series)) if cg_resid_series else np.nan,
        "median_cg_relative_residual": float(np.median(cg_resid_series)) if cg_resid_series else np.nan,
        "max_cg_relative_residual": float(np.max(cg_resid_series)) if cg_resid_series else np.nan,
        "mean_lambda": float(np.mean(lam_used_series)) if lam_used_series else np.nan,
        "median_lambda": float(np.median(lam_used_series)) if lam_used_series else np.nan,
        "mean_gain_ratio": float(np.mean(gain_ratio_series)) if gain_ratio_series else np.nan,
        "median_gain_ratio": float(np.median(gain_ratio_series)) if gain_ratio_series else np.nan,
        "cg_converged_frac": float(np.mean(cg_converged_series)) if cg_converged_series else np.nan,
        "is_unstable": bool(unstable),
        "unstable_reason": unstable_reason,
        "hardware_name": computer_name,
        "hardware_cpu": cpu_info,
        "hardware_ram": ram_info,
        "hardware_gpu": gpu_info,
        "timing_definition": (
            "train_wall_clock_s excludes feature extraction, JIT compile warmup, "
            "validation passes, host logging, CSV writes, and plotting"
        ),
    }
    for key, value in last_ctrl_info.items():
        summary_row[f"last_{key[5:]}"] = value

    append_csv_row(runs_csv, summary_row)
    append_csv_rows(traces_eval_csv, eval_rows)
    append_csv_rows(traces_ctrl_csv, ctrl_rows)

    write_json(
        raw_dir / "run_metadata" / f"{run_name}.json",
        {
            "args": vars(args),
            "feature_metadata": metadata,
            "run_name": run_name,
            "summary": summary_row,
        },
    )

    logging.info("Saved summary row to %s", runs_csv)
    logging.info("Saved %d eval rows to %s", len(eval_rows), traces_eval_csv)
    logging.info("Saved %d control rows to %s", len(ctrl_rows), traces_ctrl_csv)
    if unstable:
        raise RuntimeError(f"Training failed: {unstable_reason}; details saved to {runs_csv}")


if __name__ == "__main__":
    main()
