import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["JAX_LOG_COMPILES"] = "0"

import argparse
import csv
from importlib.metadata import version
import logging
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import somax

from common import artifact_root, append_csv_row, append_csv_rows, get_hardware_info, is_finite_tree, str2bool
from configs import DEFAULTS_1A, METHOD_IMPL_1A, METHOD_LABELS_1A, METHOD_ORDER_1A


def parse_args():
    p = argparse.ArgumentParser(
        description="Figure 1a generator: frozen-feature linear-head timing microbenchmark."
    )

    p.add_argument("--method-id", type=str, required=True, choices=METHOD_ORDER_1A)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--n-classes", type=int, required=True)

    p.add_argument("--feat-dim", type=int, default=DEFAULTS_1A["feat_dim"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS_1A["batch_size"])
    p.add_argument("--n-samples", type=int, default=DEFAULTS_1A["n_samples"])
    p.add_argument("--data-seed-base", type=int, default=DEFAULTS_1A["data_seed_base"])

    p.add_argument("--warmup-steps", type=int, default=DEFAULTS_1A["warmup_steps"])
    p.add_argument("--timed-steps", type=int, default=DEFAULTS_1A["timed_steps"])
    p.add_argument("--measure-every", type=int, default=DEFAULTS_1A["measure_every"])

    p.add_argument("--learning-rate", type=float, required=True)
    p.add_argument("--weight-decay", type=float, default=0.0)

    p.add_argument("--lam0", type=float, default=1.0)
    p.add_argument("--cg-maxiter", type=int, default=5)
    p.add_argument("--cg-tol", type=float, default=1e-5)
    p.add_argument("--cg-stabilise-every", type=int, default=0)
    p.add_argument("--cg-warm-start", type=str2bool, default=False, nargs="?", const=True)
    p.add_argument("--cg-lifted", type=str2bool, default=False, nargs="?", const=True)

    p.add_argument("--device", type=str, default=DEFAULTS_1A["device"], choices=["gpu", "cpu"])
    p.add_argument(
        "--stop-on-instability",
        type=str2bool,
        default=DEFAULTS_1A["stop_on_instability"],
        nargs="?",
        const=True,
    )
    p.add_argument("--artifact-subdir", type=str, default="1a")

    args = p.parse_args()

    if args.measure_every <= 0:
        raise ValueError("--measure-every must be positive")
    if args.warmup_steps < args.measure_every:
        raise ValueError("--warmup-steps must include at least one full timing chunk to exclude JIT compilation")
    if args.timed_steps <= 0:
        raise ValueError("--timed-steps must be positive")
    if args.warmup_steps % args.measure_every != 0:
        raise ValueError("--warmup-steps must be divisible by --measure-every")
    if args.timed_steps % args.measure_every != 0:
        raise ValueError("--timed-steps must be divisible by --measure-every")
    if args.n_classes < 2 or args.feat_dim <= 0:
        raise ValueError("--n-classes must be at least 2 and --feat-dim must be positive")
    if args.method_id == "adam_ref" and args.weight_decay != 0.0:
        raise ValueError("The Adam reference does not use weight decay")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.n_samples < args.batch_size:
        raise ValueError("--n-samples must be at least --batch-size")

    return args


def init_linear_head_params(key, feat_dim: int, n_classes: int):
    k_w, _ = jax.random.split(key)
    w = jax.random.normal(k_w, (feat_dim, n_classes), dtype=jnp.float32) / jnp.sqrt(float(feat_dim))
    b = jnp.zeros((n_classes,), dtype=jnp.float32)
    return {"W": w, "b": b}


def predict_linear_head(params, x):
    return x @ params["W"] + params["b"]


def make_frozen_feature_batches(
        *,
        n_samples: int,
        batch_size: int,
        feat_dim: int,
        n_classes: int,
        data_seed: int,
):
    usable = (int(n_samples) // int(batch_size)) * int(batch_size)
    if usable <= 0:
        raise ValueError("batch size must be <= number of samples")

    key = jax.random.PRNGKey(int(data_seed))
    kx, ky = jax.random.split(key)

    x_all = jax.random.normal(kx, (usable, feat_dim), dtype=jnp.float32)
    y_all = jax.random.randint(ky, (usable,), 0, n_classes, dtype=jnp.int32)

    n_batches = usable // int(batch_size)
    x_batches = x_all.reshape((n_batches, int(batch_size), int(feat_dim)))
    y_batches = y_all.reshape((n_batches, int(batch_size)))
    return x_batches, y_batches, usable, n_batches


def make_somax_stepper(method_name: str, params0, args):
    wd = float(args.weight_decay) if float(args.weight_decay) > 0.0 else None

    common_kwargs = dict(
        predict_fn=predict_linear_head,
        x_key="x",
        y_key="y",
        learning_rate=float(args.learning_rate),
        weight_decay=wd,
        lam_policy="const",
        lam0=float(args.lam0),
        tol=float(args.cg_tol),
        maxiter=int(args.cg_maxiter),
        warm_start=bool(args.cg_warm_start),
        stabilise_every=int(args.cg_stabilise_every),
        record_loss_every_k=-1,
        record_gain_ratio_every_k=-1,
        record_lam=False,
        record_cg_stats=False,
    )

    if method_name == "fgn_ce":
        if args.cg_warm_start:
            raise ValueError("FGN row-space warm starts are unsupported.")

        common_kwargs.update(
            lifted=bool(args.cg_lifted),
            warm_start=False,
        )

    elif method_name == "sgn_ce":
        if args.cg_lifted:
            raise ValueError("Lifted CG is only available for row-based solvers.")

        common_kwargs.update(
            warm_start=bool(args.cg_warm_start),
        )

    method = somax.make(method_name, **common_kwargs)
    state0 = method.init(params0)

    def step_once(params, state, x, y, rng):
        batch = {"x": x, "y": y}
        params_new, state_new, info = method.step(params=params, batch=batch, state=state, rng=rng)
        del info
        return params_new, state_new

    return state0, step_once, True


def build_stepper(method_id: str, params0, args):
    def ce_loss(params, x, y):
        logits = predict_linear_head(params, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    if method_id == "adam_ref":
        tx = optax.adam(
            learning_rate=float(args.learning_rate),
        )
        state0 = tx.init(params0)

        def step_once(params, state, x, y, rng):
            del rng
            loss, grads = jax.value_and_grad(ce_loss)(params, x, y)
            updates, state_new = tx.update(grads, state, params)
            params_new = optax.apply_updates(params, updates)
            del loss
            return params_new, state_new

        return state0, step_once, METHOD_IMPL_1A[method_id], False

    method_name = METHOD_IMPL_1A[method_id]
    state0, step_once, uses_rng = make_somax_stepper(method_name, params0, args)
    return state0, step_once, method_name, uses_rng


def make_advance_chunk(step_once, uses_rng: bool):
    @partial(jax.jit, static_argnames=("chunk_len",))
    def advance_chunk(params, state, rng, x_batches, y_batches, start_idx, chunk_len):
        num_batches = x_batches.shape[0]

        def body_fn(i, carry):
            params_i, state_i, rng_i = carry
            batch_idx = (start_idx + i) % num_batches
            xb = x_batches[batch_idx]
            yb = y_batches[batch_idx]
            if uses_rng:
                rng_i, sub = jax.random.split(rng_i)
            else:
                sub = jax.random.PRNGKey(0)
            params_i, state_i = step_once(params_i, state_i, xb, yb, sub)
            return params_i, state_i, rng_i

        params_out, state_out, rng_out = jax.lax.fori_loop(
            0,
            chunk_len,
            body_fn,
            (params, state, rng),
        )
        return params_out, state_out, rng_out, start_idx + chunk_len

    return advance_chunk


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    jax.config.update("jax_platform_name", args.device)
    devices = jax.devices(args.device)
    logging.info("JAX device: %s", devices[0])

    raw_dir = artifact_root() / "raw" / args.artifact_subdir
    runs_csv = raw_dir / "runs.csv"
    traces_csv = raw_dir / "traces_timing.csv"
    if runs_csv.exists():
        with runs_csv.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if (row["method_id"], int(row["n_classes"]), int(row["seed"])) == (
                        args.method_id, args.n_classes, args.seed
                ):
                    raise ValueError(
                        "This method/class/seed already exists; use a new artifacts directory or runner --overwrite")

    computer_name, cpu_info, ram_info, gpu_info = get_hardware_info()
    method_label = METHOD_LABELS_1A[args.method_id]

    data_seed = int(args.data_seed_base + args.n_classes)

    x_batches, y_batches, n_samples_used, n_batches = make_frozen_feature_batches(
        n_samples=int(args.n_samples),
        batch_size=int(args.batch_size),
        feat_dim=int(args.feat_dim),
        n_classes=int(args.n_classes),
        data_seed=data_seed,
    )

    rng = jax.random.PRNGKey(int(args.seed))
    rng, init_key = jax.random.split(rng)
    params = init_linear_head_params(init_key, feat_dim=int(args.feat_dim), n_classes=int(args.n_classes))
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    state, step_once, impl_name, uses_rng = build_stepper(args.method_id, params, args)

    advance_chunk = make_advance_chunk(step_once, uses_rng=uses_rng)

    run_name = (
        f"figure1_1a__{args.method_id}__impl_{impl_name}"
        f"__c{args.n_classes}__d{args.feat_dim}__b{args.batch_size}"
        f"__s{args.seed}__{int(time.time())}"
    )

    logging.info(
        "Starting 1a | method=%s | impl=%s | C=%d | D=%d | B=%d | seed=%d",
        args.method_id,
        impl_name,
        args.n_classes,
        args.feat_dim,
        args.batch_size,
        args.seed,
    )

    start_idx = jnp.asarray(0, dtype=jnp.int32)

    warmup_chunks = int(args.warmup_steps) // int(args.measure_every)
    for _ in range(warmup_chunks):
        params, state, rng, start_idx = advance_chunk(
            params=params,
            state=state,
            rng=rng,
            x_batches=x_batches,
            y_batches=y_batches,
            start_idx=start_idx,
            chunk_len=int(args.measure_every),
        )
    jax.block_until_ready((params, state, rng, start_idx))

    timing_rows = []
    step_times_ms = []
    unstable = False
    unstable_reason = ""
    chunks_completed = 0

    num_chunks = int(args.timed_steps) // int(args.measure_every)
    for chunk_idx in range(num_chunks):
        t0 = time.perf_counter()
        params, state, rng, start_idx = advance_chunk(
            params=params,
            state=state,
            rng=rng,
            x_batches=x_batches,
            y_batches=y_batches,
            start_idx=start_idx,
            chunk_len=int(args.measure_every),
        )
        jax.block_until_ready((params, state, rng, start_idx))
        dt = time.perf_counter() - t0

        mean_step_ms = 1e3 * dt / int(args.measure_every)
        step_times_ms.append(mean_step_ms)
        chunks_completed += 1

        timing_rows.append({
            "run_name": run_name,
            "panel": "1a",
            "method_id": args.method_id,
            "method_label": method_label,
            "impl_name": impl_name,
            "seed": int(args.seed),
            "n_classes": int(args.n_classes),
            "feat_dim": int(args.feat_dim),
            "batch_size": int(args.batch_size),
            "chunk_idx": int(chunk_idx),
            "steps_in_chunk": int(args.measure_every),
            "end_step": int(args.warmup_steps + (chunk_idx + 1) * args.measure_every),
            "step_time_ms": float(mean_step_ms),
        })

        logging.info(
            "1a | method=%s | C=%d | seed=%d | chunk=%d/%d | step_ms=%.3f",
            args.method_id,
            args.n_classes,
            args.seed,
            chunk_idx + 1,
            num_chunks,
            mean_step_ms,
        )

        if not is_finite_tree(params):
            unstable = True
            unstable_reason = "nonfinite_params"
            if args.stop_on_instability:
                logging.warning(
                    "Stopping early due to instability | method=%s | C=%d | seed=%d",
                    args.method_id,
                    args.n_classes,
                    args.seed,
                )
                break

    step_times_ms = np.asarray(step_times_ms, dtype=np.float64)

    summary_row = {
        "run_name": run_name,
        "panel": "1a",
        "study": "figure1_panel1a_frozen_linear_head_timing",
        "method_id": args.method_id,
        "method_label": method_label,
        "impl_name": impl_name,
        "seed": int(args.seed),
        "n_classes": int(args.n_classes),
        "feat_dim": int(args.feat_dim),
        "batch_size": int(args.batch_size),
        "n_samples": int(args.n_samples),
        "n_samples_used": int(n_samples_used),
        "n_batches": int(n_batches),
        "n_params": int(n_params),
        "data_seed": int(data_seed),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "lam0": float(args.lam0),
        "cg_maxiter": int(args.cg_maxiter),
        "cg_tol": float(args.cg_tol),
        "cg_stabilise_every": int(args.cg_stabilise_every),
        "cg_warm_start": bool(args.cg_warm_start),
        "cg_lifted": bool(args.cg_lifted),
        "warmup_steps": int(args.warmup_steps),
        "timed_steps": int(args.timed_steps),
        "measure_every": int(args.measure_every),
        "chunks_completed": int(chunks_completed),
        "step_time_mean_ms": float(np.mean(step_times_ms)) if len(step_times_ms) else np.nan,
        "step_time_median_ms": float(np.median(step_times_ms)) if len(step_times_ms) else np.nan,
        "step_time_std_ms": float(np.std(step_times_ms, ddof=0)) if len(step_times_ms) else np.nan,
        "is_unstable": bool(unstable),
        "unstable_reason": unstable_reason,
        "device": args.device,
        "somax_version": version("python-somax"),
        "jax_version": jax.__version__,
        "jaxlib_version": version("jaxlib"),
        "hardware_name": computer_name,
        "hardware_cpu": cpu_info,
        "hardware_ram": ram_info,
        "hardware_gpu": gpu_info,
    }

    append_csv_row(runs_csv, summary_row)
    append_csv_rows(traces_csv, timing_rows)

    logging.info("Saved summary row to %s", runs_csv)
    logging.info("Saved %d timing rows to %s", len(timing_rows), traces_csv)


if __name__ == "__main__":
    main()
