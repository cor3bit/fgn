import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_LOG_COMPILES", "0")

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path(os.environ.get("FGN_ARTIFACTS_DIR", ROOT / "artifacts")).resolve()

import jax
import jax.numpy as jnp
import optax
import somax
from flax import linen as nn
from flax import struct
from flax.core import FrozenDict, freeze
from flax.traverse_util import flatten_dict, unflatten_dict
from somax import metrics as sm
from somax.utils import prefetch_to_device_single, tree_all_finite

CIFAR_STATS = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
}

DEFAULT_LR = {
    "sgdm": 0.1,
    "fgn_ce": 0.3,
    "sgn_ce": 0.3,
}

CSV_FIELDS = [
    "event", "status", "step", "epoch", "examples_seen",
    "batch_loss", "train_loss", "train_accuracy", "test_loss", "test_accuracy",
    "learning_rate", "damping", "damping_next",
    "cg_iters", "cg_maxiter", "cg_residual_final", "cg_converged", "cg_hit_budget",
    "nonfinite_event",
    "step_ms", "examples_per_second", "train_time_s", "eval_duration_s", "eval_time_s",
    "wall_time_s", "compile_step_s", "compile_eval_s",
    "reason",
]


# =============================================================================
# CLI / serialization
# =============================================================================


def parse_ints(value: str) -> tuple[int, ...]:
    return tuple(int(x) for x in value.split(",") if x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        type=str,
        choices=tuple(CIFAR_STATS),
        default="cifar100",
    )
    p.add_argument(
        "--optimizer",
        type=str,
        choices=tuple(DEFAULT_LR),
        default="fgn_ce",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every-epochs", type=int, default=5)
    p.add_argument("--initial-eval", action=argparse.BooleanOptionalAction, default=True)

    # Fixed training-monitor subset used only for periodic train metrics.
    p.add_argument("--train-eval-size", type=int, default=5000)

    p.add_argument("--learning-rate", type=float)
    p.add_argument("--milestone-epochs", type=parse_ints, default=(100, 150))
    p.add_argument("--lr-decay", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)

    # Fixed damping and fixed CG cap
    p.add_argument("--damping", type=float, default=1.0)
    p.add_argument("--cg-maxiter", type=int, default=5)
    p.add_argument("--cg-tol", type=float, default=1e-5)
    p.add_argument(
        "--cg-lifted", action=argparse.BooleanOptionalAction, default=False,
        help="Use lifted row-space CG for FGN.",
    )

    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--data-dir", type=Path, default=ARTIFACTS / "data" / "tfds")
    p.add_argument("--smoke", action="store_true", help="Use tiny synthetic inputs without downloads.")
    p.add_argument("--run-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--stop-on-instability", action=argparse.BooleanOptionalAction, default=True)

    args = p.parse_args()
    if args.learning_rate is None:
        args.learning_rate = DEFAULT_LR[args.optimizer]
    return args


def validate(a: argparse.Namespace) -> None:
    if a.batch_size < 2 or a.eval_batch_size < 1:
        raise ValueError("invalid batch size")
    if a.train_eval_size < 1:
        raise ValueError("train-eval-size must be positive")
    if a.epochs < 1 or a.max_steps == 0 or a.max_steps < -1:
        raise ValueError("invalid training horizon")
    if min(a.log_every, a.eval_every_epochs) < 1:
        raise ValueError("invalid logging cadence")
    if a.learning_rate <= 0 or a.weight_decay < 0:
        raise ValueError("invalid learning rate or weight decay")
    if not 0 <= a.momentum < 1:
        raise ValueError("momentum must be in [0, 1)")
    if a.optimizer in {"fgn_ce", "sgn_ce"}:
        if a.damping <= 0 or a.cg_maxiter < 1 or a.cg_tol <= 0:
            raise ValueError("invalid damping/CG configuration")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def dump_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(jsonable(data), indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def config_dict(a: argparse.Namespace) -> dict[str, Any]:
    skip = {"run_dir", "overwrite"}
    if a.optimizer != "fgn_ce":
        skip.add("cg_lifted")
    return {k: jsonable(v) for k, v in vars(a).items() if k not in skip}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_python_tree(root: Path) -> str:
    paths = sorted(root.rglob("*.py"))
    if not paths:
        raise FileNotFoundError(f"no Python sources to fingerprint under {root}")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root)
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    versions = {}
    for name in ("python-somax", "jax", "jaxlib", "flax", "optax", "numpy",
                 "tensorflow", "tensorflow-cpu", "tensorflow-datasets"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def protocol_fingerprint() -> dict[str, Any]:
    """Fingerprint code that defines the experiment and Somax update semantics."""
    return {
        "schema_version": 2,
        "train_sha256": sha256_file(Path(__file__).resolve()),
        "somax_sha256": sha256_python_tree(Path(somax.__file__).resolve().parent),
        "package_versions": package_versions(),
        "device_kinds": [device.device_kind for device in jax.devices()],
        "jax_enable_x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def default_run_dir(a: argparse.Namespace) -> Path:
    name = f"lr{a.learning_rate:g}"
    if a.optimizer in {"fgn_ce", "sgn_ce"}:
        name += f"_lam{a.damping:g}_cg{a.cg_maxiter}"
    if a.optimizer == "fgn_ce":
        name += "_lifted" if a.cg_lifted else "_standard"
    return (
            ARTIFACTS / "figure3" / "single" / a.dataset /
            a.optimizer / f"{timestamp_id()}_{name}_seed{a.seed}"
    )


def prepare_run(a: argparse.Namespace) -> tuple[Path, bool]:
    """Create a fresh run directory, except for matching completed runs."""
    run_dir = a.run_dir or default_run_dir(a)
    config = config_dict(a)
    protocol = protocol_fingerprint()
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"
    protocol_path = run_dir / "protocol.json"

    if run_dir.exists() and any(run_dir.iterdir()) and not a.overwrite:
        if config_path.exists() and protocol_path.exists() and summary_path.exists():
            try:
                old_config = json.loads(config_path.read_text())
                old_protocol = json.loads(protocol_path.read_text())
                summary = json.loads(summary_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
            else:
                if old_config == config:
                    if summary.get("status") == "completed" and old_protocol == protocol:
                        print(f"SKIP completed: {run_dir}", flush=True)
                        return run_dir, True

                    if summary.get("status") in ("failed", "interrupted"):
                        print(f"RETRY {summary['status']}: {run_dir}", flush=True)
                        shutil.rmtree(run_dir)

        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Existing run with unknown status or changed protocol: {run_dir}. "
                "Choose another directory or pass --overwrite."
            )

    if run_dir.exists() and a.overwrite:
        shutil.rmtree(run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    dump_json(run_dir / "config.json", config)
    dump_json(run_dir / "protocol.json", protocol)
    (run_dir / "RUNNING").touch()
    with (run_dir / "metrics.csv").open("w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()
    return run_dir, False


def append_csv(path: Path, **row: Any) -> None:
    clean = {}
    for key in CSV_FIELDS:
        value = row.get(key)
        if isinstance(value, (np.floating, np.integer)):
            value = value.item()
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        clean[key] = value
    with path.open("a", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(clean)


def block(tree: Any) -> Any:
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def scalar(x: Any) -> float:
    return float(np.asarray(jax.device_get(x)))


# =============================================================================
# CIFAR pipeline
# =============================================================================


def subset_dataset(ds, indices: np.ndarray, n_total: int, tf):
    mask = np.zeros(n_total, dtype=np.bool_)
    mask[indices] = True
    mask_tf = tf.constant(mask)

    def keep(index, sample):
        del sample
        return tf.gather(mask_tf, index)

    def drop_index(index, sample):
        del index
        return sample

    return ds.enumerate().filter(keep).map(drop_index, num_parallel_calls=tf.data.AUTOTUNE)


def load_cifar(
        dataset: str,
        batch_size: int,
        eval_batch_size: int,
        seed: int,
        data_dir: Path,
        *,
        train_eval_size: int,
):
    try:
        import tensorflow as tf
        import tensorflow_datasets as tfds
    except ImportError as exc:
        raise RuntimeError("tensorflow and tensorflow-datasets are required") from exc

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as exc:
        if tf.config.get_visible_devices("GPU"):
            raise RuntimeError("TensorFlow GPU visibility must be disabled before initialization") from exc
    tf.random.set_seed(seed)

    (train_raw, test_raw), info = tfds.load(
        dataset,
        data_dir=str(data_dir),
        split=["train", "test"],
        as_supervised=True,
        with_info=True,
        shuffle_files=False,
    )

    n_train = int(info.splits["train"].num_examples)
    n_test = int(info.splits["test"].num_examples)
    n_classes = int(info.features["label"].num_classes)
    image_shape = tuple(info.features["image"].shape)
    mean, std = CIFAR_STATS[dataset]
    mean = tf.constant(mean, tf.float32)
    std = tf.constant(std, tf.float32)

    options = tf.data.Options()
    options.experimental_deterministic = True
    train_raw = train_raw.with_options(options)
    test_raw = test_raw.with_options(options)

    monitor_size = min(train_eval_size, n_train)
    monitor_rng = np.random.default_rng(20260801)
    monitor_indices = np.sort(
        monitor_rng.choice(np.arange(n_train, dtype=np.int64), monitor_size, replace=False)
    )
    train_monitor_raw = subset_dataset(train_raw, monitor_indices, n_train, tf)

    def normalize(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return (image - mean) / std, tf.cast(label, tf.int32)

    def augment_and_normalize(index, sample):
        image, label = sample
        index = tf.cast(index, tf.int32)
        seed_pair = tf.stack([tf.cast(seed, tf.int32), index])
        seeds = tf.random.experimental.stateless_split(seed_pair, 2)
        # Match the standard CIFAR recipe: pad/crop/flip in raw pixel space,
        # then normalize. Padding therefore inserts raw black pixels rather
        # than zeros in normalized coordinates.
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)
        image = tf.image.stateless_random_crop(image, [32, 32, 3], seed=seeds[0])
        image = tf.image.stateless_random_flip_left_right(image, seed=seeds[1])
        return normalize(image, label)

    # Cache compact uint8 examples. Augmentation and normalization remain
    # downstream so every repeated pass receives fresh crop/flip randomness.
    training_ds = train_raw.cache()
    training_ds = training_ds.shuffle(n_train, seed=seed, reshuffle_each_iteration=True)
    training_ds = training_ds.repeat().enumerate()
    training_ds = training_ds.map(
        augment_and_normalize,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    training_ds = training_ds.batch(batch_size, drop_remainder=True)
    training_ds = training_ds.prefetch(tf.data.AUTOTUNE)
    train_iter = prefetch_to_device_single(tfds.as_numpy(training_ds), size=2)

    def device_batches(raw_ds):
        ds = raw_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(eval_batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return tuple(
            (jax.device_put(x.astype(np.float32)), jax.device_put(y.astype(np.int32)))
            for x, y in tfds.as_numpy(ds)
        )

    eval_batches = {
        "train": device_batches(train_monitor_raw),
        "test": device_batches(test_raw),
    }
    dataset_metadata = {
        "n_train": n_train,
        "n_train_monitor": monitor_size,
        "n_test": n_test,
    }
    return (
        train_iter,
        eval_batches,
        n_train,
        n_classes,
        image_shape,
        str(info.version),
        dataset_metadata,
    )


def load_synthetic(batch_size: int, eval_batch_size: int, seed: int):
    """Tiny inputs for exercising the real ResNet/BatchNorm/optimizer path."""
    rng = np.random.default_rng(seed)
    n_train = 2 * batch_size
    image_shape = (8, 8, 3)
    n_classes = 100
    x = jax.device_put(rng.normal(size=(n_train,) + image_shape).astype(np.float32))
    y = jax.device_put(rng.integers(n_classes, size=n_train, dtype=np.int32))
    test_x = jax.device_put(rng.normal(size=(n_train,) + image_shape).astype(np.float32))
    test_y = jax.device_put(rng.integers(n_classes, size=n_train, dtype=np.int32))

    def training_batches():
        while True:
            for start in range(0, n_train, batch_size):
                yield x[start:start + batch_size], y[start:start + batch_size]

    def batches(inputs, labels):
        return tuple(
            (inputs[start:start + eval_batch_size], labels[start:start + eval_batch_size])
            for start in range(0, n_train, eval_batch_size)
        )

    return (
        training_batches(),
        {"train": batches(x, y), "test": batches(test_x, test_y)},
        n_train, n_classes, image_shape, "synthetic-smoke",
        {"n_train": n_train, "n_train_monitor": n_train, "n_test": n_train},
    )


# =============================================================================
# ResNet-20
# =============================================================================


CONV_INIT = nn.initializers.variance_scaling(2.0, "fan_out", "truncated_normal")


class BasicBlock(nn.Module):
    filters: int
    strides: tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x, *, train: bool):
        conv = partial(nn.Conv, use_bias=False, padding="SAME", kernel_init=CONV_INIT)
        bn = partial(nn.BatchNorm, use_running_average=not train, momentum=0.9, epsilon=1e-5)
        residual = x

        y = conv(self.filters, (3, 3), strides=self.strides, name="conv1")(x)
        y = nn.relu(bn(name="bn1")(y))
        y = conv(self.filters, (3, 3), name="conv2")(y)
        y = bn(name="bn2")(y)

        if residual.shape != y.shape:
            residual = conv(self.filters, (1, 1), strides=self.strides, name="proj_conv")(residual)
            residual = bn(name="proj_bn")(residual)
        return nn.relu(residual + y)


class ResNet20(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, *, train: bool):
        x = nn.Conv(
            16, (3, 3), use_bias=False, padding="SAME", kernel_init=CONV_INIT, name="stem_conv"
        )(x)
        x = nn.BatchNorm(
            use_running_average=not train, momentum=0.9, epsilon=1e-5, name="stem_bn"
        )(x)
        x = nn.relu(x)

        for stage, filters in enumerate((16, 32, 64)):
            for block_index in range(3):
                strides = (2, 2) if stage > 0 and block_index == 0 else (1, 1)
                x = BasicBlock(
                    filters=filters,
                    strides=strides,
                    name=f"stage{stage}_block{block_index}",
                )(x, train=train)

        x = jnp.mean(x, axis=(1, 2))
        return nn.Dense(
            self.num_classes,
            kernel_init=nn.initializers.normal(0.01),
            name="classifier",
        )(x)


# =============================================================================
# Schedules and optimizers
# =============================================================================


@struct.dataclass
class FirstOrderState:
    step: jax.Array
    opt_state: Any
    batch_stats: Any


def step_at_epoch(epoch: float, n_train: int, batch_size: int) -> int:
    """First optimizer step at or beyond an absolute dataset-pass boundary."""
    return int(math.ceil(float(epoch) * n_train / batch_size))


def epoch_at_step(step: int, n_train: int, batch_size: int) -> float:
    return float(step * batch_size / n_train)


def periodic_epoch_steps(
        interval_epochs: int,
        max_epochs: int,
        n_train: int,
        batch_size: int,
        total_steps: int,
) -> frozenset[int]:
    return frozenset(
        step
        for epoch in range(interval_epochs, max_epochs + 1, interval_epochs)
        if (step := step_at_epoch(epoch, n_train, batch_size)) <= total_steps
    )


def make_lr_schedule(
        a: argparse.Namespace,
        n_train: int,
        batch_size: int,
):
    """Decay at absolute dataset-pass boundaries (100 and 150 by default)."""
    base_lr = jnp.asarray(a.learning_rate, jnp.float32)
    milestones = tuple(
        step_at_epoch(epoch, n_train, batch_size) for epoch in a.milestone_epochs
    )

    def schedule(step):
        lr = base_lr
        for boundary in milestones:
            lr = jnp.where(step >= boundary, lr * a.lr_decay, lr)
        return jnp.asarray(lr, jnp.float32)

    return schedule


def make_wd_mask(params):
    flat = flatten_dict(params)
    mask = unflatten_dict({path: path[-1] == "kernel" for path in flat})
    return freeze(mask) if isinstance(params, FrozenDict) else mask


def apply_train(model: ResNet20, params, batch_stats, x):
    logits, mutable = model.apply(
        {"params": params, "batch_stats": batch_stats},
        x,
        train=True,
        mutable=["batch_stats"],
    )
    return logits, mutable["batch_stats"]


def build_optimizer(
        a: argparse.Namespace,
        model: ResNet20,
        params: Any,
        batch_stats: Any,
        lr_schedule,
):
    mask = make_wd_mask(params)

    if a.optimizer == "sgdm":
        transforms = []
        if a.weight_decay > 0:
            transforms.append(optax.add_decayed_weights(a.weight_decay, mask=mask))
        transforms.append(optax.sgd(lr_schedule, momentum=a.momentum, nesterov=a.nesterov))
        tx = optax.chain(*transforms)

        state = FirstOrderState(jnp.asarray(0, jnp.int32), tx.init(params), batch_stats)

        @jax.jit
        def train_step(p, state, x, y, rng):
            del rng
            step = state.step

            def loss_fn(pp):
                logits, new_stats = apply_train(model, pp, state.batch_stats, x)
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
                return loss, new_stats

            (loss, new_stats), grad = jax.value_and_grad(loss_fn, has_aux=True)(p)
            updates, opt_state = tx.update(grad, state.opt_state, p)
            p_new = optax.apply_updates(p, updates)
            state_new = FirstOrderState(step + 1, opt_state, new_stats)
            return (
                p_new,
                state_new,
                loss,
                lr_schedule(step),
                jnp.nan,
                jnp.nan,
                jnp.asarray(-1, jnp.int32),
                jnp.asarray(-1, jnp.int32),
                jnp.nan,
                jnp.asarray(False),
            )

        return state, train_step, lambda state: state.batch_stats

    def predict_fn(p, x, model_state):
        # Somax linearizes this once with model_state fixed. The returned
        # BatchNorm update is auxiliary and committed once by method.step.
        return apply_train(model, p, model_state, x)

    common = dict(
        predict_fn=predict_fn,
        has_aux=True,
        lam_policy="const",
        lam0=a.damping,
        tol=a.cg_tol,
        maxiter=a.cg_maxiter,
        stabilise_every=0,
        learning_rate=lr_schedule,
        weight_decay=a.weight_decay if a.weight_decay > 0 else None,
        weight_decay_mask=mask,
        direction_momentum=a.momentum if a.momentum > 0 else None,
        nesterov=a.nesterov,
        record_loss_every_k=1,
        record_lam=True,
        record_cg_stats=True,
    )
    if a.optimizer == "fgn_ce":
        method = somax.fgn_ce(
            **common, lifted=a.cg_lifted, warm_start=False, gram_mode="none"
        )
    else:
        method = somax.sgn_ce(**common, warm_start=True)

    state = method.init(params, model_state=batch_stats)

    @jax.jit
    def train_step(p, state, x, y, rng):
        step = state.step
        p_new, state_new, info = method.step(p, {"x": x, "y": y}, state, rng)
        return (
            p_new,
            state_new,
            info[sm.LOSS_BEFORE],
            lr_schedule(step),
            info[sm.LAM_USED],
            info[sm.LAM_NEXT],
            info[sm.CG_ITERS],
            info[sm.CG_MAXITER],
            info[sm.CG_RESID],
            info[sm.CG_CONVERGED],
        )

    return state, train_step, lambda state: state.model_state


# =============================================================================
# Evaluation
# =============================================================================


def build_eval(model: ResNet20):
    @jax.jit
    def eval_batch(params, batch_stats, x, y):
        logits = model.apply({"params": params, "batch_stats": batch_stats}, x, train=False)
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, y)
        correct = jnp.sum(jnp.argmax(logits, axis=-1) == y)
        return jnp.sum(losses), correct, y.shape[0]

    return eval_batch


def evaluate(eval_batch, params, batch_stats, batches):
    totals = [jnp.asarray(0.0) for _ in range(3)]
    for x, y in batches:
        values = eval_batch(params, batch_stats, x, y)
        totals = [a + b for a, b in zip(totals, values)]
    totals = [scalar(x) for x in block(totals)]
    n = max(totals[2], 1.0)
    return {
        "loss": totals[0] / n,
        "accuracy": totals[1] / n,
    }


# =============================================================================
# Main loop
# =============================================================================


def run(a: argparse.Namespace) -> tuple[Path, str]:
    validate(a)
    if a.device != "auto":
        jax.config.update("jax_platform_name", a.device)

    run_dir, skipped = prepare_run(a)
    if skipped:
        return run_dir, json.loads((run_dir / "summary.json").read_text())["status"]

    metrics_file = run_dir / "metrics.csv"
    process_start = time.perf_counter()
    started_at = utc_now()
    status, reason = "failed", "unknown"
    step = 0
    compile_step_s = compile_eval_s = math.nan
    train_time_s = 0.0
    eval_time_s = 0.0
    last_train_loss = math.nan
    last_train_accuracy = math.nan
    last_test: dict[str, Any] = {}
    best_test_accuracy = -math.inf
    best_test_epoch = 0.0
    step_times_ms: list[float] = []
    cg_iters_samples: list[float] = []
    cg_residual_samples: list[float] = []
    cg_converged_samples: list[float] = []
    cg_hit_budget_samples: list[float] = []
    damping_samples: list[float] = []

    try:
        np.random.seed(a.seed)
        (
            train_iter,
            eval_batches,
            n_train,
            n_classes,
            image_shape,
            dataset_version,
            split_metadata,
        ) = (
            load_synthetic(a.batch_size, a.eval_batch_size, a.seed)
            if a.smoke else load_cifar(
                a.dataset,
                a.batch_size,
                a.eval_batch_size,
                a.seed,
                a.data_dir,
                train_eval_size=a.train_eval_size,
            )
        )

        requested_total_steps = step_at_epoch(a.epochs, n_train, a.batch_size)
        total_steps = requested_total_steps
        if a.max_steps > 0:
            total_steps = min(total_steps, a.max_steps)
        eval_steps = periodic_epoch_steps(
            a.eval_every_epochs,
            a.epochs,
            n_train,
            a.batch_size,
            total_steps,
        )

        model = ResNet20(num_classes=n_classes)
        rng = jax.random.PRNGKey(a.seed)
        rng, init_key = jax.random.split(rng)
        variables = model.init(init_key, jnp.ones((a.batch_size,) + image_shape), train=True)
        params, batch_stats = variables["params"], variables["batch_stats"]
        n_params = int(sum(x.size for x in jax.tree_util.tree_leaves(params)))

        lr_schedule = make_lr_schedule(a, n_train, a.batch_size)
        state, train_step, get_batch_stats = build_optimizer(
            a,
            model,
            params,
            batch_stats,
            lr_schedule,
        )
        eval_batch = build_eval(model)
        finite = jax.jit(tree_all_finite)

        metadata = {
            "started_at": started_at,
            "run_id": run_dir.name,
            "train_monitor_seed": 20260801,
            "dataset": a.dataset,
            "dataset_version": dataset_version,
            "optimizer": a.optimizer,
            "seed": a.seed,
            "n_classes": n_classes,
            "n_params": n_params,
            "epoch_accounting": "examples_seen / n_train",
            "steps_per_epoch": n_train / a.batch_size,
            "requested_epochs": a.epochs,
            "requested_total_steps": requested_total_steps,
            "total_steps": total_steps,
            "total_examples_processed": total_steps * a.batch_size,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(x) for x in jax.devices()],
            "device_kinds": [device.device_kind for device in jax.devices()],
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": package_versions(),
            "somax_source": str(Path(somax.__file__).resolve()),
            "cg_lifted": a.cg_lifted if a.optimizer == "fgn_ce" else None,
            "sgn_warm_start": True if a.optimizer == "sgn_ce" else None,
            "training_time_definition": (
                "synchronized training segments including loop/input overhead; "
                "excludes evaluation, logging, and the first input wait of each segment"
            ),
            "step_time_definition": "median of per-segment mean training-step times",
            "cg_statistics_definition": "samples at logging/evaluation boundaries, not every update",
            "train_metrics_definition": "fixed unaugmented monitor subset, inference BatchNorm",
            "weight_decay_definition": (
                "kernel-only decay before momentum" if a.optimizer == "sgdm"
                else "kernel-only decay after direction momentum"
            ),
            **split_metadata,
        }
        dump_json(run_dir / "metadata.json", metadata)

        # Compile on the next real batch and discard all outputs, including BN state.
        warm_x, warm_y = next(train_iter)
        block((warm_x, warm_y))
        compile_key = jax.random.PRNGKey(0)
        t0 = time.perf_counter()
        compiled_out = train_step(params, state, warm_x, warm_y, compile_key)
        block(compiled_out)
        compile_step_s = time.perf_counter() - t0
        del compiled_out

        # Compile every evaluation batch shape, including a possible short tail.
        t0 = time.perf_counter()
        compiled_shapes = set()
        for batches in eval_batches.values():
            for eval_x, eval_y in batches:
                shape = (eval_x.shape, eval_y.shape)
                if shape not in compiled_shapes:
                    block(eval_batch(params, get_batch_stats(state), eval_x, eval_y))
                    compiled_shapes.add(shape)
        compile_eval_s = time.perf_counter() - t0
        if a.stop_on_instability:
            block(finite((params, state)))

        run_start = time.perf_counter()

        if a.initial_eval:
            t0 = time.perf_counter()
            train_metrics = evaluate(
                eval_batch, params, get_batch_stats(state), eval_batches["train"]
            )
            selected_metrics = evaluate(
                eval_batch, params, get_batch_stats(state), eval_batches["test"]
            )
            initial_eval_s = time.perf_counter() - t0
            eval_time_s += initial_eval_s
            last_train_loss = train_metrics["loss"]
            last_train_accuracy = train_metrics["accuracy"]
            last_test = selected_metrics
            best_test_accuracy = selected_metrics["accuracy"]
            best_test_epoch = 0.0

            append_csv(
                metrics_file,
                event="eval",
                status="running",
                step=0,
                epoch=0.0,
                examples_seen=0,
                train_loss=train_metrics["loss"],
                train_accuracy=train_metrics["accuracy"],
                test_loss=selected_metrics["loss"],
                test_accuracy=selected_metrics["accuracy"],
                learning_rate=a.learning_rate,
                damping=a.damping if a.optimizer in {"fgn_ce", "sgn_ce"} else None,
                train_time_s=train_time_s,
                eval_duration_s=initial_eval_s,
                eval_time_s=eval_time_s,
                wall_time_s=time.perf_counter() - run_start,
                compile_step_s=compile_step_s,
                compile_eval_s=compile_eval_s,
            )

        status = "running"
        segment_start = None
        segment_steps = 0
        last_step_ms = math.nan

        for index in range(total_steps):
            x, y = (warm_x, warm_y) if index == 0 else next(train_iter)
            if segment_steps == 0:
                block((x, y))
                segment_start = time.perf_counter()

            rng, step_key = jax.random.split(rng)
            out = train_step(params, state, x, y, step_key)
            params, state = out[0], out[1]
            step = index + 1
            segment_steps += 1

            do_log = step % a.log_every == 0 or step == total_steps
            do_eval = step in eval_steps or step == total_steps
            if not (do_log or do_eval):
                continue

            block(out)
            segment_s = time.perf_counter() - float(segment_start)
            train_time_s += segment_s
            last_step_ms = 1000.0 * segment_s / segment_steps
            step_times_ms.append(last_step_ms)
            examples_per_second = a.batch_size * segment_steps / max(segment_s, 1e-12)
            segment_start, segment_steps = None, 0

            (
                _,
                _,
                loss_before,
                lr,
                damping,
                damping_next,
                cg_iters,
                cg_maxiter,
                cg_resid,
                cg_converged,
            ) = out
            loss_before = scalar(loss_before)
            lr = scalar(lr)
            damping = scalar(damping)
            damping_next = scalar(damping_next)
            cg_iters = int(np.asarray(jax.device_get(cg_iters)))
            cg_maxiter = int(np.asarray(jax.device_get(cg_maxiter)))
            cg_resid = scalar(cg_resid)
            cg_converged = bool(np.asarray(jax.device_get(cg_converged)))

            if cg_iters >= 0:
                hit_budget = cg_iters >= cg_maxiter and not cg_converged
                cg_iters_samples.append(float(cg_iters))
                cg_converged_samples.append(float(cg_converged))
                cg_hit_budget_samples.append(float(hit_budget))
                if math.isfinite(cg_resid):
                    cg_residual_samples.append(cg_resid)
                if math.isfinite(damping):
                    damping_samples.append(damping)

            eval_duration_s = 0.0
            train_metrics = None
            selected_metrics = None

            if do_eval:
                t0 = time.perf_counter()
                train_metrics = evaluate(
                    eval_batch, params, get_batch_stats(state), eval_batches["train"]
                )
                selected_metrics = evaluate(
                    eval_batch,
                    params,
                    get_batch_stats(state),
                    eval_batches["test"],
                )
                eval_duration_s = time.perf_counter() - t0
                eval_time_s += eval_duration_s
                last_train_loss = train_metrics["loss"]
                last_train_accuracy = train_metrics["accuracy"]

                last_test = selected_metrics
                epoch = epoch_at_step(step, n_train, a.batch_size)
                if selected_metrics["accuracy"] > best_test_accuracy:
                    best_test_accuracy = selected_metrics["accuracy"]
                    best_test_epoch = epoch

            nonfinite_event = False
            if a.stop_on_instability:
                state_finite = bool(np.asarray(jax.device_get(finite((params, state)))))
                nonfinite_event = not state_finite or not math.isfinite(loss_before)

            append_csv(
                metrics_file,
                event="eval" if do_eval else "train",
                status="running",
                step=step,
                epoch=epoch_at_step(step, n_train, a.batch_size),
                examples_seen=step * a.batch_size,
                batch_loss=loss_before,
                train_loss=train_metrics["loss"] if train_metrics else None,
                train_accuracy=train_metrics["accuracy"] if train_metrics else None,
                test_loss=selected_metrics["loss"] if selected_metrics else None,
                test_accuracy=selected_metrics["accuracy"] if selected_metrics else None,
                learning_rate=lr,
                damping=damping if math.isfinite(damping) else None,
                damping_next=damping_next if math.isfinite(damping_next) else None,
                cg_iters=cg_iters if cg_iters >= 0 else None,
                cg_maxiter=cg_maxiter if cg_maxiter >= 0 else None,
                cg_residual_final=cg_resid if math.isfinite(cg_resid) else None,
                cg_converged=cg_converged if cg_iters >= 0 else None,
                cg_hit_budget=hit_budget if cg_iters >= 0 else None,
                nonfinite_event=nonfinite_event,
                step_ms=last_step_ms,
                examples_per_second=examples_per_second,
                train_time_s=train_time_s,
                eval_duration_s=eval_duration_s,
                eval_time_s=eval_time_s,
                wall_time_s=time.perf_counter() - run_start,
                compile_step_s=compile_step_s,
                compile_eval_s=compile_eval_s,
            )

            split_text = ""
            if selected_metrics:
                split_text = f" | test_acc={selected_metrics['accuracy']:.4f}"
            residual_text = f" | resid={cg_resid:.2e}" if math.isfinite(cg_resid) else ""
            print(
                f"step={step:6d}/{total_steps} | "
                f"epoch={epoch_at_step(step, n_train, a.batch_size):6.2f}"
                f" | batch_loss={loss_before:.4f}{split_text} | step_ms={last_step_ms:.2f}"
                f" | lam={damping:.4g} | cg={cg_iters if cg_iters >= 0 else '-'}"
                f"{residual_text}",
                flush=True,
            )

            if nonfinite_event:
                status, reason = "failed", "nonfinite parameters, BatchNorm state, or loss"
                break

        if status == "running":
            status, reason = "completed", "completed all requested steps"

    except KeyboardInterrupt:
        status, reason = "interrupted", "keyboard interrupt or SIGTERM"
    except Exception as exc:
        status, reason = "failed", f"{type(exc).__name__}: {exc}"
        error = traceback.format_exc()
        (run_dir / "error.txt").write_text(error)
        print(error, file=sys.stderr)

    wall_time_s = time.perf_counter() - locals().get("run_start", process_start)
    summary = {
        "status": status,
        "started_at": started_at,
        "reason": reason,
        "dataset": a.dataset,
        "optimizer": a.optimizer,
        "seed": a.seed,
        "steps_completed": step,
        "epochs_completed": (
            epoch_at_step(step, n_train, a.batch_size)
            if "n_train" in locals()
            else 0.0
        ),
        "compile_step_s": compile_step_s,
        "compile_eval_s": compile_eval_s,
        "step_time_mean_ms": 1000.0 * train_time_s / max(step, 1),
        "step_time_median_ms": float(np.median(step_times_ms)) if step_times_ms else None,
        "train_time_s": train_time_s,
        "eval_time_s": eval_time_s,
        "wall_time_s": wall_time_s,
        "process_time_s": time.perf_counter() - process_start,
        "finished_at": utc_now(),
        "logged_cg_iters_mean": (float(np.mean(cg_iters_samples)) if cg_iters_samples else None),
        "logged_cg_samples": len(cg_iters_samples),
        "logged_cg_residual_median": (
            float(np.median(cg_residual_samples)) if cg_residual_samples else None
        ),
        "logged_cg_converged_fraction": (
            float(np.mean(cg_converged_samples)) if cg_converged_samples else None
        ),
        "logged_cg_hit_budget_fraction": (
            float(np.mean(cg_hit_budget_samples)) if cg_hit_budget_samples else None
        ),
        "logged_damping_mean": (float(np.mean(damping_samples)) if damping_samples else None),
        "final_train_loss": last_train_loss,
        "final_train_accuracy": last_train_accuracy,
        "final_test_loss": last_test.get("loss"),
        "final_test_accuracy": last_test.get("accuracy"),
        "best_test_accuracy": best_test_accuracy if math.isfinite(best_test_accuracy) else None,
        "best_test_epoch": best_test_epoch,
    }
    if "metadata" in locals():
        summary.update(metadata)
    dump_json(run_dir / "summary.json", summary)
    append_csv(
        metrics_file,
        event="status",
        status=status,
        step=step,
        epoch=summary["epochs_completed"],
        examples_seen=step * a.batch_size,
        train_loss=last_train_loss,
        train_accuracy=last_train_accuracy,
        test_loss=summary["final_test_loss"],
        test_accuracy=summary["final_test_accuracy"],
        train_time_s=train_time_s,
        eval_time_s=eval_time_s,
        wall_time_s=wall_time_s,
        compile_step_s=compile_step_s,
        compile_eval_s=compile_eval_s,
        reason=reason,
    )
    (run_dir / "RUNNING").unlink(missing_ok=True)
    for marker in ("COMPLETED", "FAILED", "INTERRUPTED"):
        (run_dir / marker).unlink(missing_ok=True)
    (run_dir / status.upper()).touch()
    print(f"{status.upper()}: {run_dir} | {reason}", flush=True)
    return run_dir, status


def main() -> None:
    def stop(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    _, status = run(parse_args())
    if status == "failed":
        raise SystemExit(1)
    if status == "interrupted":
        raise SystemExit(130)


if __name__ == "__main__":
    main()
