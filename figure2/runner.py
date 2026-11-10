"""Run the fixed Cars196 head benchmark and regenerate its figures and summaries."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from common import artifact_root
from configs import DEFAULTS_2, METHOD_KWARGS_2, METHOD_ORDER_2

HERE = Path(__file__).resolve().parent


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--methods", nargs="+", choices=METHOD_ORDER_2, default=METHOD_ORDER_2)
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--device", choices=["gpu", "cpu"], default=None)
    p.add_argument("--smoke", action="store_true", help="Use tiny synthetic features; no downloads.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--overwrite", action="store_true", help="Replace existing Figure 2 results.")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--n-epochs", type=int)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--log-every", type=int)
    args = p.parse_args()
    args.device = args.device or ("cpu" if args.smoke else "gpu")
    if args.plot_only and args.overwrite:
        p.error("--plot-only and --overwrite cannot be combined")
    args.seeds = args.seeds if args.seeds is not None else ([1] if args.smoke else DEFAULTS_2["seeds"])
    if len(args.seeds) != len(set(args.seeds)) or len(args.methods) != len(set(args.methods)):
        p.error("--seeds and --methods must not contain duplicates")
    if any(seed < 0 for seed in args.seeds):
        p.error("--seeds must be nonnegative")
    for name in ("batch_size", "n_epochs", "eval_every", "log_every"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.smoke and args.feature_cache is not None:
        p.error("--smoke creates its own synthetic cache; omit --feature-cache")
    if args.smoke and "FGN_ARTIFACTS_DIR" not in os.environ:
        os.environ["FGN_ARTIFACTS_DIR"] = str(artifact_root() / "smoke")
    if args.feature_cache is None:
        args.feature_cache = (
            artifact_root() / "features" / "smoke_head.npz"
            if args.smoke else Path(DEFAULTS_2["feature_cache"])
        )
    args.feature_cache = args.feature_cache.expanduser().resolve()
    return args


def extend_cmd(command, values):
    for name, value in values.items():
        if value is not None:
            command.extend(
                ["--" + name.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)])


def make_smoke_cache(path):
    import json
    import numpy as np

    rng = np.random.default_rng(17)
    weights = rng.normal(size=(8, 3))
    x_train = rng.normal(size=(32, 8)).astype(np.float32)
    x_test = rng.normal(size=(16, 8)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x_train=x_train,
        y_train=np.argmax(x_train @ weights, axis=1).astype(np.int32),
        x_val=x_test,
        y_val=np.argmax(x_test @ weights, axis=1).astype(np.int32),
        metadata_json=np.asarray(json.dumps({
            "dataset_id": "smoke", "encoder_name": "synthetic", "n_classes": 3,
            "feature_dim": 8, "eval_split_name": "test", "cache_kind": "synthetic_smoke",
        })),
    )


def main():
    args = parse_args()
    output = artifact_root() / "figure2"
    raw = output / "raw"
    plot = [
        sys.executable, str(HERE / "plot.py"),
        "--runs-csv", str(raw / "runs.csv"),
        "--traces-eval-csv", str(raw / "traces_eval.csv"),
        "--out-dir-derived", str(output / "derived"),
        "--out-dir-plots", str(output / "plots"),
        "--band", "std",
    ]
    if args.smoke:
        plot += ["--acc-ymin", "nan", "--acc-ymax", "nan"]

    common = {
        "feature_cache": args.feature_cache,
        "device": args.device,
        "batch_size": args.batch_size or (8 if args.smoke else DEFAULTS_2["batch_size"]),
        "n_epochs": args.n_epochs or (2 if args.smoke else DEFAULTS_2["n_epochs"]),
        "eval_every": args.eval_every or (2 if args.smoke else DEFAULTS_2["eval_every"]),
        "log_every": args.log_every or (2 if args.smoke else DEFAULTS_2["log_every"]),
        "init_scale": DEFAULTS_2["init_scale"],
    }
    jobs = []
    if not args.plot_only:
        for method in args.methods:
            settings = dict(METHOD_KWARGS_2[method])
            rates = settings.pop("learning_rates")
            if args.smoke and "cg_maxiter" in settings:
                settings["cg_maxiter"] = 2
            for rate in rates:
                for seed in args.seeds:
                    command = [sys.executable, str(HERE / "train_head.py")]
                    extend_cmd(command,
                               {**common, **settings, "method_id": method, "learning_rate": rate, "seed": seed})
                    jobs.append(command)
    jobs.append(plot)
    if args.dry_run:
        if args.smoke and not args.plot_only:
            print(f"Create synthetic smoke cache: {args.feature_cache}")
        for command in jobs:
            print(shlex.join(command))
        return

    if not args.plot_only:
        if raw.exists() and any(raw.iterdir()) and not args.overwrite:
            raise SystemExit(
                f"Results already exist: {raw}. Use --plot-only, --overwrite, or another artifact directory.")
        if not args.smoke and not args.feature_cache.is_file():
            raise SystemExit(
                f"Missing feature cache: {args.feature_cache}\n"
                "Run python reproduce_results.py --figures 2 --stage prepare first."
            )
        if args.overwrite:
            for name in ("raw", "derived", "plots"):
                path = output / name
                if path.exists():
                    shutil.rmtree(path)
        if args.smoke:
            make_smoke_cache(args.feature_cache)

    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "JAX_LOG_COMPILES": "0"}
    if args.device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    for index, command in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {shlex.join(command)}", flush=True)
        subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
