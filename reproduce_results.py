"""Prepare data, run the three experiments, and rebuild figures and summaries."""

import argparse
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE_NAME = "cars196__torchvision_resnet50_imagenet1k_v2.npz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", nargs="+", type=int, choices=(1, 2, 3), default=[1, 2, 3],
                        help="Experiment folders to run (default: 1 2 3).")
    parser.add_argument("--stage", choices=("all", "prepare", "run", "plot"), default="all",
                        help="all: prepare missing features, train, plot, and summarize; "
                             "prepare: Cars196 features only; run: skip feature extraction; "
                             "plot: regenerate figures and summaries from saved results.")
    parser.add_argument("--device", choices=("gpu", "cpu"),
                        help="Default: gpu; cpu for --smoke.")
    parser.add_argument("--seeds", nargs="+", type=int,
                        help="Override seeds for every selected experiment.")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "artifacts",
                        help="Shared output/download root. Relative paths use the current directory.")
    parser.add_argument("--feature-cache", type=Path,
                        help="Use an existing Cars196 feature cache, or create it at this path.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny synthetic runs without downloads, saved under <artifacts-dir>/smoke.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Explicitly replace selected experiment results; keep downloaded data/features.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print preparation and training commands without writing files.")
    args = parser.parse_args()
    if len(args.figures) != len(set(args.figures)):
        parser.error("--figures must not contain duplicates")
    if args.seeds is not None and (min(args.seeds) < 0 or len(args.seeds) != len(set(args.seeds))):
        parser.error("--seeds must be distinct nonnegative integers")
    if args.stage in ("prepare", "plot") and args.overwrite:
        parser.error("--overwrite applies to training; omit it for preparation or plotting")
    if args.stage in ("prepare", "plot") and args.seeds is not None:
        parser.error("--seeds selects training runs; plotting uses all saved seeds")
    if args.smoke and args.stage == "prepare":
        parser.error("--smoke needs no data preparation")
    if args.smoke and args.feature_cache is not None:
        parser.error("--smoke creates synthetic features; omit --feature-cache")
    if args.feature_cache is not None and 2 not in args.figures:
        parser.error("--feature-cache is only used by experiment 2")
    args.device = args.device or ("cpu" if args.smoke else "gpu")
    args.artifacts_dir = args.artifacts_dir.expanduser().resolve()
    if args.smoke:
        args.artifacts_dir /= "smoke"
    if args.feature_cache is not None:
        args.feature_cache = args.feature_cache.expanduser().resolve()
    return args


def environment(artifacts, device):
    env = dict(os.environ)
    env.update({
        "FGN_ARTIFACTS_DIR": str(artifacts),
        "MPLBACKEND": "Agg",
        "MPLCONFIGDIR": str(artifacts / "cache" / "matplotlib"),
        "TORCH_HOME": str(artifacts / "cache" / "torch"),
        "HF_HOME": str(artifacts / "cache" / "huggingface"),
        "HF_DATASETS_CACHE": str(artifacts / "cache" / "huggingface" / "datasets"),
        "HF_HUB_CACHE": str(artifacts / "cache" / "huggingface" / "hub"),
        "TFDS_DATA_DIR": str(artifacts / "data" / "tfds"),
        "TF_CPP_MIN_LOG_LEVEL": "3",
        "JAX_LOG_COMPILES": "0",
    })
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
    return env


def version_record():
    versions = {}
    for name in ("python-somax", "jax", "jaxlib", "flax", "optax", "numpy", "pandas",
                 "matplotlib", "torch", "torchvision", "datasets", "tensorflow",
                 "tensorflow-cpu", "tensorflow-datasets"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def main():
    args = parse_args()
    artifacts = args.artifacts_dir
    env = environment(artifacts, args.device)
    feature_cache = args.feature_cache or artifacts / "features" / CACHE_NAME
    commands = []

    if (args.stage == "run" and 2 in args.figures and not args.smoke
            and not args.dry_run and not feature_cache.is_file()):
        raise SystemExit(f"Missing feature cache: {feature_cache}. "
                         "Run --figures 2 --stage prepare first, or supply --feature-cache.")

    if args.stage in ("all", "run") and not args.overwrite and not args.dry_run:
        for figure in set(args.figures) & {1, 2}:
            raw = artifacts / f"figure{figure}" / "raw"
            if raw.exists() and any(raw.iterdir()):
                raise SystemExit(f"Existing experiment {figure} results: {raw}. "
                                 "Use --stage plot, --overwrite, or a new --artifacts-dir.")

    if args.stage in ("all", "prepare") and 2 in args.figures and not args.smoke:
        if feature_cache.is_file():
            print(f"Using feature cache: {feature_cache}", flush=True)
        else:
            commands.append(("prepare", [
                sys.executable, str(ROOT / "figure2" / "make_feature_cache.py"),
                "--out-cache", str(feature_cache), "--extract-device", "auto",
            ]))

    if args.stage != "prepare":
        for figure in args.figures:
            command = [sys.executable, str(ROOT / f"figure{figure}" / "runner.py"),
                       "--device", args.device]
            if args.smoke:
                command.append("--smoke")
            if args.stage == "plot":
                command.append("--plot-only")
            elif args.overwrite:
                command.append("--overwrite")
            if args.seeds is not None:
                command.extend(["--seeds", *map(str, args.seeds)])
            if figure == 2 and not args.smoke:
                command.extend(["--feature-cache", str(feature_cache)])
            commands.append((f"figure{figure}", command))

    print(f"Artifacts: {artifacts}", flush=True)
    if args.smoke:
        print("Synthetic smoke checks; these are not paper results.", flush=True)
    if not commands:
        print("No preparation needed. CIFAR-100 downloads on its first training run.")
        return

    if args.dry_run:
        for kind, command in commands:
            print(shlex.join(command), flush=True)
            if kind != "prepare":
                subprocess.run([*command, "--dry-run"], cwd=ROOT, env=env, check=True)
        return

    started = datetime.now(timezone.utc)
    record = {
        "started_at": started.isoformat(), "status": "running",
        "arguments": vars(args), "python": sys.version, "platform": platform.platform(),
        "packages": version_record(), "commands": [command for _, command in commands],
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(ROOT.glob("*.py")) + sorted(ROOT.glob("figure*/*.py"))
        },
    }
    manifest = artifacts / "manifests" / (started.strftime("%Y%m%d-%H%M%S-%f") + ".json")
    manifest.parent.mkdir(parents=True, exist_ok=True)

    def save_record():
        manifest.write_text(json.dumps(record, indent=2, default=str) + "\n")

    save_record()
    try:
        for kind, command in commands:
            print(shlex.join(command), flush=True)
            subprocess.run(command, cwd=ROOT, env=env, check=True)
        record["status"] = "completed"
    except BaseException as exc:
        record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        record["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_record()

    print(f"Done. Results: {artifacts}")


if __name__ == "__main__":
    main()
