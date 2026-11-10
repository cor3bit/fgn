"""Run the mechanism checks and regenerate Figure 1."""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from common import artifact_root, write_json
from configs import (
    CLASS_COUNTS_1A,
    DEFAULTS_1A,
    DEFAULTS_1B,
    DEFAULTS_1C,
    METHOD_KWARGS_1A,
    METHOD_ORDER_1A,
    SEEDS_1A,
)


def extend_cmd(command: list[str], values: dict) -> None:
    for key, value in values.items():
        if value is not None:
            command.extend(
                ["--" + key.replace("_", "-"), str(value).lower() if isinstance(value, bool) else str(value)])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["gpu", "cpu"])
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--smoke", action="store_true", help="Tiny run to check the pipeline; not paper results.")
    parser.add_argument("--plot-only", action="store_true", help="Regenerate plots and summaries from saved CSVs.")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing Figure 1 outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them or changing files.")
    args = parser.parse_args()
    args.device = args.device or ("cpu" if args.smoke else DEFAULTS_1A["device"])
    args.seeds = args.seeds if args.seeds is not None else ([1] if args.smoke else SEEDS_1A)
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("--seeds must be distinct nonnegative integers")
    if args.plot_only and args.overwrite:
        parser.error("--plot-only and --overwrite cannot be combined")
    return args


def main():
    args = parse_args()
    here = Path(__file__).resolve().parent
    if args.smoke and "FGN_ARTIFACTS_DIR" not in os.environ:
        os.environ["FGN_ARTIFACTS_DIR"] = str(here.parent / "artifacts" / "smoke")
    root = artifact_root()
    raw = root / "raw"
    env = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "JAX_LOG_COMPILES": "0"}
    if args.device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"

    jobs = []
    if not args.plot_only:
        common = dict(DEFAULTS_1A, device=args.device, artifact_subdir="1a")
        if args.smoke:
            common.update(feat_dim=8, batch_size=4, n_samples=16, warmup_steps=2, timed_steps=4, measure_every=2)
        for method in METHOD_ORDER_1A:
            for classes in ([2, 8] if args.smoke else CLASS_COUNTS_1A):
                for seed in args.seeds:
                    command = [sys.executable, str(here / "1a_generate.py")]
                    extend_cmd(command, {"method_id": method, "n_classes": classes, "seed": seed, **common,
                                         **METHOD_KWARGS_1A[method]})
                    jobs.append(command)

        command = [sys.executable, str(here / "1b_generate.py")]
        extend_cmd(command, {**DEFAULTS_1B, "artifact_subdir": "1b"})
        jobs.append(command)
        config_1c = dict(DEFAULTS_1C)
        config_1c["lambda_regime_scales"] = ",".join(
            f"{label}:{scale:g}" for label, scale in config_1c["lambda_regime_scales"].items())
        command = [sys.executable, str(here / "1c_generate.py")]
        extend_cmd(command, {**config_1c, "artifact_subdir": "1c"})
        jobs.append(command)

    jobs.append([
        sys.executable, str(here / "plot.py"), "--panel", "all", "--band", "std",
        "--runs-csv", str(raw / "1a/runs.csv"),
        "--panel-1b-csv", str(raw / "1b/decomposition_traces.csv"),
        "--panel-1c-csv", str(raw / "1c/decrease_ratios.csv"),
        "--out-dir-derived", str(root / "derived"), "--out-dir-plots", str(root / "plots"),
    ])
    if args.dry_run:
        for command in jobs:
            print(shlex.join(command))
        print(f"Outputs: {root}")
        return

    if not args.plot_only:
        if raw.exists() and any(raw.rglob("*.csv")) and not args.overwrite:
            raise SystemExit(
                f"Existing Figure 1 data in {raw}. Use --plot-only, a new --artifacts-dir, or --overwrite.")
        if args.overwrite:
            for directory in (raw, root / "derived", root / "plots"):
                if directory.exists():
                    shutil.rmtree(directory)
        write_json(root / "run_manifest.json", {
            "status": "running", "smoke": args.smoke, "device": args.device, "seeds": args.seeds, "commands": jobs,
        })

    for index, command in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {shlex.join(command)}", flush=True)
        subprocess.run(command, env=env, check=True)
    if not args.plot_only:
        write_json(root / "run_manifest.json", {
            "status": "completed", "smoke": args.smoke, "device": args.device, "seeds": args.seeds, "commands": jobs,
        })
    print(f"Figure 1 outputs: {root}")


if __name__ == "__main__":
    main()
