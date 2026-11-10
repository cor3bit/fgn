import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("FGN_ARTIFACTS_DIR", HERE.parent / "artifacts")).resolve()
TRAIN = HERE / "train.py"
PLOT = HERE / "plot.py"
ENV = {**os.environ, "TF_CPP_MIN_LOG_LEVEL": "3", "JAX_LOG_COMPILES": "0"}

CONFIGS: list[dict[str, Any]] = [
    {
        "name": "sgdm_step",
        "optimizer": "sgdm",
        "args": {
            "learning_rate": 0.1,
            "milestone_epochs": (100, 150),
            "lr_decay": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "nesterov": True,
        },
    },
    {
        "name": "fgn_cg5_lifted",
        "optimizer": "fgn_ce",
        "args": {
            "learning_rate": 0.3,
            "milestone_epochs": (100, 150),
            "lr_decay": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "nesterov": True,
            "damping": 1.0,
            "cg_maxiter": 5,
            "cg_tol": 1e-5,
            "cg_lifted": True,
        },
    },
    {
        "name": "sgn_cg5",
        "optimizer": "sgn_ce",
        "args": {
            "learning_rate": 0.3,
            "milestone_epochs": (100, 150),
            "lr_decay": 0.1,
            "weight_decay": 5e-4,
            "momentum": 0.9,
            "nesterov": True,
            "damping": 1.0,
            "cg_maxiter": 5,
            "cg_tol": 1e-5,
        },
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce ResNet-20 / CIFAR-100 results.")
    p.add_argument("--configs", nargs="+", choices=[c["name"] for c in CONFIGS],
                   help="Run only the specified configurations (default: all).")
    p.add_argument("--seeds", nargs="+", type=int, help="Default: 1 2 3 4 5 (one seed for smoke).")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=1000)
    p.add_argument("--train-eval-size", type=int, default=5000)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--eval-every-epochs", type=int, default=5)
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], help="Default: gpu (cpu for smoke).")
    p.add_argument("--data-dir", type=Path)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--plot-only", action="store_true", help="Rebuild plots and aggregates without training.")
    p.add_argument("--smoke", action="store_true", help="Run two updates on synthetic inputs; no downloads.")
    args = p.parse_args()
    args.seeds = args.seeds or ([1] if args.smoke else [1, 2, 3, 4, 5])
    args.device = args.device or ("cpu" if args.smoke else "gpu")
    if args.smoke:
        args.epochs = 1
        args.max_steps = 2
        args.batch_size = args.eval_batch_size = 2
        args.train_eval_size = 4
        args.log_every = args.eval_every_epochs = 1
    if args.plot_only and args.overwrite:
        p.error("--plot-only and --overwrite cannot be combined")
    if any(seed < 0 for seed in args.seeds):
        p.error("--seeds must be nonnegative")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("--seeds must not contain duplicates")
    if args.configs and len(set(args.configs)) != len(args.configs):
        p.error("--configs must not contain duplicates")
    artifact_root = ARTIFACTS
    if args.smoke and "FGN_ARTIFACTS_DIR" not in os.environ:
        artifact_root = artifact_root / "smoke"
    if args.output_dir is None:
        args.output_dir = artifact_root / "figure3"
    if args.data_dir is None:
        args.data_dir = artifact_root / "data" / "tfds"
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    return args


def add_args(command: list[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        if value is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(flag if value else "--no-" + key.replace("_", "-"))
        elif isinstance(value, (tuple, list)):
            command.extend([flag, ",".join(map(str, value))])
        else:
            command.extend([flag, str(value)])


def flatten_summary(summary, dataset, config, run_dir) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "config_name": config["name"],
        "optimizer": config["optimizer"],
        "run_dir": str(run_dir),
    }
    row.update({k: v for k, v in summary.items() if not isinstance(v, (dict, list))})
    for key, value in config["args"].items():
        row[f"cfg_{key}"] = ",".join(map(str, value)) if isinstance(value, (tuple, list)) else value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate(path: Path, rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "completed":
            key = (row["dataset"], row["optimizer"], row["config_name"])
            groups.setdefault(key, []).append(row)

    for dataset in {key[0] for key in groups}:
        selected = {name: runs for (ds, _, name), runs in groups.items() if ds == dataset}
        seeds = {name: tuple(sorted(int(row["seed"]) for row in runs))
                 for name, runs in selected.items()}
        if any(len(values) != len(set(values)) for values in seeds.values()):
            raise ValueError(f"Duplicate seeds in {dataset} results")
        if len(set(seeds.values())) > 1:
            raise ValueError(f"Methods use different seed sets for {dataset}: {seeds}")
        horizons = {row["steps_completed"] for runs in selected.values() for row in runs}
        if len(horizons) != 1:
            raise ValueError(f"Methods use different training horizons for {dataset}: {sorted(horizons)}")

    metric_names = (
        "final_train_loss", "final_train_accuracy", "final_test_loss", "final_test_accuracy",
        "step_time_mean_ms", "step_time_median_ms", "train_time_s", "wall_time_s",
        "logged_cg_iters_mean", "logged_cg_residual_median", "logged_cg_converged_fraction",
        "logged_cg_hit_budget_fraction",
    )
    output = []
    for (dataset, optimizer, name), runs in sorted(groups.items()):
        horizons = {row["steps_completed"] for row in runs}
        if len(horizons) != 1:
            raise ValueError(f"Unequal final horizons for {dataset}/{name}: {sorted(horizons)}")
        aggregate = {
            "dataset": dataset,
            "optimizer": optimizer,
            "config_name": name,
            "n_runs": len(runs),
            "seeds": ",".join(str(row["seed"]) for row in runs),
            "steps_completed": next(iter(horizons)),
            "epochs_completed": statistics.mean(row["epochs_completed"] for row in runs),
            "std_definition": "sample standard deviation (ddof=1)",
        }
        for key in metric_names:
            values = [float(row[key]) for row in runs if row.get(key) is not None]
            if values:
                aggregate[f"{key}_mean"] = statistics.mean(values)
                aggregate[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else None
        output.append(aggregate)
    write_csv(path, output)


def completed_jobs(output_dir: Path, configs):
    """Find all completed seeds, including seeds from earlier invocations."""
    found = []
    run_protocol = None
    for config in configs:
        parent = output_dir / "runs" / "cifar100" / config["optimizer"] / config["name"]
        seed_dirs = (p for p in parent.glob("seed*") if p.is_dir() and p.name[4:].isdigit())
        for run_dir in sorted(seed_dirs, key=lambda p: int(p.name[4:])):
            summary_path = run_dir / "summary.json"
            if not summary_path.exists():
                continue
            summary = json.loads(summary_path.read_text())
            if summary.get("status") != "completed":
                continue
            seed = int(summary["seed"])
            if run_dir.name != f"seed{seed}" or summary["optimizer"] != config["optimizer"]:
                raise ValueError(f"Inconsistent run identity: {run_dir}")
            current_protocol = json.loads((run_dir / "protocol.json").read_text())
            if run_protocol is not None and current_protocol != run_protocol:
                raise ValueError("Completed seeds use different train/Somax versions; cannot aggregate")
            run_protocol = current_protocol
            found.append(("cifar100", config, seed, run_dir, None))
    return found


def plot_results(output_dir: Path, configs, *, no_plot: bool = False) -> int:
    jobs = completed_jobs(output_dir, configs)
    if not jobs:
        raise FileNotFoundError(f"No completed runs under {output_dir / 'runs'}")
    missing = {config["name"] for config in configs} - {config["name"] for _, config, _, _, _ in jobs}
    if missing:
        raise ValueError(f"Selected configurations have no completed runs: {sorted(missing)}")
    rows = [flatten_summary(json.loads((path / "summary.json").read_text()), ds, cfg, path)
            for ds, cfg, _, path, _ in jobs]
    write_csv(output_dir / "runs.csv", rows)
    write_aggregate(output_dir / "aggregate.csv", rows)
    if not no_plot:
        subprocess.run([
            sys.executable, str(PLOT), "--input-dir", str(output_dir / "runs"),
            "--output-dir", str(output_dir / "plots"),
            "--configs", *[config["name"] for config in configs],
        ], env=ENV, check=True)
    return len(jobs)


def main() -> None:
    args = parse_args()
    if args.device == "cpu":
        ENV["JAX_PLATFORMS"] = "cpu"
    manifest_path = args.output_dir / "run_manifest.json"
    if args.plot_only:
        if args.dry_run:
            print(f"Rebuild plots and aggregates from {args.output_dir}")
            return
        if not manifest_path.exists():
            raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        configs = [c for c in manifest["protocol"]["configs"]
                   if args.configs is None or c["name"] in args.configs]
        if not configs:
            raise ValueError("No selected configurations in the run manifest")
        plot_results(args.output_dir, configs, no_plot=args.no_plot)
        return
    configs = [{**c, "args": dict(c["args"])} for c in CONFIGS
               if args.configs is None or c["name"] in args.configs]
    if args.smoke:
        for config in configs:
            if "cg_maxiter" in config["args"]:
                config["args"]["cg_maxiter"] = 1
    common = {
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "train_eval_size": args.train_eval_size,
        "log_every": args.log_every,
        "eval_every_epochs": args.eval_every_epochs,
        "device": args.device,
        "data_dir": args.data_dir,
    }
    if args.smoke:
        common["smoke"] = True
    jobs = []
    for config in configs:
        for seed in args.seeds:
            run_dir = args.output_dir / "runs" / "cifar100" / config["optimizer"] / config["name"] / f"seed{seed}"
            command = [sys.executable, str(TRAIN)]
            add_args(command, {
                **common, "dataset": "cifar100", "optimizer": config["optimizer"],
                "seed": seed, "run_dir": run_dir, **config["args"],
            })
            if args.overwrite:
                command.append("--overwrite")
            jobs.append(("cifar100", config, seed, run_dir, command))

    if args.dry_run:
        for _, _, _, _, command in jobs:
            print(" ".join(command))
        print(f"{len(jobs)} runs; output_dir={args.output_dir}")
        return

    protocol = json.loads(json.dumps({
        **common, "dataset": "cifar100", "seeds": args.seeds, "configs": configs,
    }, default=str))
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not old and not args.overwrite:
        raise SystemExit("Nonempty output directory has no manifest; choose a new --output-dir.")
    if old:
        previous = old["protocol"]
        previous_settings = {k: v for k, v in previous.items() if k not in ("seeds", "configs")}
        current_settings = {k: v for k, v in protocol.items() if k not in ("seeds", "configs")}
        if previous_settings != current_settings:
            raise SystemExit("Experiment settings changed; choose a new --output-dir.")
        old_configs = {c["name"]: c for c in previous["configs"]}
        new_configs = {c["name"]: c for c in protocol["configs"]}
        for name in old_configs.keys() & new_configs.keys():
            if old_configs[name] != new_configs[name]:
                raise SystemExit(f"Configuration {name} changed; choose a new --output-dir.")

        configs_all = list(old_configs.values()) + [
            c for c in configs if c["name"] not in old_configs
        ]

        protocol["seeds"] = sorted(set(previous["seeds"]) | set(args.seeds))
        protocol["configs"] = configs_all
    else:
        configs_all = configs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    job_records = {job["run_dir"]: job for job in old.get("jobs", [])}
    for dataset, config, seed, run_dir, command in jobs:
        job_records[str(run_dir)] = {
            "dataset": dataset, "optimizer": config["optimizer"], "config_name": config["name"],
            "seed": seed, "run_dir": str(run_dir), "command": command,
        }
    manifest = {
        "run_id": args.output_dir.name,
        "started_at": old.get("started_at", utc_now()),
        "protocol": protocol,
        "jobs": list(job_records.values()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    for index, (dataset, config, seed, run_dir, command) in enumerate(jobs, start=1):
        print(f"[{index}/{len(jobs)}] {dataset} | {config['name']} | seed={seed}", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, env=ENV, check=True)
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary.get("status") != "completed":
            raise RuntimeError(f"Run did not complete: {run_dir}: {summary.get('reason')}")
        all_jobs = completed_jobs(args.output_dir, configs_all)
        rows = [flatten_summary(json.loads((path / "summary.json").read_text()), ds, cfg, path)
                for ds, cfg, _, path, _ in all_jobs]
        write_csv(args.output_dir / "runs.csv", rows)

    n_jobs = plot_results(args.output_dir, configs_all, no_plot=args.no_plot)
    manifest.update(finished_at=utc_now(), n_jobs=n_jobs, n_failures=0)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"COMPLETED runner group: {args.output_dir}")


if __name__ == "__main__":
    main()
