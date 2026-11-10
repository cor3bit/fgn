import csv
import json
import os
import platform
from pathlib import Path
from typing import Any


def artifact_root() -> Path:
    base = Path(os.environ.get("FGN_ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts"))
    return base.expanduser().resolve() / "figure1"


def str2bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {x}")


def append_csv_row(path: Path, row: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_csv_rows(path: Path, rows: list[dict[str, Any]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(path: Path, rows: list[dict[str, Any]]):
    """Overwrite ``path`` with deterministic rows."""
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _json_default(x: Any):
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def get_hardware_info():
    import jax

    cpu = platform.processor() or platform.machine()
    cpu_info = f"{cpu}, logical cores: {os.cpu_count()}"
    try:
        ram_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        ram_info = f"{ram_bytes / 1024 ** 3:.1f} GB"
    except (AttributeError, OSError, ValueError):
        ram_info = "Unknown"
    devices = ", ".join(f"{d.platform}: {d.device_kind}" for d in jax.devices())
    return platform.node(), cpu_info, ram_info, devices


def is_finite_tree(tree) -> bool:
    import jax
    import jax.numpy as jnp

    for leaf in jax.tree_util.tree_leaves(tree):
        if not bool(jnp.all(jnp.isfinite(leaf))):
            return False
    return True


def setup_plot_style():
    import matplotlib.pyplot as plt

    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.titlesize": 12.5,
        "axes.titleweight": "regular",
        "axes.labelsize": 10.5,
        "legend.fontsize": 8.6,
        "xtick.labelsize": 9.2,
        "ytick.labelsize": 9.2,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.0,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def safe_mean(series):
    import numpy as np
    import pandas as pd

    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)) if len(vals) else float("nan")


def safe_std(series):
    import numpy as np
    import pandas as pd

    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.std(vals, ddof=0)) if len(vals) else float("nan")


def safe_sem(series):
    import numpy as np
    import pandas as pd

    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(np.std(vals, ddof=0) / np.sqrt(len(vals))) if len(vals) else float("nan")
