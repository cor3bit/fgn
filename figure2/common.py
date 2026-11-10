import csv
import json
import os
from pathlib import Path
from typing import Any


def artifact_root() -> Path:
    return Path(os.environ.get("FGN_ARTIFACTS_DIR", Path(__file__).resolve().parents[1] / "artifacts")).expanduser().resolve()


def str2bool(x):
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {x}")


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


def _json_default(x: Any):
    try:
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass
    if isinstance(x, Path):
        return str(x)
    if hasattr(x, "tolist"):
        return x.tolist()
    raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=_json_default)
        f.write("\n")


def read_json_string(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "shape") and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str) and value.strip():
        return json.loads(value)
    return {}


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
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": 2.0,
        "lines.markersize": 4.2,
        "grid.alpha": 0.20,
        "grid.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
