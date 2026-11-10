import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from common import artifact_root, str2bool
from configs import (
    FEATURE_CACHE_DATASETS_2,
    FEATURE_CACHE_DEFAULTS_2,
)


# CLI


def _default_cache_path(dataset_id: str) -> Path:
    try:
        return Path(FEATURE_CACHE_DATASETS_2[dataset_id]["cache_path"])
    except KeyError as exc:
        raise ValueError(
            f"No default feature-cache path configured for dataset_id={dataset_id!r}"
        ) from exc


def _dataset_default(dataset_id: str, key: str, fallback: Any = None) -> Any:
    return FEATURE_CACHE_DATASETS_2.get(dataset_id, {}).get(key, fallback)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Create the Cars196 frozen-feature cache. Extraction path: "
            "image dataset -> TorchVision ImageNet-pretrained ResNet-50 -> .npz. "
            "Cars196 defaults to a HuggingFace Datasets mirror because "
            "TorchVision cannot download Stanford Cars anymore. "
        )
    )

    p.add_argument(
        "--dataset-id",
        type=str,
        default="cars196",
        choices=sorted(FEATURE_CACHE_DATASETS_2.keys()),
    )
    p.add_argument(
        "--out-cache",
        type=Path,
        default=None,
        help="Output .npz path. If omitted, use the configured default for --dataset-id.",
    )

    # Real feature extraction knobs.
    p.add_argument(
        "--torchvision-weights",
        type=str,
        default=FEATURE_CACHE_DEFAULTS_2["torchvision_weights"],
        help="TorchVision ResNet50 weights enum name, e.g. DEFAULT, IMAGENET1K_V2, IMAGENET1K_V1.",
    )
    p.add_argument("--data-root", type=Path, default=Path(FEATURE_CACHE_DEFAULTS_2["data_root"]))
    p.add_argument("--extract-batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=FEATURE_CACHE_DEFAULTS_2.get("num_workers", 4))
    p.add_argument(
        "--extract-device",
        type=str,
        default=FEATURE_CACHE_DEFAULTS_2.get("extract_device", "auto"),
        choices=["cuda", "gpu", "cpu", "auto"],
        help="Device for offline feature extraction only. The timed head benchmark is separate.",
    )
    p.add_argument(
        "--standardize",
        type=str2bool,
        default=FEATURE_CACHE_DEFAULTS_2.get("standardize", True),
        nargs="?",
        const=True,
        help="Apply train-set mean/std standardization to cached embeddings.",
    )
    p.add_argument(
        "--overwrite",
        type=str2bool,
        default=False,
        nargs="?",
        const=True,
        help="Overwrite an existing cache file.",
    )
    p.add_argument("--limit-train", type=int, default=-1)
    p.add_argument("--limit-val", type=int, default=-1)

    # Cars196-specific data source.
    p.add_argument(
        "--cars196-source",
        type=str,
        default=FEATURE_CACHE_DEFAULTS_2.get("cars196_source", "hf"),
        choices=["hf", "torchvision-local"],
        help=(
            "Cars196 source. 'hf' downloads from HuggingFace Datasets; "
            "'torchvision-local' reads a manually installed TorchVision StanfordCars tree."
        ),
    )
    p.add_argument(
        "--cars196-hf-dataset",
        type=str,
        default=FEATURE_CACHE_DEFAULTS_2.get("cars196_hf_dataset", "tanganke/stanford_cars"),
        help="HuggingFace Datasets id used when --dataset-id cars196 --cars196-source hf.",
    )
    p.add_argument(
        "--cars196-hf-train-split",
        type=str,
        default=FEATURE_CACHE_DEFAULTS_2.get("cars196_hf_train_split", "train"),
    )
    p.add_argument(
        "--cars196-hf-val-split",
        type=str,
        default=FEATURE_CACHE_DEFAULTS_2.get("cars196_hf_val_split", "test"),
    )
    p.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=artifact_root() / "cache" / "huggingface" / "datasets",
        help="HuggingFace Datasets cache directory for Cars196.",
    )

    args = p.parse_args()

    if args.out_cache is None:
        args.out_cache = _default_cache_path(args.dataset_id)

    if args.extract_batch_size is None:
        args.extract_batch_size = int(
            _dataset_default(
                args.dataset_id,
                "extract_batch_size",
                FEATURE_CACHE_DEFAULTS_2.get("extract_batch_size", 256),
            )
        )

    if args.extract_batch_size <= 0:
        raise ValueError("--extract-batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.limit_train == 0 or args.limit_train < -1:
        raise ValueError("--limit-train must be -1 or positive")
    if args.limit_val == 0 or args.limit_val < -1:
        raise ValueError("--limit-val must be -1 or positive")

    if args.out_cache.suffix.lower() != ".npz":
        raise ValueError("--out-cache must end in .npz")

    if args.out_cache.exists() and not bool(args.overwrite):
        raise FileExistsError(
            f"Feature cache already exists: {args.out_cache}. "
            f"Pass --overwrite true to replace it, or choose another --out-cache."
        )

    return args


# Shared cache helpers


def _normalise_features(x_train: np.ndarray, x_val: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6).astype(np.float32)

    x_train_n = (x_train - mean) / std
    x_val_n = (x_val - mean) / std

    stats = {
        "feature_standardization": "train_mean_std",
        "feature_mean_l2": float(np.linalg.norm(mean.reshape(-1))),
        "feature_std_min": float(std.min()),
        "feature_std_max": float(std.max()),
    }
    return x_train_n.astype(np.float32), x_val_n.astype(np.float32), stats


def _save_cache(
        path: Path,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        metadata: dict[str, Any],
):
    path.parent.mkdir(parents=True, exist_ok=True)

    if x_train.ndim != 2 or x_val.ndim != 2:
        raise ValueError(f"Expected 2D feature arrays, got train={x_train.shape}, val={x_val.shape}")
    if x_train.shape[1] != x_val.shape[1]:
        raise ValueError(f"Feature dims differ: train={x_train.shape[1]}, val={x_val.shape[1]}")
    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError("x_train/y_train size mismatch")
    if x_val.shape[0] != y_val.shape[0]:
        raise ValueError("x_val/y_val size mismatch")

    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        temporary,
        x_train=np.asarray(x_train, dtype=np.float32),
        y_train=np.asarray(y_train, dtype=np.int32),
        x_val=np.asarray(x_val, dtype=np.float32),
        y_val=np.asarray(y_val, dtype=np.int32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    temporary.replace(path)


def _maybe_limit_dataset(dataset, limit: int):
    if limit is None or int(limit) < 0:
        return dataset

    try:
        from torch.utils.data import Subset
    except ImportError as exc:  # pragma: no cover
        raise ImportError("torch is required for real feature extraction") from exc

    n = min(int(limit), len(dataset))
    return Subset(dataset, list(range(n)))


def _progress(iterable, *, total: int | None, desc: str):
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


# TorchVision feature extraction


def _import_torchvision_stack():
    try:
        import torch
        import torchvision
        from torch.utils.data import DataLoader
        from torchvision import datasets
        from torchvision.models import ResNet50_Weights, resnet50
    except ImportError as exc:
        raise ImportError(
            "Real feature extraction needs torch and torchvision. "
            "The timed benchmark itself remains JAX/Somax and consumes only the resulting .npz."
        ) from exc

    return torch, torchvision, DataLoader, datasets, ResNet50_Weights, resnet50


def _select_device(torch, requested: str):
    requested = "cuda" if requested == "gpu" else str(requested)

    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--extract-device cuda/gpu requested, but torch.cuda.is_available() is false. "
                "Use --extract-device cpu or fix CUDA/PyTorch."
            )
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raise ValueError(f"Unknown extract device: {requested}")


def _resolve_resnet50_weights(weights_arg: str, ResNet50_Weights):
    key = str(weights_arg).strip()
    if key.upper() == "DEFAULT":
        return ResNet50_Weights.DEFAULT

    try:
        return ResNet50_Weights[key.upper()]
    except KeyError as exc:
        valid = ["DEFAULT"] + [w.name for w in ResNet50_Weights]
        raise ValueError(f"Unknown ResNet50 weights={weights_arg!r}. Valid values: {valid}") from exc


def _make_resnet50_feature_extractor(args):
    torch, torchvision, _, _, ResNet50_Weights, resnet50 = _import_torchvision_stack()

    weights = _resolve_resnet50_weights(args.torchvision_weights, ResNet50_Weights)
    model = resnet50(weights=weights)

    # Export the pooled pre-classifier representation.
    # Original ImageNet head would be Linear(2048, 1000).
    feature_dim = int(model.fc.in_features)
    model.fc = torch.nn.Identity()

    transform = weights.transforms()
    weights_name = weights.name
    encoder_name = f"torchvision_resnet50_{weights_name.lower()}"

    return model, transform, feature_dim, weights_name, encoder_name, torchvision.__version__


# Dataset sources


class _HFDatasetImageClassification:
    """Minimal torch Dataset wrapper around a HuggingFace image-classification dataset."""

    def __init__(self, ds, transform, *, image_col: str, label_col: str):
        self.ds = ds
        self.transform = transform
        self.image_col = image_col
        self.label_col = label_col
        self.label_feature = getattr(ds, "features", {}).get(label_col)

    def __len__(self):
        return len(self.ds)

    def _label_to_int(self, label) -> int:
        if isinstance(label, (np.integer, int)):
            return int(label)
        if hasattr(label, "item"):
            return int(label.item())
        if isinstance(label, str) and hasattr(self.label_feature, "str2int"):
            return int(self.label_feature.str2int(label))
        return int(label)

    def __getitem__(self, idx: int):
        ex = self.ds[int(idx)]
        image = ex[self.image_col]
        label = self._label_to_int(ex[self.label_col])

        # HF image columns usually return PIL Images. Enforce RGB to match ImageNet transforms.
        if hasattr(image, "convert"):
            image = image.convert("RGB")

        image = self.transform(image)
        return image, label


def _infer_hf_columns(ds) -> tuple[str, str]:
    features = getattr(ds, "features", {})
    keys = list(features.keys()) if features else list(ds[0].keys())

    image_candidates = ["image", "img", "jpeg", "jpg"]
    label_candidates = ["label", "labels", "fine_label", "class", "target"]

    image_col = next((c for c in image_candidates if c in keys), None)
    label_col = next((c for c in label_candidates if c in keys), None)

    if image_col is None:
        raise ValueError(f"Could not infer HF image column. Available columns: {keys}")
    if label_col is None:
        raise ValueError(f"Could not infer HF label column. Available columns: {keys}")

    return image_col, label_col


def _load_cars196_hf(args, transform):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Cars196 automatic download uses HuggingFace Datasets. Install it in the "
            "feature-extractor venv with: pip install datasets"
        ) from exc

    cache_dir = str(args.hf_cache_dir)

    logging.info(
        "Loading Cars196 from HuggingFace Datasets | dataset=%s | train_split=%s | val_split=%s",
        args.cars196_hf_dataset,
        args.cars196_hf_train_split,
        args.cars196_hf_val_split,
    )

    train_hf = load_dataset(
        str(args.cars196_hf_dataset),
        split=str(args.cars196_hf_train_split),
        cache_dir=cache_dir,
    )
    val_hf = load_dataset(
        str(args.cars196_hf_dataset),
        split=str(args.cars196_hf_val_split),
        cache_dir=cache_dir,
    )

    image_col, label_col = _infer_hf_columns(train_hf)
    val_image_col, val_label_col = _infer_hf_columns(val_hf)
    if (image_col, label_col) != (val_image_col, val_label_col):
        raise ValueError(
            "HF train/val column mismatch: "
            f"train={(image_col, label_col)}, val={(val_image_col, val_label_col)}"
        )

    train_classes = getattr(train_hf.features[label_col], "names", None)
    val_classes = getattr(val_hf.features[val_label_col], "names", None)
    if train_classes is not None and val_classes is not None and train_classes != val_classes:
        raise ValueError("HF train/test class-label order differs")

    train_ds = _HFDatasetImageClassification(
        train_hf,
        transform,
        image_col=image_col,
        label_col=label_col,
    )
    val_ds = _HFDatasetImageClassification(
        val_hf,
        transform,
        image_col=image_col,
        label_col=label_col,
    )

    return train_ds, val_ds, "hf_datasets"


def _load_cars196_torchvision_local(root: Path, transform):
    _, _, _, datasets, _, _ = _import_torchvision_stack()

    try:
        train_ds = datasets.StanfordCars(
            root=str(root),
            split="train",
            transform=transform,
            download=False,
        )
        val_ds = datasets.StanfordCars(
            root=str(root),
            split="test",
            transform=transform,
            download=False,
        )
    except Exception as exc:
        raise RuntimeError(
            "TorchVision local StanfordCars was requested, but the dataset was not found "
            "in the expected local layout. Either use the default --cars196-source hf, or "
            "manually place the dataset under:\n"
            f"  {root / 'stanford_cars'}\n"
            "with devkit/, cars_train/, cars_test/, and cars_test_annos_withlabels.mat, "
            "then rerun with --cars196-source torchvision-local."
        ) from exc

    return train_ds, val_ds, "torchvision_local"


def _load_dataset(args, root: Path, transform):
    cfg = FEATURE_CACHE_DATASETS_2[str(args.dataset_id)]
    dataset_id = str(args.dataset_id)

    if dataset_id == "cars196":
        if args.cars196_source == "hf":
            train_ds, val_ds, dataset_source = _load_cars196_hf(args, transform)
        elif args.cars196_source == "torchvision-local":
            train_ds, val_ds, dataset_source = _load_cars196_torchvision_local(root, transform)
        else:
            raise ValueError(f"Unknown Cars196 source: {args.cars196_source}")

    else:
        raise ValueError(f"Unsupported real dataset_id={dataset_id!r}")

    return train_ds, val_ds, int(cfg["n_classes"]), dict(cfg["split_names"]), dataset_source


# Extraction


def _make_loader(dataset, batch_size: int, num_workers: int, device):
    _, _, DataLoader, _, _, _ = _import_torchvision_stack()

    kwargs = {
        "batch_size": int(batch_size),
        "shuffle": False,
        "drop_last": False,
        "num_workers": int(num_workers),
        "pin_memory": bool(device.type == "cuda"),
    }
    if int(num_workers) > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2

    return DataLoader(dataset, **kwargs)


def _extract_split(
        *,
        model,
        dataset,
        batch_size: int,
        num_workers: int,
        device,
        desc: str,
) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _, _, _, _ = _import_torchvision_stack()

    loader = _make_loader(dataset, batch_size=batch_size, num_workers=num_workers, device=device)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    model.eval()
    with torch.inference_mode():
        for images, labels in _progress(loader, total=len(loader), desc=desc):
            images = images.to(device, non_blocking=True)
            feats = model(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            feats = torch.flatten(feats, start_dim=1)
            xs.append(feats.detach().to("cpu", dtype=torch.float32).numpy())
            ys.append(labels.detach().to("cpu").numpy().astype(np.int32))

    x = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0).astype(np.int32)

    if x.ndim != 2:
        raise ValueError(f"Extractor returned non-2D features: shape={x.shape}")

    return x, y


def _make_real_image_cache(args) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    torch, _, _, _, _, _ = _import_torchvision_stack()

    model, transform, feature_dim, weights_name, encoder_name, torchvision_version = _make_resnet50_feature_extractor(
        args)
    device = _select_device(torch, str(args.extract_device))
    model = model.to(device)

    train_ds, val_ds, n_classes, split_names, dataset_source = _load_dataset(
        args=args,
        root=Path(args.data_root),
        transform=transform,
    )

    source_provenance = {}
    if dataset_source == "hf_datasets":
        from importlib.metadata import version

        source_provenance = {
            "datasets_version": version("datasets"),
            "hf_train_fingerprint": getattr(train_ds.ds, "_fingerprint", None),
            "hf_test_fingerprint": getattr(val_ds.ds, "_fingerprint", None),
        }
    train_ds = _maybe_limit_dataset(train_ds, int(args.limit_train))
    val_ds = _maybe_limit_dataset(val_ds, int(args.limit_val))

    logging.info(
        "Extracting frozen features | dataset=%s | source=%s | encoder=%s | weights=%s | device=%s | "
        "batch=%d | train=%d | val=%d",
        args.dataset_id,
        dataset_source,
        encoder_name,
        weights_name,
        device,
        int(args.extract_batch_size),
        len(train_ds),
        len(val_ds),
    )

    t0 = time.perf_counter()
    x_train, y_train = _extract_split(
        model=model,
        dataset=train_ds,
        batch_size=int(args.extract_batch_size),
        num_workers=int(args.num_workers),
        device=device,
        desc=f"{args.dataset_id}/train",
    )
    x_val, y_val = _extract_split(
        model=model,
        dataset=val_ds,
        batch_size=int(args.extract_batch_size),
        num_workers=int(args.num_workers),
        device=device,
        desc=f"{args.dataset_id}/val",
    )
    extraction_time_s = time.perf_counter() - t0

    if args.limit_train < 0 and len(y_train) != 8144:
        raise ValueError(f"Expected all 8144 Cars196 training examples, got {len(y_train)}")
    if args.limit_val < 0 and len(y_val) != 8041:
        raise ValueError(f"Expected all 8041 Cars196 test examples, got {len(y_val)}")

    if x_train.shape[1] != feature_dim:
        raise ValueError(f"Expected feature_dim={feature_dim}, got {x_train.shape[1]}")
    if int(np.min(y_train)) < 0 or int(np.min(y_val)) < 0:
        raise ValueError("Observed negative labels")
    if int(np.max(y_train)) >= n_classes or int(np.max(y_val)) >= n_classes:
        raise ValueError(
            f"Observed labels exceed configured number of classes {n_classes}: "
            f"train max={int(np.max(y_train))}, val max={int(np.max(y_val))}"
        )

    standardize_meta: dict[str, Any] = {"feature_standardization": "none"}
    if bool(args.standardize):
        x_train, x_val, standardize_meta = _normalise_features(x_train, x_val)

    metadata = {
        "dataset_id": str(args.dataset_id),
        "dataset_source": str(dataset_source),
        "encoder_name": encoder_name,
        "encoder_backend": "torchvision",
        "architecture": "resnet50",
        "weights": str(weights_name),
        "exported_layer": "avgpool_flatten_pre_classifier",
        "preprocessing": repr(transform),
        "feature_dim": int(x_train.shape[1]),
        "n_classes": int(n_classes),
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "split_names": split_names,
        "cache_kind": "frozen_image_encoder_features",
        "timing_note": "Feature extraction is excluded from head-optimization timing.",
        "extract_batch_size": int(args.extract_batch_size),
        "extract_device": str(device),
        "num_workers": int(args.num_workers),
        "download": args.cars196_source == "hf",
        "limit_train": int(args.limit_train),
        "limit_val": int(args.limit_val),
        "extraction_time_s": float(extraction_time_s),
        "torch_version": str(torch.__version__),
        "torchvision_version": str(torchvision_version),
        "python_version": platform.python_version(),
    }
    if str(args.dataset_id) == "cars196":
        metadata.update({
            "cars196_source": str(args.cars196_source),
            "cars196_hf_dataset": str(args.cars196_hf_dataset) if args.cars196_source == "hf" else None,
            "cars196_hf_train_split": str(args.cars196_hf_train_split) if args.cars196_source == "hf" else None,
            "cars196_hf_val_split": str(args.cars196_hf_val_split) if args.cars196_source == "hf" else None,
        })

    metadata.update(standardize_meta)
    metadata.update(source_provenance)
    return x_train, y_train, x_val, y_val, metadata


# Main


def main():
    args = parse_args()
    cache_root = artifact_root() / "cache"
    os.environ.setdefault("TORCH_HOME", str(cache_root / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_root / "huggingface"))
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    x_train, y_train, x_val, y_val, metadata = _make_real_image_cache(args)

    _save_cache(args.out_cache, x_train, y_train, x_val, y_val, metadata)

    logging.info("Saved feature cache to %s", args.out_cache)
    logging.info(
        "cache | dataset=%s | source=%s | encoder=%s | backend=%s | train=%s | val=%s | classes=%d | standardization=%s",
        metadata["dataset_id"],
        metadata["dataset_source"],
        metadata["encoder_name"],
        metadata.get("encoder_backend", "unknown"),
        tuple(x_train.shape),
        tuple(x_val.shape),
        int(metadata["n_classes"]),
        metadata.get("feature_standardization", "unknown"),
    )


if __name__ == "__main__":
    main()
