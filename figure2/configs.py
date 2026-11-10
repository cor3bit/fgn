from common import artifact_root


METHOD_ORDER_2 = [
    "adam",
    "sgn",
    "fgn",
]

METHOD_LABELS_2 = {
    "adam": "Adam (reference)",
    "sgn": "SGN",
    "fgn": "FGN",
}

METHOD_IMPL_2 = {
    "adam": "adam",
    "sgn": "sgn_ce",
    "fgn": "fgn_ce",
}

FEATURE_CACHE_DEFAULTS_2 = {
    "torchvision_weights": "IMAGENET1K_V2",
    "data_root": artifact_root() / "data",
    "extract_device": "auto",
    "extract_batch_size": 256,
    "num_workers": 4,
    "standardize": True,

    # Cars196 automatic source. TorchVision cannot download Stanford Cars anymore,
    # so make_feature_cache.py uses HuggingFace Datasets by default.
    "cars196_source": "hf",
    "cars196_hf_dataset": "tanganke/stanford_cars",
    "cars196_hf_train_split": "train",
    "cars196_hf_val_split": "test",
}

FEATURE_CACHE_DATASETS_2 = {
    "cars196": {
        "cache_path": artifact_root() / "features" / "cars196__torchvision_resnet50_imagenet1k_v2.npz",
        "n_classes": 196,
        "feature_dim": 2048,
        "encoder_name": "torchvision_resnet50_imagenet1k_v2",
        "extract_batch_size": 128,
        "split_names": {
            "train": "train",
            "val": "test",
        },
    },
}

DEFAULTS_2 = {
    "feature_cache": FEATURE_CACHE_DATASETS_2["cars196"]["cache_path"],
    "seeds": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "batch_size": 128,
    "n_epochs": 150,
    "eval_every": 100,
    "log_every": 100,
    "device": "gpu",
    "stop_on_instability": True,
    "init_scale": 0.01,
    "target_val_loss": float("nan"),
}

METHOD_KWARGS_2 = {
    "adam": {
        "learning_rates": [0.0003],
        "weight_decay": 0.0,
    },
    "sgn": {
        "learning_rates": [0.1],
        "weight_decay": 0.0,
        "lam0": 1.0,
        "cg_maxiter": 5,
        "cg_tol": 1e-5,
        "cg_stabilise_every": 0,
        "cg_warm_start": True,
    },
    "fgn": {
        "learning_rates": [0.1],
        "weight_decay": 0.0,
        "lam0": 1.0,
        "cg_maxiter": 5,
        "cg_tol": 1e-5,
        "cg_stabilise_every": 0,
        "cg_warm_start": False,
        "gram_mode": True,
    },
}

# Columns used to verify that each method has one fixed final configuration
CONFIG_COLUMNS_2 = [
    "method_id",
    "n_epochs",
    "eval_every",
    "log_every",
    "batch_size",
    "init_scale",
    "learning_rate",
    "weight_decay",
    "lam0",
    "cg_maxiter",
    "cg_tol",
    "cg_stabilise_every",
    "cg_warm_start",
    "gram_mode",
]
