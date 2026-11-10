CLASS_COUNTS_1A = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

SEEDS_1A = [1, 2, 3, 4, 5]

METHOD_ORDER_1A = [
    "adam_ref",
    "fgn",
    "sgn",
]

METHOD_LABELS_1A = {
    "adam_ref": "Adam (reference)",
    "fgn": "FGN",
    "sgn": "SGN",
}

METHOD_IMPL_1A = {
    "adam_ref": "adam",
    "fgn": "fgn_ce",
    "sgn": "sgn_ce",
}

DEFAULTS_1A = {
    "feat_dim": 2048,
    "batch_size": 512,
    "n_samples": 32768,
    "data_seed_base": 1337,
    "warmup_steps": 128,
    "timed_steps": 640,
    "measure_every": 64,
    "device": "gpu",
    "stop_on_instability": True,
}

METHOD_KWARGS_1A = {
    "adam_ref": {
        "learning_rate": 1e-3,
    },
    "fgn": {
        "learning_rate": 1e-2,
        "lam0": 1.0,
        "cg_maxiter": 5,
        "cg_tol": 1e-5,
        "cg_stabilise_every": 0,
        "cg_warm_start": False,
        "cg_lifted": False,
    },
    "sgn": {
        "learning_rate": 1e-2,
        "lam0": 1.0,
        "cg_maxiter": 5,
        "cg_tol": 1e-5,
        "cg_stabilise_every": 0,
        "cg_warm_start": False,
    },
}

DEFAULTS_1B = {
    "n_classes": 10,
    "p_star": 0.6,
    "n_points": 201,
}

DEFAULTS_1C = {
    "n_classes": 10,
    "p_star": 0.7,
    "n_points": 201,
    "alpha_min": 1e-5,
    "eta": 1.25,
    "beta": 3.0,
    "lambda_regime_scales": {
        "low damping": 0.01,
        "medium damping": 0.10,
        "high damping": 1.00,
    },
}
