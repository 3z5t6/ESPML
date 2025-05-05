"""
Centralised project-wide constants for the ESPML repository.

This module should **only** hold simple, immutable values (paths, numbers,
static dictionaries).  
Avoid在此处写入任何会在导入时执行重量级计算或 I/O 的逻辑，
确保其他包导入本文件时不会产生副作用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Final

# --------------------------------------------------------------------------- #
#                         Path-related constants                               #
# --------------------------------------------------------------------------- #

# Project root = two levels up from this file:  module/utils/const.py -> module/utils -> module -> <ROOT>
ROOT_DIR:     Final[Path] = Path(__file__).resolve().parents[2]
CONFIG_DIR:   Final[Path] = ROOT_DIR / "config"
DATA_DIR:     Final[Path] = ROOT_DIR / "data"
OUTPUT_DIR:   Final[Path] = ROOT_DIR / "output"

MODEL_DIR:    Final[Path] = OUTPUT_DIR / "models"
REPORT_DIR:   Final[Path] = OUTPUT_DIR / "reports"
CACHE_DIR:    Final[Path] = OUTPUT_DIR / "cache"

# 按需创建目录（轻量级操作，确保下游代码无需显式判断）
for _p in (MODEL_DIR, REPORT_DIR, CACHE_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
#                           General constants                                  #
# --------------------------------------------------------------------------- #

# 全局随机种子
RANDOM_SEED: Final[int] = 42

# 日志格式 & 级别
LOG_FORMAT:   Final[str] = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
LOG_LEVEL:    Final[str] = os.getenv("ESPML_LOG_LEVEL", "INFO").upper()

# 数据列名
TIMESTAMP_COL: Final[str] = "timestamp"
TARGET_COL:    Final[str] = "target"

# --------------------------------------------------------------------------- #
#                       Dataset-related parameters                             #
# --------------------------------------------------------------------------- #

DATASET_PARAMS: Dict[str, Any] = {
    "test_size":      0.2,          # fraction of data used as test set
    "val_size":       0.1,          # fraction of training data used as validation
    "shuffle":        True,
    "random_state":   RANDOM_SEED,
    "categorical_na": "missing",    # placeholder for missing categorical values
    "numeric_na":     0.0,          # placeholder for missing numeric values
}

# --------------------------------------------------------------------------- #
#                       Training-related parameters                            #
# --------------------------------------------------------------------------- #

TRAINING_PARAMS: Dict[str, Any] = {
    "epochs":                   50,
    "batch_size":               256,
    "early_stopping_rounds":    10,
    "learning_rate_schedule":   "plateau",
    "num_workers":              os.cpu_count() or 4,
    "verbose":                  1,
}

# --------------------------------------------------------------------------- #
#                        Model-specific parameters                             #
# --------------------------------------------------------------------------- #

MODEL_PARAMS_XGB: Dict[str, Any] = {
    "n_estimators":        400,
    "learning_rate":       0.05,
    "max_depth":           6,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "gamma":               0.0,
    "reg_lambda":          1.0,
    "random_state":        RANDOM_SEED,
    "n_jobs":              os.cpu_count() or 4,
}

MODEL_PARAMS_LGBM: Dict[str, Any] = {
    "num_leaves":          31,
    "learning_rate":       0.05,
    "n_estimators":        500,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "random_state":        RANDOM_SEED,
}


# --------------------------------------------------------------------------- #
#                          Public export control                               #
# --------------------------------------------------------------------------- #

__all__ = [
    "ROOT_DIR",
    "CONFIG_DIR",
    "DATA_DIR",
    "OUTPUT_DIR",
    "MODEL_DIR",
    "REPORT_DIR",
    "CACHE_DIR",
    "RANDOM_SEED",
    "LOG_FORMAT",
    "LOG_LEVEL",
    "TIMESTAMP_COL",
    "TARGET_COL",
    "DATASET_PARAMS",
    "TRAINING_PARAMS",
    "MODEL_PARAMS_XGB",
    "MODEL_PARAMS_LGBM",
]