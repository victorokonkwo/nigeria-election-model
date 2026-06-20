"""Spatial weight matrix construction and transformations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.sparse as sp
import yaml
from loguru import logger

from ngforecast.data.geography import (
    build_adjacency_matrix,
    build_zone_block_matrix,
    load_constitution_units,
    row_standardize,
)


def build_W(config_path: Path = Path("config/model.yaml")) -> np.ndarray:
    """Build the spatial weight matrix W per model config.

    Supports: contiguity, zone_block, hybrid (average of both).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    w_type = cfg["spatial"]["W_type"]
    do_standardize = cfg["spatial"]["row_standardize"]

    if w_type == "contiguity":
        adj_file = Path("data/external/state_adjacency.csv")
        W = build_adjacency_matrix(adj_file)
    elif w_type == "zone_block":
        W = build_zone_block_matrix()
    elif w_type == "hybrid":
        adj_file = Path("data/external/state_adjacency.csv")
        W_cont = build_adjacency_matrix(adj_file)
        W_zone = build_zone_block_matrix()
        W = 0.5 * W_cont + 0.5 * W_zone
    else:
        raise ValueError(f"Unknown W_type: {w_type}")

    if do_standardize:
        W = row_standardize(W)

    logger.info("W matrix built (type={}): shape={}", w_type, W.shape)
    return W


def save_W(W: np.ndarray, dest: Path) -> None:
    """Save weight matrix as sparse .npz."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(dest, sp.csr_matrix(W))
    logger.info("W saved to {}", dest)


def load_W(path: Path) -> np.ndarray:
    """Load weight matrix from .npz."""
    return sp.load_npz(path).toarray()


def get_state_codes() -> list[str]:
    """Return ordered list of state codes matching W rows/cols."""
    return [u["code"] for u in load_constitution_units()]
