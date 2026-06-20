"""Geography module — builds adjacency matrix and six-zone weight matrix W."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from loguru import logger


def load_constitution_units() -> list[dict]:
    """Load the 37 geographic units from constitution.yaml."""
    with open("config/constitution.yaml") as f:
        cfg = yaml.safe_load(f)
    return cfg["units"]


def build_zone_map() -> dict[str, str]:
    """Return {state_code: zone} mapping."""
    units = load_constitution_units()
    return {u["code"]: u["zone"] for u in units}


def build_adjacency_matrix(adjacency_file: Path) -> np.ndarray:
    """Build binary contiguity matrix from an external adjacency list.

    The adjacency file should be CSV with columns: state_a, state_b.
    """
    units = load_constitution_units()
    codes = [u["code"] for u in units]
    n = len(codes)
    idx = {c: i for i, c in enumerate(codes)}

    W = np.zeros((n, n), dtype=np.float64)
    adj = pd.read_csv(adjacency_file)

    for _, row in adj.iterrows():
        a, b = row["state_a"], row["state_b"]
        if a in idx and b in idx:
            W[idx[a], idx[b]] = 1.0
            W[idx[b], idx[a]] = 1.0

    logger.info("Built contiguity W: {} units, {} edges", n, int(W.sum() / 2))
    return W


def build_zone_block_matrix() -> np.ndarray:
    """Build zone-block weight matrix: W[i,j]=1 iff same zone, i≠j."""
    units = load_constitution_units()
    n = len(units)
    zones = [u["zone"] for u in units]

    W = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j and zones[i] == zones[j]:
                W[i, j] = 1.0

    logger.info("Built zone-block W: {} units", n)
    return W


def row_standardize(W: np.ndarray) -> np.ndarray:
    """Row-standardize a weight matrix (each row sums to 1)."""
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0  # avoid division by zero for isolates
    return W / row_sums
