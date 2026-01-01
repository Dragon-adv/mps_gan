#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os
import time
import pickle
import random
from typing import Any, Dict, Optional

import numpy as np
import torch


def _ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def get_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: Dict[str, Any]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def export_component_state_dicts(model: torch.nn.Module) -> Optional[Dict[str, Dict[str, Any]]]:
    """
    Export 4-component state_dicts for a given model.
    Returns None if model does not provide a component export interface.
    """
    if hasattr(model, "get_component_state_dicts") and callable(getattr(model, "get_component_state_dicts")):
        return model.get_component_state_dicts()
    return None


def _fmt_acc(x: float) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "nan"


def save_latest(
    ckpt_dir: str,
    payload: Dict[str, Any],
    filename: str = "latest.pt",
) -> str:
    _ensure_dir(ckpt_dir)
    path = os.path.join(ckpt_dir, filename)
    torch.save(payload, path)
    return path


def save_best(
    ckpt_dir: str,
    metric_type: str,
    round_idx: int,
    mean_acc_wo: float,
    mean_acc_wp: float,
    payload: Dict[str, Any],
    overwrite: bool = True,
) -> str:
    """
    Save best checkpoint.
    metric_type: 'best-wo' | 'best-wp'

    If overwrite=True, always save to a fixed filename: <metric_type>.pt
    (e.g., best-wo.pt / best-wp.pt), so only one checkpoint is kept per metric.

    Note: round/metrics should be read from payload['meta'] instead of filename.
    """
    _ensure_dir(ckpt_dir)
    if overwrite:
        fname = f"{metric_type}.pt"
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        fname = (
            f"r{round_idx:04d}__{metric_type}"
            f"__wo{_fmt_acc(mean_acc_wo)}__wp{_fmt_acc(mean_acc_wp)}__{ts}.pt"
        )
    path = os.path.join(ckpt_dir, fname)
    torch.save(payload, path)
    return path


def load_checkpoint(path: str, map_location: Optional[str] = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location)


def save_pickle(path: str, obj: Any) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


