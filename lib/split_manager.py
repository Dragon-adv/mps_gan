#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os
import pickle
from typing import Any, Dict

import numpy as np


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def save_split(split_path: str, split: Dict[str, Any]) -> None:
    """
    Persist dataset split information for reproducible multi-stage experiments.
    Expected keys: n_list, k_list, user_groups, user_groups_lt, classes_list, classes_list_gt (optional), meta (optional).
    """
    _ensure_dir_for_file(split_path)
    with open(split_path, "wb") as f:
        pickle.dump(split, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_split(split_path: str) -> Dict[str, Any]:
    with open(split_path, "rb") as f:
        split = pickle.load(f)

    # Normalize indices to plain Python ints to avoid issues like:
    #   IndexError: only integers, slices (...) are valid indices
    # when using torchvision datasets + torch.utils.data.Subset.
    def _to_int_list(v: Any) -> list:
        if v is None:
            return []
        # torch Tensor -> numpy
        if hasattr(v, "detach") and hasattr(v, "cpu") and hasattr(v, "numpy"):
            v = v.detach().cpu().numpy()
        # numpy array / list / set / tuple
        arr = np.asarray(list(v) if isinstance(v, set) else v)
        if arr.size == 0:
            return []
        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(np.isfinite(arr)):
                raise ValueError("split indices contain non-finite floats")
            if not np.allclose(arr, np.round(arr)):
                raise ValueError("split indices contain non-integer floats")
            arr = np.round(arr)
        # At this point, cast to int64 then Python int
        arr = arr.astype(np.int64, copy=False)
        return [int(x) for x in arr.tolist()]

    def _normalize_groups(groups: Any) -> Any:
        if isinstance(groups, dict):
            out: Dict[int, Any] = {}
            for k, v in groups.items():
                out[int(k)] = _to_int_list(v)
            return out
        # Some code paths may store as list indexed by client id
        if isinstance(groups, (list, tuple)):
            return [_to_int_list(v) for v in groups]
        return groups

    if isinstance(split, dict):
        if "user_groups" in split:
            split["user_groups"] = _normalize_groups(split["user_groups"])
        if "user_groups_lt" in split:
            split["user_groups_lt"] = _normalize_groups(split["user_groups_lt"])

    return split


