#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6+

import os
import pickle
from typing import Any, Dict


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
        return pickle.load(f)


