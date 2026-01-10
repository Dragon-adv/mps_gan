from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import torch


def ensure_repo_root_on_path() -> str:
    """
    Ensure project root is on sys.path when running scripts directly.
    Returns the resolved repo root path.
    """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def resolve_device(gpu: int | None = None) -> str:
    if torch.cuda.is_available():
        if gpu is not None:
            torch.cuda.set_device(int(gpu))
        return "cuda"
    return "cpu"


def seed_all(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ns_from_dict(d: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**(d or {}))


def as_int(x) -> int:
    if isinstance(x, (np.integer, int)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if not float(x).is_integer():
            raise ValueError(f"Non-integer float cannot be converted to int: {x}")
        return int(round(float(x)))
    if torch.is_tensor(x):
        if x.numel() != 1:
            raise ValueError(f"Tensor with numel!=1 cannot be converted to int: {x.shape}")
        return int(x.item())
    raise TypeError(f"Unsupported int conversion type: {type(x)}")


def normalize_split_indices(idxs) -> list[int]:
    """
    Convert split indices (possibly nested numpy arrays / floats) into List[int].
    Mirrors logic used in exps/run_stage4.py but kept minimal here.
    """
    def _flatten(v) -> list[int]:
        if v is None:
            return []
        if torch.is_tensor(v):
            v = v.detach().cpu()
            return [int(x) for x in v.flatten().tolist()]
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return []
            if v.dtype == object:
                out: list[int] = []
                for e in v.tolist():
                    out.extend(_flatten(e))
                return out
            if np.issubdtype(v.dtype, np.floating):
                if not np.all(np.isfinite(v)):
                    raise ValueError("Found non-finite indices in split")
                if not np.allclose(v, np.round(v)):
                    raise ValueError("Found non-integer float indices in split")
                v = np.round(v)
            v = v.astype(np.int64, copy=False).reshape(-1)
            return [int(x) for x in v.tolist()]
        if isinstance(v, (list, tuple, set)):
            out: list[int] = []
            for e in v:
                out.extend(_flatten(e))
            return out
        if isinstance(v, (np.integer, int, np.floating, float)) or torch.is_tensor(v):
            return [as_int(v)]
        raise TypeError(f"Unsupported index type in split: {type(v)}")

    out = _flatten(idxs)
    if any(not isinstance(i, int) for i in out):
        raise TypeError("normalize_split_indices failed to produce plain ints.")
    return out


@torch.no_grad()
def eval_model_on_images(
    model: torch.nn.Module,
    dl: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
) -> Tuple[float, float]:
    import torch.nn.functional as F

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for images, labels in dl:
        images = images.to(device)
        labels = labels.to(device).long()
        out = model(images)
        if not (isinstance(out, tuple) and len(out) >= 1):
            raise ValueError("Model forward output unexpected.")
        logits = out[0][:, 0:int(num_classes)]
        loss = F.cross_entropy(logits, labels, reduction="sum")
        loss_sum += float(loss.item())
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += int(labels.numel())
    acc = (correct / total) if total > 0 else 0.0
    avg_loss = (loss_sum / total) if total > 0 else 0.0
    return acc, avg_loss


def infer_logdir_from_stage1_ckpt(stage1_ckpt_path: str, ckpt_meta: Dict[str, Any] | None = None, ckpt_args: Dict[str, Any] | None = None) -> str:
    """
    Infer <logdir> for a Stage-1 checkpoint path.

    Priority:
      1) ckpt_meta['logdir'] if present
      2) ckpt_args['log_dir'] or ckpt_args['logdir'] if present
      3) parse path layouts:
         - <logdir>/stage1/ckpts/<file>.pt
         - <logdir>/stage1_ckpts/<file>.pt   (legacy)
      4) fallback to dirname(stage1_ckpt_path)
    """
    if isinstance(ckpt_meta, dict):
        v = ckpt_meta.get("logdir", None)
        if isinstance(v, str) and v.strip():
            return os.path.abspath(v)
    if isinstance(ckpt_args, dict):
        for k in ("log_dir", "logdir"):
            v = ckpt_args.get(k, None)
            if isinstance(v, str) and v.strip():
                return os.path.abspath(v)

    p = os.path.abspath(stage1_ckpt_path)
    d = os.path.dirname(p)
    b = os.path.basename(d)
    # new layout: <logdir>/stage1/ckpts/*.pt
    if b == "ckpts" and os.path.basename(os.path.dirname(d)) == "stage1":
        return os.path.abspath(os.path.join(d, "..", ".."))
    # legacy layout: <logdir>/stage1_ckpts/*.pt
    if b == "stage1_ckpts":
        return os.path.abspath(os.path.join(d, ".."))
    return d


def make_timestamp_tag() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_isolated_run_dir(*, base_out_dir: str, run_name: str | None = None, auto_timestamp_subdir: bool = True) -> str:
    """
    Create an isolated run directory under base_out_dir.

    If auto_timestamp_subdir=True (default), creates:
      <base_out_dir>/<YYYYmmdd-HHMMSS>[_<run_name>]

    If the directory exists and is non-empty, append a monotonic suffix.
    """
    base_out_dir = os.path.abspath(str(base_out_dir))
    os.makedirs(base_out_dir, exist_ok=True)

    if not auto_timestamp_subdir:
        os.makedirs(base_out_dir, exist_ok=True)
        return base_out_dir

    tag = make_timestamp_tag()
    if run_name:
        run_name = str(run_name).strip()
    name = f"{tag}_{run_name}" if run_name else tag
    out_dir = os.path.join(base_out_dir, name)

    # Avoid accidental overwrite if re-running within the same second
    if os.path.exists(out_dir) and os.listdir(out_dir):
        # append millis-ish suffix using time_ns
        import time
        suffix = str(time.time_ns() % 1_000_000_000)
        out_dir = f"{out_dir}_{suffix}"

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def collect_env_info() -> Dict[str, Any]:
    """
    Minimal environment info for reproducibility (no heavy calls).
    """
    import platform
    import sys
    info: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", None),
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        try:
            info["cuda_device_count"] = int(torch.cuda.device_count())
            info["cuda_device_name0"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info

