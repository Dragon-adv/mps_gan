#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
分析 dump_local_low_logits_and_mixup.py 输出的 dump.pt：
  - 预测一致率（orig vs mix）
  - 真类 logit 变化、margin 变化、置信度/熵变化
  - KL(softmax(orig) || softmax(mix)) 分布
  - per-class 分组统计
  - （可选）保存 png 图

使用示例（Windows PowerShell）：
  py analysis/analyze_local_mixup_logits.py `
    --dump_path "D:\...\analysis\local_mixup\...\client_00\dump.pt"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path when running as a script:
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _maybe_win_long_path(path: str) -> str:
    if os.name != "nt":
        return path
    p = os.path.abspath(path)
    if len(p) < 240:
        return p
    if p.startswith("\\\\?\\"):
        return p
    if p.startswith("\\\\"):
        return "\\\\?\\UNC\\" + p.lstrip("\\")
    return "\\\\?\\" + p


def _try_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _summarize_array(x: np.ndarray) -> dict:
    x = np.asarray(x).reshape(-1)
    if x.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p25": float(np.quantile(x, 0.25)),
        "p75": float(np.quantile(x, 0.75)),
        "p95": float(np.quantile(x, 0.95)),
    }


def _entropy_from_probs(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return -(p * torch.log(p.clamp_min(eps))).sum(dim=1)


def _margin_from_probs(p: torch.Tensor) -> torch.Tensor:
    top2 = torch.topk(p, k=2, dim=1, largest=True).values
    return top2[:, 0] - top2[:, 1]


def _kl_pq(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    KL(p||q) per sample.
    p,q: (N,C) probabilities
    """
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (torch.log(p) - torch.log(q))).sum(dim=1)


def _per_class_stats(values: np.ndarray, y: np.ndarray) -> Dict[int, dict]:
    out: Dict[int, dict] = {}
    for c in np.unique(y).tolist():
        c = int(c)
        mask = y == c
        out[c] = _summarize_array(values[mask])
        out[c]["n"] = int(mask.sum())
    return out


def _analyze_one_dump(
    *,
    dump_path: str,
    out_dir: str,
    save_plots: int,
    topk: int,
) -> dict:
    dump_path = os.path.abspath(dump_path)
    out_dir = os.path.abspath(out_dir)
    os.makedirs(_maybe_win_long_path(out_dir), exist_ok=True)
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(_maybe_win_long_path(plots_dir), exist_ok=True)

    obj = torch.load(_maybe_win_long_path(dump_path), map_location="cpu")
    meta = obj.get("meta", {}) if isinstance(obj, dict) else {}

    y = obj["y"].long()  # (N,)
    z_o = obj.get("logits_orig_from_forward", None)
    if z_o is None:
        z_o = obj["logits_orig_from_low"]
    z_o = z_o.float()
    z_m = obj["logits_mix"].float()

    if z_o.shape != z_m.shape:
        raise ValueError(f"logits shape mismatch: orig={tuple(z_o.shape)} mix={tuple(z_m.shape)}")

    n = int(y.numel())
    c = int(z_o.shape[1])

    p_o = torch.softmax(z_o, dim=1)
    p_m = torch.softmax(z_m, dim=1)

    pred_o = p_o.argmax(dim=1)
    pred_m = p_m.argmax(dim=1)
    pred_agree = (pred_o == pred_m)

    conf_o = p_o.max(dim=1).values
    conf_m = p_m.max(dim=1).values
    ent_o = _entropy_from_probs(p_o)
    ent_m = _entropy_from_probs(p_m)
    mar_o = _margin_from_probs(p_o)
    mar_m = _margin_from_probs(p_m)

    # True-class logit / prob
    idx = torch.arange(n, dtype=torch.long)
    zy_o = z_o[idx, y]
    zy_m = z_m[idx, y]
    py_o = p_o[idx, y]
    py_m = p_m[idx, y]

    d_zy = zy_m - zy_o
    d_py = py_m - py_o
    d_conf = conf_m - conf_o
    d_ent = ent_m - ent_o
    d_mar = mar_m - mar_o
    kl_om = _kl_pq(p_o, p_m)

    # convert to numpy for summarization
    y_np = y.numpy()
    pred_o_np = pred_o.numpy()
    pred_m_np = pred_m.numpy()
    pred_agree_np = pred_agree.numpy()

    def _to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy().astype(np.float32, copy=False)

    d_zy_np = _to_np(d_zy)
    d_py_np = _to_np(d_py)
    d_conf_np = _to_np(d_conf)
    d_ent_np = _to_np(d_ent)
    d_mar_np = _to_np(d_mar)
    kl_np = _to_np(kl_om)
    conf_o_np = _to_np(conf_o)
    conf_m_np = _to_np(conf_m)
    ent_o_np = _to_np(ent_o)
    ent_m_np = _to_np(ent_m)
    mar_o_np = _to_np(mar_o)
    mar_m_np = _to_np(mar_m)
    zy_o_np = _to_np(zy_o)
    zy_m_np = _to_np(zy_m)

    summary = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": "analysis/analyze_local_mixup_logits.py",
            "dump_path": dump_path,
            "out_dir": out_dir,
            "n": int(n),
            "num_classes": int(c),
            "source_meta": meta,
        },
        "global": {
            "pred_agree_frac": float(pred_agree.float().mean().item()),
            "orig_acc_on_y": float((pred_o == y).float().mean().item()),
            "mix_acc_on_y": float((pred_m == y).float().mean().item()),
            "kl_orig_mix": _summarize_array(kl_np),
            "delta_true_logit": _summarize_array(d_zy_np),
            "delta_true_prob": _summarize_array(d_py_np),
            "delta_margin": _summarize_array(d_mar_np),
            "delta_conf": _summarize_array(d_conf_np),
            "delta_entropy": _summarize_array(d_ent_np),
            "orig": {
                "conf": _summarize_array(conf_o_np),
                "entropy": _summarize_array(ent_o_np),
                "margin": _summarize_array(mar_o_np),
                "true_logit": _summarize_array(zy_o_np),
            },
            "mix": {
                "conf": _summarize_array(conf_m_np),
                "entropy": _summarize_array(ent_m_np),
                "margin": _summarize_array(mar_m_np),
                "true_logit": _summarize_array(zy_m_np),
            },
        },
        "per_class": {
            "pred_agree_frac": {},
            "kl_orig_mix": _per_class_stats(kl_np, y_np),
            "delta_true_logit": _per_class_stats(d_zy_np, y_np),
            "delta_margin": _per_class_stats(d_mar_np, y_np),
            "delta_conf": _per_class_stats(d_conf_np, y_np),
            "delta_entropy": _per_class_stats(d_ent_np, y_np),
        },
        "worst_samples": {},
    }

    # per-class pred agreement
    for cls in np.unique(y_np).tolist():
        cls = int(cls)
        mask = y_np == cls
        if mask.sum() == 0:
            continue
        summary["per_class"]["pred_agree_frac"][cls] = float(pred_agree_np[mask].mean())

    # worst samples lists
    topk = max(1, min(int(topk), n))
    worst_by_kl = np.argsort(-kl_np)[:topk].tolist()
    worst_by_drop_true_logit = np.argsort(d_zy_np)[:topk].tolist()  # most negative first

    def _pack_indices(idxs: List[int]) -> List[dict]:
        out: List[dict] = []
        pair = obj.get("pair_index_in_collect", None)
        mix_mask = obj.get("mixup_mask", None)
        sel_index = obj.get("sel_index_in_collect", None)
        for ii in idxs:
            rec = {
                "i": int(ii),
                "y": int(y_np[ii]),
                "pred_orig": int(pred_o_np[ii]),
                "pred_mix": int(pred_m_np[ii]),
                "pred_agree": bool(pred_agree_np[ii]),
                "conf_orig": float(conf_o_np[ii]),
                "conf_mix": float(conf_m_np[ii]),
                "margin_orig": float(mar_o_np[ii]),
                "margin_mix": float(mar_m_np[ii]),
                "true_logit_orig": float(zy_o_np[ii]),
                "true_logit_mix": float(zy_m_np[ii]),
                "delta_true_logit": float(d_zy_np[ii]),
                "kl_orig_mix": float(kl_np[ii]),
            }
            if torch.is_tensor(pair):
                rec["pair_index_in_collect"] = int(pair[ii].item())
            if torch.is_tensor(mix_mask):
                rec["mixup_applied"] = bool(mix_mask[ii].item())
            if torch.is_tensor(sel_index):
                rec["sel_index_in_collect"] = int(sel_index[ii].item())
            out.append(rec)
        return out

    summary["worst_samples"] = {
        "by_kl": _pack_indices(worst_by_kl),
        "by_drop_true_logit": _pack_indices(worst_by_drop_true_logit),
    }

    out_json = os.path.join(out_dir, "summary.json")
    with open(_maybe_win_long_path(out_json), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # plots
    if int(save_plots) == 1:
        plt = _try_import_matplotlib()
        if plt is not None:
            try:
                # 1) KL histogram
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                ax.hist(kl_np, bins=60)
                ax.set_title("KL(softmax(orig) || softmax(mix))")
                ax.set_xlabel("KL")
                ax.set_ylabel("count")
                fig.tight_layout()
                fig.savefig(_maybe_win_long_path(os.path.join(plots_dir, "kl_hist.png")), dpi=160)
                plt.close(fig)

                # 2) delta true logit histogram
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                ax.hist(d_zy_np, bins=60)
                ax.set_title("Δ true-class logit (mix - orig)")
                ax.set_xlabel("Δlogit_y")
                ax.set_ylabel("count")
                fig.tight_layout()
                fig.savefig(_maybe_win_long_path(os.path.join(plots_dir, "delta_true_logit_hist.png")), dpi=160)
                plt.close(fig)

                # 3) delta margin histogram
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                ax.hist(d_mar_np, bins=60)
                ax.set_title("Δ margin (top1-top2) (mix - orig)")
                ax.set_xlabel("Δmargin")
                ax.set_ylabel("count")
                fig.tight_layout()
                fig.savefig(_maybe_win_long_path(os.path.join(plots_dir, "delta_margin_hist.png")), dpi=160)
                plt.close(fig)

                # 4) conf scatter
                fig = plt.figure(figsize=(5, 5))
                ax = fig.add_subplot(111)
                ax.scatter(conf_o_np, conf_m_np, s=6, alpha=0.35)
                ax.plot([0, 1], [0, 1], linewidth=1)
                ax.set_title("Confidence scatter (orig vs mix)")
                ax.set_xlabel("conf_orig")
                ax.set_ylabel("conf_mix")
                fig.tight_layout()
                fig.savefig(_maybe_win_long_path(os.path.join(plots_dir, "conf_scatter.png")), dpi=160)
                plt.close(fig)
            except Exception:
                pass

    print(f"[done] wrote: {out_json}")
    if int(save_plots) == 1:
        print(f"[done] plots_dir: {plots_dir}")
    print(f"[summary] pred_agree={summary['global']['pred_agree_frac']:.2%} kl_mean={summary['global']['kl_orig_mix']['mean']:.4f}")
    return summary


def _discover_dump_files(root: str) -> List[str]:
    root = os.path.abspath(root)
    out: List[str] = []
    for cur, _dirs, files in os.walk(root):
        for fn in files:
            if fn == "dump.pt":
                out.append(os.path.join(cur, fn))
    out.sort()
    return out


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument(
        "--dump_path",
        type=str,
        required=True,
        help="Path to a dump.pt or a directory containing multiple client_XX/dump.pt files.",
    )
    p.add_argument("--out_dir", type=str, default=None, help="Output dir (default: alongside each dump.pt; for directory mode: <dump_path>/analysis_out)")
    p.add_argument("--save_plots", type=int, default=1, choices=[0, 1])
    p.add_argument("--topk", type=int, default=30, help="How many worst samples to list for inspection.")
    args = p.parse_args()

    dump_path = os.path.abspath(args.dump_path)
    if os.path.isdir(dump_path):
        dumps = _discover_dump_files(dump_path)
        if not dumps:
            raise FileNotFoundError(f"No dump.pt found under directory: {dump_path}")
        root_out = os.path.abspath(args.out_dir or os.path.join(dump_path, "analysis_out"))
        os.makedirs(_maybe_win_long_path(root_out), exist_ok=True)

        per_client: List[dict] = []
        for dp in dumps:
            # write per-client summary next to each dump by default
            out_dir = os.path.dirname(dp)
            if args.out_dir is not None:
                # mirror relative path under root_out
                rel = os.path.relpath(out_dir, dump_path)
                out_dir = os.path.join(root_out, rel)
            s = _analyze_one_dump(dump_path=dp, out_dir=out_dir, save_plots=int(args.save_plots), topk=int(args.topk))
            g = s.get("global", {})
            m = s.get("meta", {}).get("source_meta", {}) or {}
            per_client.append(
                {
                    "dump_path": dp,
                    "client_id": m.get("client_id", None),
                    "pred_agree_frac": g.get("pred_agree_frac", None),
                    "kl_mean": (g.get("kl_orig_mix", {}) or {}).get("mean", None),
                    "delta_true_logit_mean": (g.get("delta_true_logit", {}) or {}).get("mean", None),
                    "delta_margin_mean": (g.get("delta_margin", {}) or {}).get("mean", None),
                }
            )

        # aggregate
        vals_pa = [x["pred_agree_frac"] for x in per_client if isinstance(x.get("pred_agree_frac"), (int, float))]
        vals_kl = [x["kl_mean"] for x in per_client if isinstance(x.get("kl_mean"), (int, float))]
        agg = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "script": "analysis/analyze_local_mixup_logits.py",
            "mode": "directory",
            "root": dump_path,
            "n_clients": int(len(per_client)),
            "pred_agree_frac": _summarize_array(np.asarray(vals_pa, dtype=np.float32)) if vals_pa else None,
            "kl_mean": _summarize_array(np.asarray(vals_kl, dtype=np.float32)) if vals_kl else None,
            "per_client": per_client,
        }
        out_json = os.path.join(root_out, "summary_all_clients.json")
        with open(_maybe_win_long_path(out_json), "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False, indent=2)
        print(f"[done] wrote: {out_json}")
        return 0

    # single file mode
    out_dir = os.path.abspath(args.out_dir or os.path.dirname(dump_path))
    _analyze_one_dump(dump_path=dump_path, out_dir=out_dir, save_plots=int(args.save_plots), topk=int(args.topk))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

