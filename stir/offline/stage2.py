from __future__ import annotations

import logging
import math
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from stir.config import STIRConfig
from stir.utils.io import ensure_dir, read_jsonl, write_jsonl, write_text

logger = logging.getLogger(__name__)


def _solve_lower_triangular(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve L x = b for x where L is lower-triangular.
    """
    n = L.shape[0]
    x = np.zeros((n,), dtype=np.float64)
    for i in range(n):
        s = float(b[i])
        if i > 0:
            s -= float(np.dot(L[i, :i], x[:i]))
        denom = float(L[i, i])
        x[i] = s / denom
    return x


def _load_vector_from_pt(path: str) -> np.ndarray:
    """
    Load a single vector from a .pt file and L2-normalize it.
    Used for hidden-state keys (not delta vectors).
    """
    import torch

    v = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError(f"Zero-norm vector: {path}")
    return v / n


def greedy_quality_dpp(
    vectors: np.ndarray,  # (N, d), unit
    qualities: np.ndarray,  # (N,)
    B: int,
    lam: float,
    eps: float,
) -> list[int]:
    """
    Greedy maximize: sum log(1+q_i) + lam * logdet(K_S + eps I),
    where K = V V^T (Gram, PSD).
    """
    N = vectors.shape[0]
    if B <= 0:
        return []
    B = min(B, N)

    # Gram kernel
    K = vectors @ vectors.T  # (N,N)
    diag = np.diag(K) + float(eps)  # (N,)

    selected: list[int] = []
    selected_mask = np.zeros((N,), dtype=bool)

    # Cholesky factor for A = K_S + eps I
    L = np.zeros((0, 0), dtype=np.float64)

    for _t in range(B):
        best_gain = -1e100
        best_i = None
        best_y = None
        best_s = None

        for i in range(N):
            if selected_mask[i]:
                continue
            if len(selected) == 0:
                s = float(diag[i])
                y = None
            else:
                k = K[np.ix_(selected, [i])].reshape(-1)  # (|S|,)
                y = _solve_lower_triangular(L, k)
                s = float(diag[i] - float(np.dot(y, y)))
            if not math.isfinite(s) or s <= 1e-12:
                continue

            q = float(qualities[i])
            q = max(0.0, q)
            gain_q = math.log1p(q)
            gain_div = math.log(s)
            gain = gain_q + float(lam) * gain_div
            if gain > best_gain:
                best_gain = gain
                best_i = i
                best_y = y
                best_s = s

        if best_i is None:
            break

        # Update Cholesky factor
        if len(selected) == 0:
            L = np.array([[math.sqrt(float(best_s))]], dtype=np.float64)
        else:
            assert best_y is not None and best_s is not None
            y = best_y.reshape(1, -1)
            z = np.zeros((L.shape[0], 1), dtype=np.float64)
            diag_new = np.array([[math.sqrt(float(best_s))]], dtype=np.float64)
            L = np.block([[L, z], [y, diag_new]])

        selected.append(int(best_i))
        selected_mask[int(best_i)] = True

    return selected


def select_library(cfg: STIRConfig) -> None:
    """
    Stage II: select a compact, diverse memory library from Stage-I candidates.

    Reads:  outputs/<run>/mine/candidates.jsonl
    Writes: outputs/<run>/library/library.jsonl + keys/* + vectors/*
    """
    run_root = Path(cfg.outputs.run_dir)
    mine_dir = run_root / "mine"
    cand_path = mine_dir / "candidates.jsonl"
    if not cand_path.exists():
        raise FileNotFoundError(f"Stage I candidates not found: {cand_path}")

    candidates = read_jsonl(cand_path)
    if not candidates:
        raise ValueError(f"No candidates found at: {cand_path}")

    # Load key hidden-states (used for diversity).
    logger.info("Loading %d candidate keys from .pt...", len(candidates))
    keys = []
    qs = []
    for c in candidates:
        kp = c.get("key_path", None)
        if kp is None or str(kp).strip() == "":
            raise ValueError("Stage I candidate missing key_path")
        h = _load_vector_from_pt(str(kp))
        keys.append(h)
        qs.append(float(c.get("quality", 0.0)))
    H = np.stack(keys, axis=0)  # (N,d), unit
    Q = np.array(qs, dtype=np.float64)

    method = str(cfg.offline_select.method).lower()
    B = int(cfg.offline_select.B)
    min_per_m = int(getattr(cfg.offline_select, "min_per_control_point", 0))

    forced: list[int] = []
    if min_per_m > 0 and B > 0:
        by_m: dict[int, list[int]] = {}
        for i, c in enumerate(candidates):
            m = int(c.get("control_point_m", -1))
            if m <= 0:
                continue
            by_m.setdefault(m, []).append(i)
        for m in sorted(by_m.keys()):
            if len(forced) >= B:
                break
            idxs = by_m[m]
            idxs = sorted(idxs, key=lambda j: float(candidates[j].get("quality", 0.0)), reverse=True)
            take = min(int(min_per_m), len(idxs), int(B) - len(forced))
            forced.extend(idxs[:take])

    forced_set = set(forced)
    remaining_idx = [i for i in range(len(candidates)) if i not in forced_set]
    B_rem = max(0, int(B) - len(forced))

    if B_rem <= 0:
        chosen = []
    elif method == "top":
        order = sorted(remaining_idx, key=lambda i: float(Q[i]), reverse=True)
        chosen = [int(i) for i in order[: min(B_rem, len(order))]]
    elif method == "random":
        rnd = random.Random(int(cfg.offline_select.random_seed))
        order = list(remaining_idx)
        rnd.shuffle(order)
        chosen = [int(i) for i in order[: min(B_rem, len(order))]]
    elif method == "dpp":
        if not remaining_idx:
            chosen = []
        else:
            H_rem = H[remaining_idx]
            Q_rem = Q[remaining_idx]
            picked_local = greedy_quality_dpp(
                vectors=H_rem,
                qualities=Q_rem,
                B=B_rem,
                lam=float(cfg.offline_select.lambda_diversity),
                eps=float(cfg.offline_select.epsilon),
            )
            chosen = [int(remaining_idx[i]) for i in picked_local]
    else:
        raise ValueError(f"Unknown offline_select.method: {cfg.offline_select.method}")

    chosen = forced + [i for i in chosen if i not in forced_set]

    out_dir = ensure_dir(run_root / "library")
    key_out = ensure_dir(Path(out_dir) / "keys")
    vec_out = ensure_dir(out_dir / "vectors")

    lib_rows: list[dict[str, Any]] = []
    for new_id, idx in enumerate(chosen):
        c = candidates[idx]
        key_src = Path(str(c["key_path"]))
        key_dst = Path(key_out) / key_src.name
        shutil.copyfile(key_src, key_dst)

        vec_dst: Path | None = None
        vec_src_raw = c.get("vector_path", None)
        if vec_src_raw is not None and str(vec_src_raw).strip() != "":
            vec_src = Path(str(vec_src_raw))
            vec_dst = Path(vec_out) / vec_src.name
            shutil.copyfile(vec_src, vec_dst)
        lib_rows.append(
            {
                "lib_id": new_id,
                "tool_id": int(c["tool_id"]),
                "tool_name": str(c["tool_name"]),
                "entry_type": str(c.get("entry_type", "")) or None,
                "key_path": str(key_dst),
                "vector_path": (str(vec_dst) if vec_dst is not None else None),
                "layer": int(c["layer"]),
                "control_point_m": int(c["control_point_m"]),
                "quality": float(c["quality"]),
                "reward_gap": float(c.get("reward_gap", 0.0)),
                "r_pos_mean": float(c.get("r_pos_mean", 0.0)),
                "r_neg_mean": float(c.get("r_neg_mean", 0.0)),
                "n_pos": int(c.get("n_pos", 0)),
                "n_neg": int(c.get("n_neg", 0)),
                "pair_id": int(c.get("pair_id", -1)) if c.get("pair_id", None) is not None else None,
            }
        )

    write_jsonl(Path(out_dir) / "library.jsonl", lib_rows)
    logger.info("Selected %d tools -> %s", len(lib_rows), str(Path(out_dir) / "library.jsonl"))

    # Update latest pointer as well (library belongs to this run).
    try:
        latest_path = Path(cfg.outputs.root_dir) / cfg.outputs.run_name / "LATEST"
        write_text(latest_path, str(cfg.outputs.run_id))
    except Exception:
        pass
