from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stir.utils.io import read_jsonl


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    n = float(np.linalg.norm(x))
    if not np.isfinite(n) or n <= float(eps):
        return x * 0.0
    return x / n


@dataclass
class EpisodicMemory:
    """
    Episodic memory for online retrieval.

    - keys[i] is the L2-normalized hidden-state key (float32) for entries[i]
    - entries[i] contains metadata, including:
        - tool_name
        - control_point_m
        - layer
        - quality
        - vector_path (nullable for null-like entries)
        - entry_type ("wrong"/"right", optional)
    """

    keys: np.ndarray  # (N, H) float32, L2-normalized
    entries: list[dict[str, Any]]  # len N

    @classmethod
    def load(cls, mem_dir: str | Path) -> "EpisodicMemory":
        mem_dir = Path(mem_dir)
        keys = np.load(mem_dir / "keys.npy").astype(np.float32)
        entries = read_jsonl(mem_dir / "entries.jsonl")
        if int(keys.shape[0]) != int(len(entries)):
            raise ValueError("Row count mismatch between memory/entries.jsonl and keys.npy")
        # Defensive normalize (Stage III already writes normalized keys).
        norms = np.linalg.norm(keys, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        keys = keys / norms
        return cls(keys=keys, entries=entries)

    def count_for_m(self, control_point_m: int) -> int:
        mm = int(control_point_m)
        return int(sum(1 for e in self.entries if int(e.get("control_point_m", -1)) == mm))

    def topk_debug(
        self,
        query_by_layer: dict[int, np.ndarray],
        *,
        k: int,
        control_point_m: int | None = None,
        min_sim: float | None = None,
        min_entries: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        info: dict[str, Any] = {
            "control_point_m": int(control_point_m) if control_point_m is not None else None,
            "n_total_entries": int(self.keys.shape[0]),
            "n_entries_m": None,
            "k": int(k),
            "k_used": 0,
            "top_sim": None,
            "min_sim": float(min_sim) if min_sim is not None else None,
            "min_entries": int(min_entries),
            "reason": None,
        }

        if int(self.keys.shape[0]) <= 0:
            info["reason"] = "empty_memory"
            return np.array([], dtype=int), np.array([], dtype=np.float32), info

        if control_point_m is not None:
            n_m = self.count_for_m(int(control_point_m))
            info["n_entries_m"] = int(n_m)
            if int(n_m) <= 0:
                info["reason"] = "no_entries_for_m"
                return np.array([], dtype=int), np.array([], dtype=np.float32), info
            if int(n_m) < int(min_entries):
                info["reason"] = "too_few_entries"
                return np.array([], dtype=int), np.array([], dtype=np.float32), info
        else:
            info["n_entries_m"] = int(self.keys.shape[0])

        # Normalize queries once.
        q_norm: dict[int, np.ndarray] = {int(l): _l2_normalize(v) for l, v in query_by_layer.items()}

        sims: list[tuple[int, float]] = []
        mm = None if control_point_m is None else int(control_point_m)
        for i, e in enumerate(self.entries):
            if mm is not None and int(e.get("control_point_m", -1)) != mm:
                continue
            try:
                lid = int(e.get("layer"))
            except Exception:
                continue
            q = q_norm.get(lid, None)
            if q is None:
                continue
            s = float(np.dot(self.keys[i], q))
            if not np.isfinite(s):
                continue
            sims.append((int(i), float(s)))

        if not sims:
            info["reason"] = "no_query_for_layers"
            return np.array([], dtype=int), np.array([], dtype=np.float32), info

        sims.sort(key=lambda x: x[1], reverse=True)
        k_eff = min(int(k), len(sims))
        idx = np.array([i for i, _ in sims[:k_eff]], dtype=int)
        sim_arr = np.array([s for _, s in sims[:k_eff]], dtype=np.float32)

        info["k_used"] = int(k_eff)
        info["top_sim"] = float(sim_arr[0]) if int(sim_arr.size) > 0 else None
        if min_sim is not None and int(sim_arr.size) > 0 and float(sim_arr[0]) < float(min_sim):
            info["reason"] = "low_sim"
            return np.array([], dtype=int), np.array([], dtype=np.float32), info

        info["reason"] = "ok"
        return idx, sim_arr, info
