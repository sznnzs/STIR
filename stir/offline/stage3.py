from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from stir.config import STIRConfig
from stir.utils.io import ensure_dir, read_jsonl, write_jsonl, write_text

logger = logging.getLogger(__name__)


def _load_pt_vector(path: str | Path) -> np.ndarray:
    import torch

    v = torch.load(str(path), map_location="cpu", weights_only=False)
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().numpy()
    return np.asarray(v, dtype=np.float32).reshape(-1)


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, float(eps), None)
    return x / norms


def build_memory(cfg: STIRConfig) -> None:
    """
    Stage III: build episodic memory index for online retrieval.

    Reads:
      - outputs/<run>/library/library.jsonl

    Writes:
      - outputs/<run>/memory/keys.npy          (float32, L2-normalized)
      - outputs/<run>/memory/entries.jsonl     (metadata aligned with keys.npy rows)
      - outputs/<run>/memory/meta.txt
    """
    run_root = Path(cfg.outputs.run_dir)
    lib_path = run_root / "library" / "library.jsonl"
    if not lib_path.exists():
        raise FileNotFoundError(f"Stage II library not found: {lib_path}")

    library = read_jsonl(lib_path)
    if not library:
        raise ValueError(f"Library is empty: {lib_path}")

    keys: list[np.ndarray] = []
    entries: list[dict[str, Any]] = []

    for r in library:
        key_path = r.get("key_path", None)
        if key_path is None or str(key_path).strip() == "":
            raise ValueError("library row missing key_path")
        h = _load_pt_vector(str(key_path))
        keys.append(h)
        entries.append(dict(r))

    key_arr = np.stack(keys, axis=0).astype(np.float32)
    if bool(getattr(cfg.offline_memory, "normalize_keys", True)):
        key_arr = _l2_normalize_rows(key_arr)

    mem_dir = ensure_dir(run_root / "memory")
    np.save(Path(mem_dir) / "keys.npy", key_arr.astype(np.float32))
    write_jsonl(Path(mem_dir) / "entries.jsonl", entries)
    write_text(Path(mem_dir) / "meta.txt", f"N={int(key_arr.shape[0])} H={int(key_arr.shape[1])}\n")

    logger.info("Memory index built: N=%d H=%d -> %s", int(key_arr.shape[0]), int(key_arr.shape[1]), str(mem_dir))
