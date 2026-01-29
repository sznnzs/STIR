#!/usr/bin/env python3
"""
Reweight (or redefine) Stage III memory advantages without rerunning GPU-heavy stages.

This is useful when:
- baseline accuracy is already high, so (cf_correct - base_correct) is mostly 0
- you want to amplify token-length differences into a stronger advantage signal

We create a new "artifact run dir" that contains:
- library/  (copied from src run)
- memory/   (copied embeddings/proj + rewritten entries/tool_stats)

Then you can point eval.artifact_run_dir to: outputs/<dst_run_name>/latest
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReweightSpec:
    acc_weight: float
    len_weight: float
    len_gate: str
    eta_source: str
    t_max: int
    src_run_dir: str
    dst_run_dir: str
    created_at: str


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            yield json.loads(s)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _resolve_outputs_root(src_run_dir: Path, src_cfg: dict[str, Any] | None, outputs_root_arg: str | None) -> Path:
    if outputs_root_arg:
        return Path(outputs_root_arg)
    if src_cfg is not None:
        root = ((src_cfg.get("outputs") or {}).get("root_dir")) if isinstance(src_cfg.get("outputs"), dict) else None
        if isinstance(root, str) and root.strip():
            return Path(root)
    # Fallback: assume outputs/<run_name>/<run_id>
    if src_run_dir.parent.name and src_run_dir.parent.parent.exists():
        return src_run_dir.parent.parent
    return Path("outputs")


def _resolve_run_dir_arg(s: str) -> Path:
    """
    Resolve a run directory argument.

    Supported:
    - existing directory: use as-is
    - ".../<run_name>/latest": resolve via ".../<run_name>/LATEST" -> run_id -> directory
    - ".../<run_name>/LATEST": resolve via file contents
    """
    p = Path(str(s)).expanduser()
    if p.is_dir():
        return p

    if p.is_file() and p.name == "LATEST":
        rid = p.read_text(encoding="utf-8").strip()
        cand = p.parent / rid
        if cand.is_dir():
            return cand
        return p

    if p.name.lower() == "latest":
        latest_path = p.parent / "LATEST"
        if latest_path.exists():
            rid = latest_path.read_text(encoding="utf-8").strip()
            cand = p.parent / rid
            if cand.is_dir():
                return cand
    return p


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reweight memory advantages (offline Stage III) without recomputing.")
    p.add_argument(
        "--src-run-dir",
        type=str,
        required=True,
        help="Source run dir that contains library/ and memory/ (e.g., outputs/gsm8k_recommended_small/<RUN_ID>).",
    )
    p.add_argument(
        "--dst-run-name",
        type=str,
        default="gsm8k_recommended_small_lenrew",
        help="Destination run_name under outputs_root (default: gsm8k_recommended_small_lenrew).",
    )
    p.add_argument(
        "--dst-run-id",
        type=str,
        default=None,
        help="Destination run_id (default: timestamp).",
    )
    p.add_argument(
        "--outputs-root",
        type=str,
        default=None,
        help="Override outputs root dir. If omitted, inferred from src config or path.",
    )

    # Reweight knobs
    p.add_argument("--acc-weight", type=float, default=1.0, help="Weight for correctness term.")
    p.add_argument(
        "--len-gate",
        type=str,
        default="always",
        choices=["always", "both_correct", "cf_correct", "base_correct"],
        help=(
            "When to apply the length-delta term (base_tokens - cf_tokens). "
            "'both_correct' matches: wrong answers ignore length."
        ),
    )
    p.add_argument(
        "--len-weight-mode",
        type=str,
        default="no_norm",
        choices=["orig", "no_norm", "custom"],
        help=(
            "How to set len_weight. "
            "'orig' uses eta/T_max (matches Stage III). "
            "'no_norm' uses eta (amplifies length by ~T_max). "
            "'custom' uses --len-weight directly."
        ),
    )
    p.add_argument(
        "--len-weight",
        type=float,
        default=None,
        help="Custom len_weight (used when --len-weight-mode=custom).",
    )
    p.add_argument(
        "--eta-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to len_weight computed from eta (orig/no_norm modes).",
    )
    p.add_argument(
        "--t-max",
        type=int,
        default=None,
        help="Override T_max for orig mode normalization (default: read from src config decode.max_new_tokens).",
    )
    p.add_argument(
        "--eta",
        type=float,
        default=None,
        help="Override eta used for len_weight (default: read from src config offline_memory.eta).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination run dir if it already exists (will delete it first).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    src_run_dir = _resolve_run_dir_arg(str(args.src_run_dir))
    src_cfg = _read_json(src_run_dir / "config_resolved.json")
    outputs_root = _resolve_outputs_root(src_run_dir, src_cfg, args.outputs_root)

    dst_run_id = args.dst_run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    dst_run_dir = outputs_root / str(args.dst_run_name) / str(dst_run_id)

    if dst_run_dir.exists():
        if not bool(args.overwrite):
            raise FileExistsError(f"Destination already exists: {dst_run_dir} (use --overwrite to replace)")
        shutil.rmtree(dst_run_dir)

    # Resolve eta / T_max for weighting
    eta_cfg = None
    t_max_cfg = None
    if isinstance(src_cfg, dict):
        om = src_cfg.get("offline_memory")
        if isinstance(om, dict):
            try:
                eta_cfg = float(om.get("eta"))
            except Exception:
                eta_cfg = None
        dc = src_cfg.get("decode")
        if isinstance(dc, dict):
            try:
                t_max_cfg = int(dc.get("max_new_tokens"))
            except Exception:
                t_max_cfg = None

    eta = float(args.eta) if args.eta is not None else (eta_cfg if eta_cfg is not None else 0.001)
    t_max = int(args.t_max) if args.t_max is not None else (t_max_cfg if t_max_cfg is not None else 2048)

    if str(args.len_weight_mode) == "custom":
        if args.len_weight is None:
            raise ValueError("--len-weight is required when --len-weight-mode=custom")
        len_weight = float(args.len_weight)
        eta_source = "custom"
    elif str(args.len_weight_mode) == "orig":
        len_weight = float(eta) / float(max(1, t_max))
        len_weight *= float(args.eta_scale)
        eta_source = "eta/T_max"
    else:
        # no_norm
        len_weight = float(eta) * float(args.eta_scale)
        eta_source = "eta"

    acc_weight = float(args.acc_weight)
    len_gate = str(args.len_gate)

    # Validate src layout
    src_lib_dir = src_run_dir / "library"
    src_mem_dir = src_run_dir / "memory"
    src_ent = src_mem_dir / "entries.jsonl"
    src_emb = src_mem_dir / "embeddings.npy"
    src_proj = src_mem_dir / "proj.npy"
    if not src_lib_dir.is_dir():
        raise FileNotFoundError(f"Missing library dir: {src_lib_dir}")
    if not src_mem_dir.is_dir():
        raise FileNotFoundError(f"Missing memory dir: {src_mem_dir}")
    if not src_ent.exists():
        raise FileNotFoundError(f"Missing memory entries: {src_ent}")
    if not src_emb.exists():
        raise FileNotFoundError(f"Missing memory embeddings: {src_emb}")
    if not src_proj.exists():
        raise FileNotFoundError(f"Missing memory proj: {src_proj}")

    # Create dst artifact directory
    dst_run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src_lib_dir, dst_run_dir / "library")
    dst_mem_dir = dst_run_dir / "memory"
    dst_mem_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_emb, dst_mem_dir / "embeddings.npy")
    shutil.copy2(src_proj, dst_mem_dir / "proj.npy")
    if (src_mem_dir / "proj_meta.txt").exists():
        shutil.copy2(src_mem_dir / "proj_meta.txt", dst_mem_dir / "proj_meta.txt")

    # Rewrite entries
    new_entries: list[dict[str, Any]] = []
    adv_vals: list[float] = []

    for e in _iter_jsonl(src_ent):
        base_correct = bool(e.get("base_correct"))
        cf_correct = bool(e.get("cf_correct"))
        base_tokens = int(e.get("base_tokens", 0))
        cf_tokens = int(e.get("cf_tokens", 0))

        # advantage = acc_weight*(cf_correct - base_correct) + len_weight*(base_tokens - cf_tokens)
        delta_correct = (1.0 if cf_correct else 0.0) - (1.0 if base_correct else 0.0)
        delta_tokens = float(base_tokens - cf_tokens)
        if len_gate == "always":
            gate = 1.0
        elif len_gate == "both_correct":
            gate = 1.0 if (base_correct and cf_correct) else 0.0
        elif len_gate == "cf_correct":
            gate = 1.0 if cf_correct else 0.0
        elif len_gate == "base_correct":
            gate = 1.0 if base_correct else 0.0
        else:
            raise ValueError(f"Unknown len_gate: {len_gate}")

        acc_term = float(acc_weight) * float(delta_correct)
        len_term = float(len_weight) * float(delta_tokens) * float(gate)
        adv_new = float(acc_term + len_term)

        e2 = dict(e)
        if "advantage_orig" not in e2:
            try:
                e2["advantage_orig"] = float(e2.get("advantage", 0.0))
            except Exception:
                e2["advantage_orig"] = 0.0
        if "r_base_orig" not in e2:
            try:
                e2["r_base_orig"] = float(e2.get("r_base", 0.0))
            except Exception:
                e2["r_base_orig"] = 0.0
        if "r_cf_orig" not in e2:
            try:
                e2["r_cf_orig"] = float(e2.get("r_cf", 0.0))
            except Exception:
                e2["r_cf_orig"] = 0.0

        # Reweighted components (kept lightweight for analysis/debug):
        # - r_base/r_cf: correctness-only reward (so wrong answers ignore length)
        # - advantage = advantage_acc_term + advantage_len_term
        e2["r_base"] = float(acc_weight) * (1.0 if base_correct else 0.0)
        e2["r_cf"] = float(acc_weight) * (1.0 if cf_correct else 0.0)
        e2["advantage_acc_term"] = float(acc_term)
        e2["advantage_len_term"] = float(len_term)
        e2["advantage"] = float(adv_new)
        new_entries.append(e2)
        adv_vals.append(float(adv_new))

    _write_jsonl(dst_mem_dir / "entries.jsonl", new_entries)

    # Per-tool stats
    stats: dict[str, dict[str, float]] = {}
    for e in new_entries:
        tn = str(e.get("tool_name", ""))
        if not tn:
            continue
        stats.setdefault(tn, {"count": 0.0, "adv_sum": 0.0})
        stats[tn]["count"] += 1.0
        stats[tn]["adv_sum"] += float(e.get("advantage", 0.0))
    out_stats = []
    for tn, s in stats.items():
        c = max(1.0, float(s["count"]))
        out_stats.append({"tool_name": tn, "count": float(s["count"]), "adv_sum": float(s["adv_sum"]), "adv_mean": float(s["adv_sum"]) / c})
    out_stats.sort(key=lambda r: float(r.get("adv_mean", 0.0)), reverse=True)
    _write_jsonl(dst_mem_dir / "tool_stats.jsonl", out_stats)

    # Metadata + LATEST pointer
    spec = ReweightSpec(
        acc_weight=float(acc_weight),
        len_weight=float(len_weight),
        len_gate=str(len_gate),
        eta_source=str(eta_source),
        t_max=int(t_max),
        src_run_dir=str(src_run_dir),
        dst_run_dir=str(dst_run_dir),
        created_at=datetime.now().isoformat(timespec="seconds"),
    )
    (dst_run_dir / "reweight_meta.json").write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    latest = (outputs_root / str(args.dst_run_name) / "LATEST")
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(str(dst_run_id) + "\n", encoding="utf-8")

    # Print short summary for terminal usage
    if adv_vals:
        adv_pos = sum(1 for x in adv_vals if x > 0)
        adv_neg = sum(1 for x in adv_vals if x < 0)
        adv_zero = len(adv_vals) - adv_pos - adv_neg
        print(f"[ok] Wrote reweighted artifacts: {dst_run_dir}")
        print(f"[ok] entries={len(adv_vals)} adv_pos={adv_pos} adv_zero={adv_zero} adv_neg={adv_neg}")
        print(
            f"[ok] acc_weight={acc_weight} len_weight={len_weight} len_gate={len_gate} "
            f"(mode={args.len_weight_mode}, eta={eta}, T_max={t_max}, eta_scale={args.eta_scale})"
        )
        print(f"[ok] Point eval.artifact_run_dir to: {outputs_root / str(args.dst_run_name) / 'latest'}")
    else:
        print(f"[warn] No entries found in {src_ent}; wrote empty memory to {dst_run_dir}")


if __name__ == "__main__":
    main()
