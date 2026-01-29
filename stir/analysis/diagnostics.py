from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from stir.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


def _percentile_int(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    return int(np.percentile(np.array(xs, dtype=np.int64), p))


def _is_truncated(tokens_used: int, finish_reason: str | None, T_max: int) -> bool:
    if finish_reason is not None:
        return str(finish_reason).lower() == "length"
    # Back-compat with older logs: treat hitting the budget as likely truncation.
    return int(tokens_used) >= int(T_max)


@dataclass(frozen=True)
class MethodDiagnostics:
    method: str
    n: int
    acc: float
    pred_none_rate: float
    hash_rate: float
    trunc_rate: float
    tokens_mean: float
    tokens_p50: int
    tokens_p90: int
    tokens_max: int
    # Only for STIR (includes discarded probes); zeros for greedy.
    budget_mean: float
    overhead_mean: float
    overhead_p90: int
    overhead_max: int
    # Only for STIR: step-level tool usage.
    null_step_rate: float
    tool_step_rate: float


def write_eval_diagnostics(
    *,
    run_dir: str | Path,
    T_max: int,
    greedy_tag: str,
    stir_tag: str,
    out_prefix: str = "",
) -> dict[str, Any]:
    """
    Generate a compact diagnostics report for one budget point.

    Writes:
    - outputs/<run>/tables/{out_prefix}diagnostics_T{T}.csv
    - outputs/<run>/tables/{out_prefix}diagnostics_T{T}.json
    - outputs/<run>/figures/{out_prefix}tokens_hist_T{T}.pdf
    - outputs/<run>/figures/{out_prefix}tokens_hist_T{T}.csv
    """
    run_dir = Path(run_dir)
    out_tables = run_dir / "tables"
    out_figs = run_dir / "figures"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_figs.mkdir(parents=True, exist_ok=True)

    greedy_path = run_dir / "eval" / greedy_tag / "per_example.jsonl"
    stir_path = run_dir / "eval" / stir_tag / "per_example.jsonl"
    if not greedy_path.exists() or not stir_path.exists():
        logger.warning("Diagnostics skipped (missing per_example): %s %s", greedy_path, stir_path)
        return {}

    greedy_rows = read_jsonl(greedy_path)
    stir_rows = read_jsonl(stir_path)

    def diag(method: str, rows: list[dict[str, Any]]) -> MethodDiagnostics:
        n = int(len(rows))
        acc = _safe_mean([1.0 if bool(r.get("correct")) else 0.0 for r in rows])
        pred_none_rate = _safe_mean([1.0 if r.get("pred") is None else 0.0 for r in rows])
        hash_rate = _safe_mean([1.0 if "####" in str(r.get("text", "")) else 0.0 for r in rows])

        toks = [int(r.get("tokens_used", 0)) for r in rows]
        finish = [r.get("finish_reason") for r in rows]
        trunc = _safe_mean([1.0 if _is_truncated(t, fr, T_max) else 0.0 for t, fr in zip(toks, finish)])

        # Optional fields for STIR
        buds = [int(r.get("budget_used", 0) or 0) for r in rows]
        overhead = [max(0, int(b) - int(t)) for b, t in zip(buds, toks)]

        null_steps = 0
        total_steps = 0
        for r in rows:
            for s in r.get("steps", []) or []:
                total_steps += 1
                if str(s.get("chosen")) == "null":
                    null_steps += 1
        null_step_rate = float(null_steps) / float(max(1, total_steps)) if total_steps > 0 else 0.0
        tool_step_rate = 1.0 - null_step_rate if total_steps > 0 else 0.0

        if method.lower() == "greedy":
            buds = [0] * n
            overhead = [0] * n
            null_step_rate = 0.0
            tool_step_rate = 0.0

        return MethodDiagnostics(
            method=method,
            n=n,
            acc=float(acc),
            pred_none_rate=float(pred_none_rate),
            hash_rate=float(hash_rate),
            trunc_rate=float(trunc),
            tokens_mean=float(_safe_mean([float(x) for x in toks])),
            tokens_p50=_percentile_int(toks, 50),
            tokens_p90=_percentile_int(toks, 90),
            tokens_max=int(max(toks) if toks else 0),
            budget_mean=float(_safe_mean([float(x) for x in buds])),
            overhead_mean=float(_safe_mean([float(x) for x in overhead])),
            overhead_p90=_percentile_int(overhead, 90),
            overhead_max=int(max(overhead) if overhead else 0),
            null_step_rate=float(null_step_rate),
            tool_step_rate=float(tool_step_rate),
        )

    dg = diag("greedy", greedy_rows)
    de = diag("stir", stir_rows)
    payload = {"T_max": int(T_max), "greedy": asdict(dg), "stir": asdict(de)}

    for d in [dg, de]:
        if float(d.trunc_rate) >= 0.2:
            logger.warning(
                "High truncation rate for %s @ T_max=%d: %.1f%% (tokens_p90=%d, tokens_max=%d). "
                "Accuracy may be underestimated; consider increasing decode.max_new_tokens (T_max) or using a more compact prompt.",
                d.method,
                int(T_max),
                100.0 * float(d.trunc_rate),
                int(d.tokens_p90),
                int(d.tokens_max),
            )

    prefix = str(out_prefix or "")
    out_json = out_tables / f"{prefix}diagnostics_T{int(T_max)}.json"
    out_csv = out_tables / f"{prefix}diagnostics_T{int(T_max)}.csv"
    out_pdf = out_figs / f"{prefix}tokens_hist_T{int(T_max)}.pdf"
    out_hist_csv = out_figs / f"{prefix}tokens_hist_T{int(T_max)}.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(asdict(dg).keys()))
        w.writerow(list(asdict(dg).values()))
        w.writerow(list(asdict(de).values()))

    # CSV backing for tokens histogram (so users can re-plot without parsing JSONL).
    with out_hist_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "method",
                "task",
                "example_id",
                "tokens_used",
                "budget_used",
                "probe_tokens_used",
                "finish_reason",
                "correct",
            ]
        )
        for method, rows in (("greedy", greedy_rows), ("stir", stir_rows)):
            for r in rows:
                w.writerow(
                    [
                        str(method),
                        r.get("task"),
                        r.get("example_id"),
                        int(r.get("tokens_used", 0) or 0),
                        int(r.get("budget_used", 0) or 0),
                        int(r.get("probe_tokens_used", 0) or 0),
                        r.get("finish_reason"),
                        bool(r.get("correct")),
                    ]
                )

    # Histogram plot (optional dependency)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure(figsize=(5.2, 3.2))
        plt.hist(
            [int(r.get("tokens_used", 0)) for r in greedy_rows],
            bins=20,
            alpha=0.6,
            label="greedy (committed)",
        )
        plt.hist(
            [int(r.get("tokens_used", 0)) for r in stir_rows],
            bins=20,
            alpha=0.6,
            label="stir (committed)",
        )
        plt.axvline(int(T_max), color="k", linestyle="--", linewidth=1.0, alpha=0.5, label="T_max")
        plt.xlabel("Generated tokens (committed)")
        plt.ylabel("Count")
        plt.title(f"Token usage @ T_max={int(T_max)}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_pdf)
        plt.close()
    except Exception as e:
        logger.warning("Skip diagnostics histogram (matplotlib unavailable): %s", e)

    logger.info("Wrote diagnostics: %s %s %s", out_csv, out_json, out_hist_csv)
    return payload
