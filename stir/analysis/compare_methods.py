from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stir.utils.io import read_jsonl

logger = logging.getLogger(__name__)


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / max(1, len(xs)))


@dataclass(frozen=True)
class MethodComparison:
    T_max: int
    n: int
    acc_greedy: float
    acc_stir: float
    acc_diff: float
    win: int
    lose: int
    both_correct: int
    both_wrong: int
    win_rate: float
    lose_rate: float
    net_win: int
    tokens_greedy_mean: float
    tokens_stir_mean: float
    tokens_diff_mean: float
    tokens_diff_mean_on_win: float
    tokens_diff_mean_on_lose: float
    hard_n: int
    acc_stir_on_hard: float
    tokens_diff_mean_on_hard: float
    stir_budget_mean: float
    stir_overhead_mean: float


def write_method_comparison(
    *,
    run_dir: str | Path,
    T_max: int,
    greedy_tag: str,
    stir_tag: str,
    out_prefix: str = "",
) -> dict[str, Any]:
    """
    Compare greedy vs STIR at one budget point.

    Writes:
    - outputs/<run>/tables/{out_prefix}compare_T{T}.csv
    - outputs/<run>/tables/{out_prefix}compare_T{T}.json
    """
    run_dir = Path(run_dir)
    out_tables = run_dir / "tables"
    out_tables.mkdir(parents=True, exist_ok=True)

    greedy_path = run_dir / "eval" / greedy_tag / "per_example.jsonl"
    stir_path = run_dir / "eval" / stir_tag / "per_example.jsonl"
    if not greedy_path.exists() or not stir_path.exists():
        logger.warning("Comparison skipped (missing per_example): %s %s", greedy_path, stir_path)
        return {}

    greedy_rows = read_jsonl(greedy_path)
    stir_rows = read_jsonl(stir_path)
    g_by_id = {str(r.get("example_id")): r for r in greedy_rows if r.get("example_id") is not None}
    e_by_id = {str(r.get("example_id")): r for r in stir_rows if r.get("example_id") is not None}
    ids = sorted(set(g_by_id.keys()) & set(e_by_id.keys()))
    if not ids:
        logger.warning("Comparison skipped (no overlapping example_id).")
        return {}

    wins = 0
    loses = 0
    both_correct = 0
    both_wrong = 0
    tokens_greedy: list[int] = []
    tokens_stir: list[int] = []
    tokens_diff: list[int] = []
    tokens_diff_win: list[int] = []
    tokens_diff_lose: list[int] = []
    hard: list[str] = []
    acc_stir_hard: list[float] = []
    stir_budget: list[int] = []
    stir_overhead: list[int] = []

    for ex_id in ids:
        g = g_by_id[ex_id]
        e = e_by_id[ex_id]

        g_acc = bool(g.get("correct"))
        e_acc = bool(e.get("correct"))
        if e_acc and not g_acc:
            wins += 1
        if not e_acc and g_acc:
            loses += 1
        if e_acc and g_acc:
            both_correct += 1
        if not e_acc and not g_acc:
            both_wrong += 1

        t_g = int(g.get("tokens_used", 0))
        t_e = int(e.get("tokens_used", 0))
        tokens_greedy.append(t_g)
        tokens_stir.append(t_e)
        tokens_diff.append(int(t_e) - int(t_g))
        if e_acc and not g_acc:
            tokens_diff_win.append(int(t_e) - int(t_g))
        if not e_acc and g_acc:
            tokens_diff_lose.append(int(t_e) - int(t_g))

        t_bud = int(e.get("budget_used", 0))
        stir_budget.append(t_bud)
        stir_overhead.append(max(0, t_bud - t_e))

        # Hard subset: greedy incorrect, STIR correct?
        if not g_acc:
            hard.append(ex_id)
            acc_stir_hard.append(1.0 if e_acc else 0.0)

    n = len(ids)
    wins = int(wins)
    loses = int(loses)
    net_win = wins - loses
    win_rate = float(wins) / float(max(1, n))
    lose_rate = float(loses) / float(max(1, n))

    comp = MethodComparison(
        T_max=int(T_max),
        n=n,
        acc_greedy=float(_safe_mean([1.0 if bool(g_by_id[i].get("correct")) else 0.0 for i in ids])),
        acc_stir=float(_safe_mean([1.0 if bool(e_by_id[i].get("correct")) else 0.0 for i in ids])),
        acc_diff=float(
            _safe_mean([1.0 if bool(e_by_id[i].get("correct")) else 0.0 for i in ids])
            - _safe_mean([1.0 if bool(g_by_id[i].get("correct")) else 0.0 for i in ids])
        ),
        win=wins,
        lose=loses,
        both_correct=int(both_correct),
        both_wrong=int(both_wrong),
        win_rate=float(win_rate),
        lose_rate=float(lose_rate),
        net_win=int(net_win),
        tokens_greedy_mean=float(_safe_mean([float(x) for x in tokens_greedy])),
        tokens_stir_mean=float(_safe_mean([float(x) for x in tokens_stir])),
        tokens_diff_mean=float(_safe_mean([float(x) for x in tokens_diff])),
        tokens_diff_mean_on_win=float(_safe_mean([float(x) for x in tokens_diff_win])),
        tokens_diff_mean_on_lose=float(_safe_mean([float(x) for x in tokens_diff_lose])),
        hard_n=int(len(hard)),
        acc_stir_on_hard=float(_safe_mean([float(x) for x in acc_stir_hard])),
        tokens_diff_mean_on_hard=float(
            _safe_mean([float(tokens_stir[ids.index(i)] - tokens_greedy[ids.index(i)]) for i in hard])
        ),
        stir_budget_mean=float(_safe_mean([float(x) for x in stir_budget])),
        stir_overhead_mean=float(_safe_mean([float(x) for x in stir_overhead])),
    )

    prefix = str(out_prefix or "")
    out_json = out_tables / f"{prefix}compare_T{int(T_max)}.json"
    out_csv = out_tables / f"{prefix}compare_T{int(T_max)}.csv"

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(asdict(comp), f, ensure_ascii=False, indent=2)

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(comp.__dataclass_fields__.keys())  # type: ignore[attr-defined]
        w.writerow([getattr(comp, k) for k in comp.__dataclass_fields__.keys()])  # type: ignore[attr-defined]

    logger.info("Wrote method comparison: %s %s", out_csv, out_json)
    return asdict(comp)
