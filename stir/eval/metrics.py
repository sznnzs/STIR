from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re


@dataclass
class EvalSummary:
    n: int
    acc: float
    avg_tokens: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"n": self.n, "acc": self.acc, "avg_tokens": self.avg_tokens}


def accuracy(pred_gold: list[tuple[str | None, str | None]]) -> float:
    correct = 0
    total = 0
    for pred, gold in pred_gold:
        if gold is None:
            continue
        total += 1
        if pred is not None and pred == gold:
            correct += 1
    return (correct / total) if total > 0 else 0.0


_MATH_TASKS = {"math", "competition_math", "math500", "math-500"}


def is_correct(task: str, pred: str | None, gold: str | None) -> bool:
    if pred is None or gold is None:
        return False
    if pred == gold:
        return True
    if str(task).lower() in _MATH_TASKS:
        return bool(_math_equiv(pred, gold))
    return False


_RE_NUM_COMMA = re.compile(r"(?<=\d),(?=\d)")


def _split_top_level_commas(s: str) -> list[str]:
    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in s:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            seg = "".join(buf).strip()
            if seg:
                parts.append(seg)
            buf = []
            continue
        buf.append(ch)
    seg = "".join(buf).strip()
    if seg:
        parts.append(seg)
    return parts


def _prep_sympy(s: str) -> str:
    s = str(s).strip()
    if not s:
        return s
    # Common "x=2" pattern in model outputs.
    if "=" in s:
        s = s.split("=")[-1].strip()
    s = _RE_NUM_COMMA.sub("", s)
    # SymPy uses ** for powers.
    s = s.replace("^", "**")
    return s


def _sympy_parse_expr(s: str):
    import sympy as sp  # type: ignore
    from sympy.parsing.sympy_parser import (  # type: ignore
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
    )

    s = _prep_sympy(s)
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(s, transformations=transformations, evaluate=True)


def _sympy_equiv(a, b) -> bool:
    import sympy as sp  # type: ignore

    try:
        diff = sp.simplify(a - b)
        if diff == 0:
            return True
    except Exception:
        pass
    try:
        eq = a.equals(b)
        if eq is True:
            return True
    except Exception:
        pass
    return False


def _math_equiv(pred: str, gold: str) -> bool:
    """
    SymPy-based equivalence check for MATH-style answers.

    This is a lightweight version of the common MATH evaluation approach: parse
    both answers into symbolic expressions and test equivalence via simplify().
    """
    try:
        # Handle common multi-answer forms: "a, b" (order-insensitive).
        p_parts = _split_top_level_commas(pred)
        g_parts = _split_top_level_commas(gold)
        if len(p_parts) > 1 or len(g_parts) > 1:
            if len(p_parts) != len(g_parts) or not p_parts or not g_parts:
                return False
            p_exprs = [_sympy_parse_expr(x) for x in p_parts]
            g_exprs = [_sympy_parse_expr(x) for x in g_parts]
            used = [False] * len(g_exprs)
            for pe in p_exprs:
                ok = False
                for j, ge in enumerate(g_exprs):
                    if used[j]:
                        continue
                    if _sympy_equiv(pe, ge):
                        used[j] = True
                        ok = True
                        break
                if not ok:
                    return False
            return True

        a = _sympy_parse_expr(pred)
        b = _sympy_parse_expr(gold)
        return _sympy_equiv(a, b)
    except Exception:
        return False

