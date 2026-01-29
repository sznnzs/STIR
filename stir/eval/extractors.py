from __future__ import annotations

import re
from typing import Optional

from stir.eval.normalize import normalize_label, normalize_math_answer, normalize_number_str, normalize_text_basic


_RE_HASH_ANSWER_NUM = re.compile(r"####\s*([-+]?\d[\d,]*\.?\d*)", re.IGNORECASE | re.MULTILINE)
_RE_LAST_NUMBER = re.compile(r"([-+]?\d[\d,]*\.?\d*)")

_RE_HASH_ANSWER_LETTER = re.compile(r"####\s*([A-E])", re.IGNORECASE | re.MULTILINE)
_RE_LAST_LETTER = re.compile(r"\b([A-E])\b")

_RE_HASH_YN = re.compile(r"####\s*(yes|no)", re.IGNORECASE | re.MULTILINE)
_RE_ANSWER_LINE = re.compile(
    r"(?im)^\s*[*_`]*\s*(?:final\s+answer|answer)\s*[*_`]*\s*[:=]\s*(.+?)\s*$"
)
_RE_ANSWER_MARKER_ONLY = re.compile(
    r"(?im)^\s*[*_`]*\s*(?:final\s+answer|answer)\s*[*_`]*\s*[:=]?\s*[*_`]*\s*$"
)
_RE_ANSWER_INLINE = re.compile(
    r"(?i)(?:final\s+answer|answer)\s*[*_`]*\s*(?:is|=|:)\s*([^\n]+)"
)


def _extract_latex_braced_command_args(text: str, command: str) -> list[str]:
    """
    Extract braced arguments for a LaTeX command, handling nested braces.

    Example: r"\\boxed{\\frac{1}{2}}" -> ["\\frac{1}{2}"]
    """
    needle = "\\" + command
    out: list[str] = []
    i = 0
    while True:
        j = text.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        while k < len(text) and text[k].isspace():
            k += 1
        if k >= len(text) or text[k] != "{":
            i = j + len(needle)
            continue
        depth = 0
        start = None
        end = None
        for t in range(k, len(text)):
            ch = text[t]
            if ch == "{":
                depth += 1
                if depth == 1:
                    start = t + 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    end = t
                    break
        if start is not None and end is not None and end >= start:
            out.append(text[start:end])
            i = end + 1
        else:
            i = k + 1
    return out


def extract_pred(task: str, text: str) -> Optional[str]:
    task = task.lower()
    if task in {
        "gsm8k",
        "svamp",
        "aime_2024",
        "aime2024",
        "aime25",
        "aime_25",
        "aime_2025",
        "amc23",
        "amc_23",
        "amc_2023",
    }:
        m = _RE_HASH_ANSWER_NUM.findall(text)
        if m:
            return normalize_number_str(m[-1])
        nums = _RE_LAST_NUMBER.findall(text)
        if nums:
            return normalize_number_str(nums[-1])
        return None

    if task in {"arc-c", "arc_challenge", "arc", "openbookqa", "openbook_qa", "commonsense_qa", "commonsenseqa"}:
        m = _RE_HASH_ANSWER_LETTER.findall(text)
        if m:
            return normalize_label(m[-1]).upper()
        m2 = _RE_LAST_LETTER.findall(text)
        if m2:
            return normalize_label(m2[-1]).upper()
        return None

    if task in {"strategyqa", "strategy_qa"}:
        m = _RE_HASH_YN.findall(text)
        if m:
            return normalize_label(m[-1]).title()
        # fallback: last yes/no occurrence
        m2 = re.findall(r"\b(yes|no)\b", text, flags=re.IGNORECASE)
        if m2:
            return normalize_label(m2[-1]).title()
        return None

    if task in {"math", "competition_math", "math500", "math-500"}:
        boxed = _extract_latex_braced_command_args(text, "boxed")
        if boxed:
            return normalize_math_answer(boxed[-1])
        lines = [ln for ln in str(text).splitlines() if ln.strip()]
        for i, ln in enumerate(lines):
            m = _RE_ANSWER_LINE.match(ln)
            if m:
                ans = m.group(1).strip()
                if ans:
                    norm = normalize_math_answer(ans)
                    if norm:
                        return norm
            if _RE_ANSWER_MARKER_ONLY.match(ln):
                for j in range(i + 1, len(lines)):
                    nxt = lines[j].strip()
                    if nxt:
                        return normalize_math_answer(nxt)
                break
        ans_inline = _RE_ANSWER_INLINE.findall(text)
        if ans_inline:
            return normalize_math_answer(ans_inline[-1])
        # fallback to #### line
        m = _RE_HASH_ANSWER_NUM.findall(text)
        if m:
            return normalize_math_answer(m[-1])
        # last resort: use the last non-empty line (works for some succinct outputs)
        lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
        if lines:
            return normalize_math_answer(lines[-1])
        return None

    raise ValueError(f"Unknown task: {task}")


def extract_gold(task: str, gold_field: str) -> Optional[str]:
    task = task.lower()
    if task == "gsm8k":
        return extract_pred("gsm8k", gold_field)
    if task == "svamp":
        # gold is usually already numeric
        return normalize_number_str(gold_field)
    if task in {
        "aime_2024",
        "aime2024",
        "aime25",
        "aime_25",
        "aime_2025",
        "amc23",
        "amc_23",
        "amc_2023",
    }:
        return normalize_number_str(gold_field)
    if task in {"arc-c", "arc_challenge", "arc", "openbookqa", "openbook_qa", "commonsense_qa", "commonsenseqa"}:
        return normalize_label(gold_field).upper()
    if task in {"strategyqa", "strategy_qa"}:
        return normalize_label(gold_field).title()
    if task in {"math", "competition_math", "math500", "math-500"}:
        # gold is full solution; try boxed in the solution.
        boxed = _extract_latex_braced_command_args(gold_field, "boxed")
        if boxed:
            return normalize_math_answer(boxed[-1])
        return None
    raise ValueError(f"Unknown task: {task}")
