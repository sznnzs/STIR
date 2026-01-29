from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction


def normalize_text_basic(s: str) -> str:
    return s.strip()


def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w]+$", "", s)  # drop trailing punctuation
    return s


def normalize_number_str(s: str) -> str:
    s = s.strip()
    s = s.replace(",", "")
    # normalize unicode minus
    s = s.replace("−", "-")
    try:
        d = Decimal(s)
        # If it's integral, prefer a plain integer string (avoid '2.6E+2').
        if d == d.to_integral():
            return str(int(d))

        # Canonical decimal form: drop trailing zeros, avoid scientific notation.
        n = d.normalize()
        out = format(n, "f").rstrip("0").rstrip(".")
        return out if out else "0"
    except (InvalidOperation, ValueError):
        return s


_RE_LATEX_INLINE_MATH = re.compile(r"^\s*\$+(.*?)\$+\s*$", re.DOTALL)
_RE_LATEX_PARENS_MATH = re.compile(r"^\s*\\\((.*?)\\\)\s*$", re.DOTALL)
_RE_LATEX_BRACKETS_MATH = re.compile(r"^\s*\\\[(.*?)\\\]\s*$", re.DOTALL)
_RE_LATEX_INLINE_ANY = re.compile(r"\\\((.*?)\\\)", re.DOTALL)
_RE_LATEX_BRACKETS_ANY = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_RE_LATEX_DOLLAR_ANY = re.compile(r"\$(.+?)\$", re.DOTALL)
_SENTENCE_WORD_RE = re.compile(
    r"(?i)(?<!\\)\b(?!sqrt\b|frac\b|pi\b|sin\b|cos\b|tan\b|log\b|ln\b|exp\b|sec\b|csc\b|cot\b)[a-z]{3,}\b"
)

_UNIT_WORDS = {
    "unit",
    "units",
    "cm",
    "mm",
    "km",
    "inch",
    "inches",
    "ft",
    "feet",
    "yd",
    "yard",
    "yards",
    "mile",
    "miles",
    "degree",
    "degrees",
    "deg",
    "radian",
    "radians",
    "rad",
    "percent",
    "pct",
    "dollar",
    "dollars",
    "usd",
}
_UNIT_SUFFIXES_NOSPACE = sorted({u for u in _UNIT_WORDS if len(u) > 1 and u.isalpha()}, key=len, reverse=True)


def _extract_balanced_braces(s: str, open_idx: int) -> tuple[str, int] | None:
    if open_idx < 0 or open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    start = None
    for i in range(open_idx, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
            if depth == 1:
                start = i + 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                return s[start:i], i + 1
    return None


def _strip_markdown_wrappers(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    s = s.strip("`")
    for _ in range(2):
        if s.startswith("**") and s.endswith("**") and len(s) > 4:
            s = s[2:-2].strip()
        if s.startswith("__") and s.endswith("__") and len(s) > 4:
            s = s[2:-2].strip()
    return s.strip("*_")


def _extract_last_tex_command_arg(s: str, command: str) -> str | None:
    needle = "\\" + command
    last = None
    i = 0
    while True:
        j = s.find(needle, i)
        if j < 0:
            break
        k = j + len(needle)
        while k < len(s) and s[k].isspace():
            k += 1
        if k < len(s) and s[k] == "{":
            res = _extract_balanced_braces(s, k)
            if res:
                inner, end = res
                last = inner
                i = end
                continue
        i = k + 1
    return last


def _replace_tex_command_with_arg(s: str, command: str) -> str:
    needle = "\\" + command
    out: list[str] = []
    i = 0
    while True:
        j = s.find(needle, i)
        if j < 0:
            break
        out.append(s[i:j])
        k = j + len(needle)
        while k < len(s) and s[k].isspace():
            k += 1
        if k < len(s) and s[k] == "{":
            res = _extract_balanced_braces(s, k)
            if res:
                inner, end = res
                out.append(inner)
                i = end
                continue
        i = k
    out.append(s[i:])
    return "".join(out)


def _extract_last_math_segment(s: str) -> str | None:
    boxed = _extract_last_tex_command_arg(s, "boxed")
    if boxed:
        return boxed
    matches: list[tuple[int, str]] = []
    for pat in (_RE_LATEX_INLINE_ANY, _RE_LATEX_BRACKETS_ANY, _RE_LATEX_DOLLAR_ANY):
        for m in pat.finditer(s):
            matches.append((m.start(), m.group(1)))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][1]
    return None


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    post_substr = substr[2:]
                    new_str += "{" + a + "}{" + b + "}" + post_substr
                else:
                    post_substr = substr[2:]
                    new_str += "{" + a + "}" + b + post_substr
    return new_str


def _fix_sqrt(string: str) -> str:
    return re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)


def _pick_last_math_token(s: str) -> str:
    tokens = [t.strip(".,;:") for t in s.split() if t.strip(".,;:")]
    if not tokens:
        return s

    def is_math_token(tok: str) -> bool:
        if re.search(r"\d", tok):
            return True
        if "\\" in tok:
            return True
        if re.search(r"[\\^_*/()+-]", tok):
            return True
        if re.search(r"(?i)\\b(pi|sqrt|frac)\\b", tok):
            return True
        return False

    last_idx = None
    for i, tok in enumerate(tokens):
        if is_math_token(tok):
            last_idx = i

    if last_idx is None:
        return tokens[-1]

    cand = tokens[last_idx]
    if last_idx > 0:
        prev = tokens[last_idx - 1]
        if re.fullmatch(r"[-+]?\\d+(?:\\.\\d+)?", prev) and re.match(r"(?:\\\\?sqrt|sqrt)\\b", cand):
            cand = prev + cand
    return cand


def _pick_text_answer(s: str) -> str:
    tokens = [t.strip(".,;:") for t in s.split() if t.strip(".,;:")]
    if not tokens:
        return s
    lower_tokens = [t.lower() for t in tokens]
    for i in range(len(tokens) - 1, -1, -1):
        if lower_tokens[i] in {"is", "are", "was", "were", "be"} and i + 1 < len(tokens):
            return tokens[i + 1]
    while tokens and tokens[0].lower() in {"the", "a", "an", "answer", "final"}:
        tokens = tokens[1:]
    return tokens[0] if tokens else s


def _is_atomic_expr(s: str) -> bool:
    # Used only for adding/removing parentheses in string conversions.
    return bool(re.fullmatch(r"[-+]?[\w.]+", s))


def _latex_to_ascii(s: str) -> str:
    """
    Convert a small subset of LaTeX math into a more comparable ASCII-ish form.
    This is intentionally conservative (covers common final-answer patterns).
    """
    # \frac{a}{b} -> a/b  (add parentheses when needed)
    i = 0
    while True:
        j = s.find("\\frac", i)
        if j < 0:
            break
        k = j + len("\\frac")
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        a = _extract_balanced_braces(s, k)
        if not a:
            i = k + 1
            continue
        num, k2 = a
        if k2 >= len(s) or s[k2] != "{":
            i = k2
            continue
        b = _extract_balanced_braces(s, k2)
        if not b:
            i = k2 + 1
            continue
        den, k3 = b
        num_s = _latex_to_ascii(num)
        den_s = _latex_to_ascii(den)
        if not _is_atomic_expr(num_s):
            num_s = f"({num_s})"
        if not _is_atomic_expr(den_s):
            den_s = f"({den_s})"
        rep = f"{num_s}/{den_s}"
        s = s[:j] + rep + s[k3:]
        i = j + len(rep)

    # \sqrt{a} -> sqrt(a)
    i = 0
    while True:
        j = s.find("\\sqrt", i)
        if j < 0:
            break
        k = j + len("\\sqrt")
        if k >= len(s) or s[k] != "{":
            i = k
            continue
        a = _extract_balanced_braces(s, k)
        if not a:
            i = k + 1
            continue
        inner, k2 = a
        inner_s = _latex_to_ascii(inner)
        rep = f"sqrt({inner_s})"
        s = s[:j] + rep + s[k2:]
        i = j + len(rep)

    return s


def normalize_math_answer(s: str) -> str:
    """
    Best-effort normalizer for MATH-style final answers.

    Goals:
    - Be robust to common LaTeX wrappers (\\boxed, $, \\(\\), \\[\\]).
    - Canonicalize simple numeric/rational answers (e.g., 0.5 == 1/2).
    - Keep non-numeric expressions as a compact, comparable string.
    """
    s = s.strip()
    if not s:
        return s

    s = s.replace("\n", " ").strip()
    s = _strip_markdown_wrappers(s)
    s = s.replace("\u2212", "-")
    s = s.replace("\u221a", "\\sqrt")

    # Prefer explicit math segments when present.
    segment = _extract_last_math_segment(s)
    if segment:
        s = segment.strip()

    # Strip common math wrappers when the whole string is wrapped.
    for pat in (_RE_LATEX_INLINE_MATH, _RE_LATEX_PARENS_MATH, _RE_LATEX_BRACKETS_MATH):
        m = pat.match(s)
        if m:
            s = m.group(1).strip()
            break

    # Replace common text commands early to avoid tokenization issues.
    for cmd in ("text", "mathrm", "mathbf", "mbox"):
        s = _replace_tex_command_with_arg(s, cmd)

    # For sentence-like answers, reduce to a likely answer token.
    if " " in s and _SENTENCE_WORD_RE.search(s):
        if re.search(r"\d", s) or re.search(r"[\\^_*/()+-]", s) or "\\" in s:
            s = _pick_last_math_token(s)
        else:
            s = _pick_text_answer(s)

    # Drop common LaTeX spacing and sizing commands.
    s = re.sub(r"\\(left|right)\b", "", s)
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\!", "")
    s = s.replace("\\displaystyle", "")
    s = s.replace("\\ ", " ")

    # Normalize common operators/symbols.
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\pm", "+-")

    # Normalize fraction command variants.
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")

    # Remove inline math delimiters.
    s = s.replace("\\(", "").replace("\\)", "").replace("\\[", "").replace("\\]", "")
    s = s.replace("\\$", "").replace("$", "")

    # Remove degree markers and percent signs.
    s = s.replace("^{\\circ}", "").replace("^\\circ", "").replace("\\circ", "")
    s = s.replace("\\%", "").replace("%", "")

    # Drop obvious answer prefixes like "x=" or "DE=".
    if "=" in s:
        parts = s.split("=")
        if len(parts) == 2 and len(parts[0].strip()) <= 2:
            s = parts[1].strip()

    # If this is a sentence with numbers and no math operators, keep the last number.
    if re.search(r"\d", s) and not re.search(r"[\\^_*/()+-]", s) and "\\" not in s:
        nums = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", s)
        if nums:
            s = nums[-1]

    # Strip trailing unit words when the answer is numeric-like.
    if re.search(r"\d", s):
        unit_words = "|".join(sorted(_UNIT_WORDS, key=len, reverse=True))
        s = re.sub(r"(?i)(?:\s+(?:" + unit_words + r"))+\s*$", "", s).strip()
        if _UNIT_SUFFIXES_NOSPACE:
            unit_suffix = "|".join(_UNIT_SUFFIXES_NOSPACE)
            s = re.sub(r"(?i)(" + unit_suffix + r")$", "", s).strip()

    s = _fix_sqrt(_fix_fracs(s))

    # Remove remaining whitespace (post replacements), but keep newlines out.
    s = re.sub(r"\s+", "", s)

    # Remove outer braces.
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()

    # Replace LaTeX braces with plain braces (we'll strip later).
    s = s.replace("\\{", "{").replace("\\}", "}")

    s = _latex_to_ascii(s)
    s = re.sub(r"\\([A-Za-z]+)", r"\1", s)

    # After minimal LaTeX conversion, try numeric/rational canonicalization again.
    # Attempt to canonicalize pure numeric / rational strings.
    # Fraction() handles "a/b" and decimals; it will raise on expressions like "(1+2)/3".
    try:
        f = Fraction(s)
        if f.denominator == 1:
            return str(f.numerator)
        return f"{f.numerator}/{f.denominator}"
    except Exception:
        pass

    # Final cleanup: remove redundant braces.
    s = s.replace("{", "").replace("}", "")
    s = s.strip()
    if re.fullmatch(r"[A-Za-z]+", s):
        return s.lower()
    return s
