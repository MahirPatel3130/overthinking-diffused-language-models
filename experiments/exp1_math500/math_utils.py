"""
Answer extraction + equivalence checking for MATH-500.

MATH-style answers are messier than GSM8K's bare integers -- they're LaTeX
snippets ("\\frac{1}{2}", "(3, -4)", "\\sqrt{5}", ...), so string equality
after light normalization is what most MATH-500 eval harnesses use (with an
optional sympy fallback for numeric equivalence). This module is deliberately
self-contained (no sympy.parsing.latex, which needs antlr4 and is fragile) --
just extraction + normalization + a numeric fallback.
"""

import re

try:
    import sympy
    from sympy.parsing.sympy_parser import parse_expr
    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False


def extract_boxed(text: str):
    """Pull the contents of the last \\boxed{...} (brace-matched) in text.
    Returns None if there is no \\boxed / \\fbox in the text -- this is the
    'no answer yet' case for early diffusion steps."""
    if text is None:
        return None
    for marker in ("\\boxed", "\\fbox"):
        idx = text.rfind(marker)
        if idx == -1:
            continue
        brace_start = text.find("{", idx)
        if brace_start == -1:
            continue
        depth = 0
        for j in range(brace_start, len(text)):
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
                if depth == 0:
                    return text[brace_start + 1 : j]
        # unbalanced braces (truncated mid-generation) -- treat as no answer yet
        return None
    return None


_LATEX_STRIP_PATTERNS = [
    (r"\\left", ""),
    (r"\\right", ""),
    (r"\\!", ""),
    (r"\\,", ""),
    (r"\\ ", " "),
    (r"\\text\{(.*?)\}", r"\1"),
    (r"\\mbox\{(.*?)\}", r"\1"),
    (r"\\dfrac", r"\\frac"),
    (r"\\tfrac", r"\\frac"),
    (r"^\\\$", ""),
    (r"\\\$", ""),
    (r"\$", ""),
    (r"\\%", "%"),
    (r"\\!", ""),
    (r"\.$", ""),
]


def normalize_answer(ans: str) -> str:
    """Canonicalize a MATH answer string for string-level comparison."""
    if ans is None:
        return ""
    s = ans.strip()
    for pat, repl in _LATEX_STRIP_PATTERNS:
        s = re.sub(pat, repl, s)
    s = s.replace(" ", "")
    s = s.rstrip(".")
    # normalize \frac{a}{b} spacing variants, drop outer braces around a
    # bare number e.g. "{5}" -> "5"
    s = re.sub(r"^\{(.*)\}$", r"\1", s)
    # "^{X}" -> "^X" for simple (no nested braces) exponents, e.g.
    # 90^{\circ} -> 90^\circ, so brace-wrapped and bare exponents match.
    s = re.sub(r"\^\{([^{}]*)\}", r"^\1", s)
    # 1,000 -> 1000 -- but ONLY for genuine thousands-grouping (groups of
    # exactly 3 digits after the first comma), so we don't mangle a
    # comma-separated multi-value answer like "3,5,7" or "1,-2" into "357".
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _try_numeric(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def is_equiv(pred: str, gold: str, numeric_tol: float = 1e-4) -> bool:
    """True if pred and gold represent the same MATH answer.

    Order: exact string match after normalization -> numeric match ->
    (optional) sympy symbolic match for simple algebraic expressions.
    """
    if pred is None or gold is None:
        return False
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ == ng and np_ != "":
        return True

    # Multi-value answers (e.g. "3,5,7") are sometimes wrapped in
    # parentheses by the model even when the reference answer isn't (or
    # vice versa) -- strip one layer of outer parens from whichever side
    # has them and retry, rather than treating that as a real mismatch.
    def _strip_outer_parens(s):
        if s.startswith("(") and s.endswith(")"):
            return s[1:-1]
        return s

    if _strip_outer_parens(np_) == _strip_outer_parens(ng) and np_ != "":
        return True

    pn, gn = _try_numeric(np_), _try_numeric(ng)
    if pn is not None and gn is not None:
        return abs(pn - gn) <= numeric_tol

    if _HAS_SYMPY:
        try:
            # only attempt for short, plain-ish algebraic strings -- avoid
            # sympy choking on raw LaTeX like \sqrt or \frac by doing a
            # couple of cheap substitutions first.
            def to_sympy_friendly(x):
                x = x.replace("\\frac", "").replace("{", "(").replace("}", ")")
                x = x.replace("\\sqrt(", "sqrt(")
                x = x.replace("^", "**")
                return x

            pe = parse_expr(to_sympy_friendly(np_))
            ge = parse_expr(to_sympy_friendly(ng))
            return bool(sympy.simplify(pe - ge) == 0)
        except Exception:
            return False

    return False


def extract_gsm8k_answer(text: str):
    """Kept here for convenience if Mahir's GSM8K extractor is needed for
    cross-checking. Not used for MATH-500."""
    if text is None:
        return None
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text.replace(",", ""))
    return matches[-1] if matches else None