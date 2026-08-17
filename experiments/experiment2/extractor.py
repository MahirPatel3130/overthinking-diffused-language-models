# EXTRACTOR_VERSION 1 - copied verbatim from experiment1_gsm8k.ipynb. Do not edit.
import re
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional

GT_PATTERN = re.compile(r"####\s*([^\n]+)\s*$")
NUMBER_BODY = r"[-+]?\s*(?:\$\s*)?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
EXPLICIT_PATTERNS = [
    re.compile(rf"(?is)####\s*({NUMBER_BODY})"),
    re.compile(rf"(?is)(?:final\s+answer|answer\s+is|therefore|thus)\s*[:=]?\s*({NUMBER_BODY})"),
    re.compile(rf"(?is)\\boxed\{{\s*({NUMBER_BODY})\s*\}}"),
]
ANY_NUMBER = re.compile(NUMBER_BODY)


def canonical_decimal(text: str) -> str:
    cleaned = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = Decimal(cleaned)
    if not value.is_finite():
        raise InvalidOperation(f"Non-finite number: {text}")
    if value == value.to_integral_value():
        return str(value.quantize(Decimal("1")))
    return format(value.normalize(), "f")


def parse_ground_truth(answer_text: str) -> str:
    match = GT_PATTERN.search(answer_text)
    if not match:
        raise ValueError("Missing GSM8K #### answer delimiter")
    return canonical_decimal(match.group(1))


def extract_answer(text: str) -> Dict[str, Optional[str]]:
    if not text or not text.strip():
        return {"status": "empty", "answer": None, "method": None}
    for pattern in EXPLICIT_PATTERNS:
        matches = list(pattern.finditer(text))
        if matches:
            raw = matches[-1].group(1)
            try:
                return {"status": "ok", "answer": canonical_decimal(re.sub(r"\s+", "", raw)), "method": "explicit"}
            except (InvalidOperation, ValueError):
                return {"status": "invalid_number", "answer": None, "method": "explicit"}
    matches = list(ANY_NUMBER.finditer(text))
    if not matches:
        return {"status": "no_number", "answer": None, "method": None}
    raw = matches[-1].group(0)
    try:
        return {"status": "ok", "answer": canonical_decimal(re.sub(r"\s+", "", raw)), "method": "last_number"}
    except (InvalidOperation, ValueError):
        return {"status": "invalid_number", "answer": None, "method": "last_number"}


assert parse_ground_truth("work\n#### 1,234") == "1234"
assert extract_answer("Therefore, final answer: 42")["answer"] == "42"
assert extract_answer("The answer is $1,250.")["answer"] == "1250"
assert extract_answer("No numeric answer")["answer"] is None
