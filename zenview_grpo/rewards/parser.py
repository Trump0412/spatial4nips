"""Robust parser for ZenView structured model outputs.

Expected format:
<think>
[Reference_Frame]: object-centric
[Target_Object]: cup
[Explanation]: First ... Therefore ...
</think>
<answer>left</answer>
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Union

from .keywords import REFERENCE_FRAME_ALIASES, VALID_REFERENCE_FRAMES


@dataclass
class ParsedResponse:
    raw_text: str
    think_text: Optional[str] = None
    answer: Optional[str] = None
    reference_frame: Optional[str] = None          # canonical form after alias mapping
    reference_frame_raw: Optional[str] = None      # raw extracted string
    target_object: Optional[Union[str, list]] = None
    explanation: Optional[str] = None
    has_valid_think: bool = False
    has_valid_answer: bool = False
    reference_frame_is_valid: bool = False
    target_object_non_empty: bool = False
    explanation_non_empty: bool = False


# ── regex patterns ────────────────────────────────────────────────────────────
_RE_THINK = re.compile(
    r"<think>(.*?)</think>",
    re.DOTALL | re.IGNORECASE,
)
_RE_ANSWER = re.compile(
    r"<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
# Matches [Reference_Frame]: value  (colon may be full-width ：)
_RE_REF_FRAME = re.compile(
    r"\[Reference[_\s]Frame\]\s*[：:]\s*(.+)",
    re.IGNORECASE,
)
_RE_TARGET_OBJ = re.compile(
    r"\[Target[_\s]Object\]\s*[：:]\s*(.+)",
    re.IGNORECASE,
)
_RE_EXPLANATION = re.compile(
    r"\[Explanation\]\s*[：:]\s*([\s\S]+?)(?=\[|$)",
    re.IGNORECASE,
)


def _normalize_frame(raw: str) -> Optional[str]:
    """Map raw reference frame string to canonical form, or None if invalid."""
    key = raw.strip().lower()
    return REFERENCE_FRAME_ALIASES.get(key)


def _parse_target_objects(raw: str) -> list:
    """Split comma-separated target objects and strip whitespace."""
    parts = [p.strip() for p in re.split(r"[,，、]", raw) if p.strip()]
    return parts if parts else []


def parse_response(text: str) -> ParsedResponse:
    """Parse a model response into a ParsedResponse dataclass.

    Never raises — on any failure the relevant fields are None / False.
    """
    result = ParsedResponse(raw_text=text)

    try:
        # ── think block ───────────────────────────────────────────────────────
        think_match = _RE_THINK.search(text)
        if think_match:
            result.think_text = think_match.group(1).strip()
            result.has_valid_think = True

        # ── answer block ──────────────────────────────────────────────────────
        answer_match = _RE_ANSWER.search(text)
        if answer_match:
            ans = answer_match.group(1).strip()
            if ans:
                result.answer = ans
                result.has_valid_answer = True

        # ── sub-fields inside think ───────────────────────────────────────────
        search_text = result.think_text if result.think_text else text

        # Reference_Frame
        rf_match = _RE_REF_FRAME.search(search_text)
        if rf_match:
            raw_rf = rf_match.group(1).strip().split("\n")[0].strip()
            result.reference_frame_raw = raw_rf
            canonical = _normalize_frame(raw_rf)
            if canonical:
                result.reference_frame = canonical
                result.reference_frame_is_valid = True
            # field exists but value is invalid → reference_frame_is_valid stays False

        # Target_Object
        to_match = _RE_TARGET_OBJ.search(search_text)
        if to_match:
            raw_to = to_match.group(1).strip().split("\n")[0].strip()
            objects = _parse_target_objects(raw_to)
            if objects:
                result.target_object = objects
                result.target_object_non_empty = True

        # Explanation
        exp_match = _RE_EXPLANATION.search(search_text)
        if exp_match:
            exp = exp_match.group(1).strip()
            if exp:
                result.explanation = exp
                result.explanation_non_empty = True

    except Exception:
        # Parsing must never crash training
        pass

    return result
