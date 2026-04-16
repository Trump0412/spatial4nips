"""Text normalization utilities for answer / object matching."""

import re
import string
from typing import Optional

from .keywords import ANSWER_ALIASES, STOPWORDS


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_answer(text: Optional[str]) -> str:
    """Normalize an answer string for matching."""
    if not text:
        return ""
    norm = normalize_text(text)
    # Try alias on full normalized text first (preserves multi-word phrases)
    if norm in ANSWER_ALIASES:
        return ANSWER_ALIASES[norm]
    # Then try stopword-stripped version
    tokens = [t for t in norm.split() if t not in STOPWORDS]
    stripped = " ".join(tokens)
    return ANSWER_ALIASES.get(stripped, stripped)


def normalize_object(text: Optional[str]) -> str:
    """Normalize a single object name for matching."""
    if not text:
        return ""
    norm = normalize_text(text)
    tokens = [t for t in norm.split() if t not in STOPWORDS]
    return " ".join(tokens)


def normalize_objects(objs) -> list:
    """Normalize a list or comma-string of objects."""
    if objs is None:
        return []
    if isinstance(objs, str):
        parts = re.split(r"[,，、]", objs)
    else:
        parts = list(objs)
    return [normalize_object(p) for p in parts if p and normalize_object(p)]


def normalize_frame(text: Optional[str]) -> str:
    """Normalize a reference frame string (already canonical after parsing)."""
    if not text:
        return ""
    return text.lower().strip()


def words_to_number(text: str) -> Optional[str]:
    """Convert word-form numbers to digit strings for numeric matching."""
    mapping = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10",
    }
    return mapping.get(text.lower().strip())
