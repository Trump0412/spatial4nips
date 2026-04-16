"""
ZenView GRPO Reward Module
==========================
R = R_fmt + R_ans + 0.25*(R_think_fmt + R_acc) + 0.05*R_word
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .parser import ParsedResponse, parse_response
from .normalize import (
    normalize_answer,
    normalize_frame,
    normalize_object,
    normalize_objects,
    words_to_number,
)
from .keywords import ALL_LOGIC_KEYWORDS


# ── answer matching ────────────────────────────────────────────────────────────

def _get_valid_answers(sample: Dict[str, Any]) -> List[str]:
    """Collect all valid answer strings from a sample dict."""
    answers = []
    raw = sample.get("answer_gt") or sample.get("answer") or ""
    if raw:
        answers.append(str(raw))
    for v in sample.get("valid_answers", []) or []:
        if v:
            answers.append(str(v))
    # Also accept option text if answer_gt is a letter like "A"
    choice_set = sample.get("choice_set") or sample.get("options") or []
    if choice_set and raw:
        raw_upper = str(raw).strip().upper()
        for opt in choice_set:
            opt_str = str(opt)
            if opt_str.startswith(raw_upper + ".") or opt_str.startswith(raw_upper + " "):
                # e.g. "A. left" → add "left"
                answers.append(opt_str.split(".", 1)[-1].strip())
                break
    return answers


def answer_match(pred: str, sample: Dict[str, Any]) -> float:
    """Return 1.0 if pred matches any valid answer, else 0.0."""
    if not pred:
        return 0.0
    pred_norm = normalize_answer(pred)
    pred_num = words_to_number(pred_norm)

    for gt in _get_valid_answers(sample):
        gt_norm = normalize_answer(str(gt))
        if pred_norm == gt_norm:
            return 1.0
        # numeric equivalence
        gt_num = words_to_number(gt_norm)
        if pred_num and gt_num and pred_num == gt_num:
            return 1.0
        # choice letter match: pred="a", gt="A. left"
        if len(pred_norm) == 1 and gt_norm.startswith(pred_norm):
            return 1.0
    return 0.0


# ── frame matching ─────────────────────────────────────────────────────────────

def _get_valid_frames(sample: Dict[str, Any]) -> List[str]:
    frames = []
    for key in ("reference_frame", "reference_type_gt", "reference_frame_gt"):
        v = sample.get(key)
        if v:
            frames.append(str(v))
    for v in sample.get("valid_reference_frames", []) or []:
        if v:
            frames.append(str(v))
    return frames


def frame_match(pred_frame: Optional[str], sample: Dict[str, Any]) -> float:
    """Return 1.0 if predicted frame matches any valid GT frame."""
    if not pred_frame:
        return 0.0
    pred_norm = normalize_frame(pred_frame)
    for gt in _get_valid_frames(sample):
        from .keywords import REFERENCE_FRAME_ALIASES
        gt_canonical = REFERENCE_FRAME_ALIASES.get(gt.lower().strip(), gt.lower().strip())
        if pred_norm == gt_canonical:
            return 1.0
    return 0.0


# ── object matching ────────────────────────────────────────────────────────────

def _get_valid_object_sets(sample: Dict[str, Any]) -> List[List[str]]:
    """Return list of valid object sets (each set is a list of normalized strings)."""
    sets = []
    for key in ("target_object", "target_object_gt"):
        v = sample.get(key)
        if v:
            if isinstance(v, list):
                normed = normalize_objects(v)
            else:
                normed = normalize_objects(str(v))
            if normed:
                sets.append(normed)
    for alias_set in sample.get("valid_target_objects", []) or []:
        if alias_set:
            normed = normalize_objects(alias_set)
            if normed:
                sets.append(normed)
    return sets


def object_match(
    pred_objects: Optional[Any],
    sample: Dict[str, Any],
    partial_credit: bool = False,
) -> float:
    """Return 1.0 if predicted objects match any valid GT object set."""
    if pred_objects is None:
        return 0.0
    if isinstance(pred_objects, str):
        pred_norm = normalize_objects(pred_objects)
    else:
        pred_norm = normalize_objects(list(pred_objects))

    if not pred_norm:
        return 0.0

    valid_sets = _get_valid_object_sets(sample)
    if not valid_sets:
        return 0.0

    pred_set = set(pred_norm)
    for gt_set in valid_sets:
        gt_set_norm = set(gt_set)
        if not gt_set_norm:
            continue
        # Strict: predicted set must cover all GT objects
        if gt_set_norm.issubset(pred_set) or pred_set.issubset(gt_set_norm):
            return 1.0
        if partial_credit:
            overlap = len(pred_set & gt_set_norm) / max(len(gt_set_norm), 1)
            if overlap >= 0.5:
                return 0.5
    return 0.0


# ── R_acc with graceful degradation ───────────────────────────────────────────

def _has_frame_gt(sample: Dict[str, Any]) -> bool:
    return bool(_get_valid_frames(sample))


def _has_object_gt(sample: Dict[str, Any]) -> bool:
    return bool(_get_valid_object_sets(sample))


def compute_r_acc(
    r_frame: float,
    r_object: float,
    sample: Dict[str, Any],
) -> float:
    has_frame = _has_frame_gt(sample)
    has_object = _has_object_gt(sample)
    if has_frame and has_object:
        return 0.5 * r_frame + 0.5 * r_object
    elif has_frame:
        return r_frame
    elif has_object:
        return r_object
    else:
        return 0.0


# ── R_word ─────────────────────────────────────────────────────────────────────

def logic_word_reward(explanation: Optional[str]) -> float:
    """Saturating reward for logic keywords in explanation."""
    if not explanation:
        return 0.0
    text_lower = explanation.lower()
    count = sum(1 for kw in ALL_LOGIC_KEYWORDS if kw.lower() in text_lower)
    if count == 0:
        return 0.0
    elif count == 1:
        return 0.5
    else:
        return 1.0


# ── main reward function ───────────────────────────────────────────────────────

def compute_reward(
    sample: Dict[str, Any],
    response_text: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute ZenView GRPO reward.

    R = R_fmt + R_ans + 0.25*(R_think_fmt + R_acc) + 0.05*R_word

    Returns (total_reward, reward_dict).
    Never raises.
    """
    try:
        parsed = parse_response(response_text)

        # 1. R_fmt
        r_fmt = 1.0 if (parsed.has_valid_think and parsed.has_valid_answer) else 0.0

        # 2. R_ans
        pred_answer = normalize_answer(parsed.answer or "")
        r_ans = answer_match(pred_answer, sample)

        # 3. R_think_fmt
        r_f = 1.0 if parsed.reference_frame_is_valid else 0.0
        r_o = 1.0 if parsed.target_object_non_empty else 0.0
        r_e = 1.0 if parsed.explanation_non_empty else 0.0
        r_think_fmt = (r_f + r_o + r_e) / 3.0

        # 4. R_acc
        r_frame = frame_match(parsed.reference_frame, sample)
        r_object = object_match(parsed.target_object, sample)
        r_acc = compute_r_acc(r_frame, r_object, sample)

        # 5. R_word
        r_word = logic_word_reward(parsed.explanation)

        reward = r_fmt + r_ans + 0.25 * (r_think_fmt + r_acc) + 0.05 * r_word

        reward_dict = {
            "r_fmt": r_fmt,
            "r_ans": r_ans,
            "r_think_fmt": r_think_fmt,
            "r_acc": r_acc,
            "r_frame": r_frame,
            "r_object": r_object,
            "r_word": r_word,
            "reward_total": reward,
        }
        return reward, reward_dict

    except Exception as e:
        # Must not crash training
        zero_dict = {
            "r_fmt": 0.0, "r_ans": 0.0, "r_think_fmt": 0.0,
            "r_acc": 0.0, "r_frame": 0.0, "r_object": 0.0,
            "r_word": 0.0, "reward_total": 0.0,
        }
        return 0.0, zero_dict


def batch_compute_rewards(
    samples: List[Dict[str, Any]],
    responses: List[str],
) -> Tuple[List[float], List[Dict[str, float]]]:
    """Compute rewards for a batch of (sample, response) pairs."""
    rewards, dicts = [], []
    for sample, resp in zip(samples, responses):
        r, d = compute_reward(sample, resp)
        rewards.append(r)
        dicts.append(d)
    return rewards, dicts
