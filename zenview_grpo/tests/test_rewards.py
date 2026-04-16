"""Unit tests for ZenView GRPO reward module."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
from zenview_grpo.rewards.parser import parse_response
from zenview_grpo.rewards.normalize import normalize_answer, normalize_object, normalize_objects
from zenview_grpo.rewards.spatial_reward import (
    answer_match, frame_match, object_match, logic_word_reward, compute_reward
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

FULL_RESPONSE = """<think>
[Reference_Frame]: camera-centric
[Target_Object]: cup
[Explanation]: First, I look at the image. Then I see the cup is on the left side. Therefore the answer is left.
</think>
<answer>left</answer>"""

SAMPLE_FULL = {
    "answer_gt": "left",
    "valid_answers": ["on the left", "to the left"],
    "reference_frame": "camera-centric",
    "valid_reference_frames": ["camera-centric"],
    "target_object": ["cup"],
    "valid_target_objects": [["cup"], ["the cup"]],
}

SAMPLE_NO_GT = {
    "answer_gt": "left",
    "valid_answers": [],
    "reference_frame": "",
    "valid_reference_frames": [],
    "target_object": [],
    "valid_target_objects": [],
}


# ── parse_response ─────────────────────────────────────────────────────────────

class TestParseResponse:
    def test_full_correct(self):
        p = parse_response(FULL_RESPONSE)
        assert p.has_valid_think
        assert p.has_valid_answer
        assert p.answer == "left"
        assert p.reference_frame == "camera-centric"
        assert p.reference_frame_is_valid
        assert p.target_object == ["cup"]
        assert p.target_object_non_empty
        assert p.explanation_non_empty

    def test_missing_think(self):
        p = parse_response("<answer>left</answer>")
        assert not p.has_valid_think
        assert p.has_valid_answer
        assert p.answer == "left"

    def test_missing_answer(self):
        p = parse_response("<think>[Reference_Frame]: camera-centric\n[Target_Object]: cup\n[Explanation]: test</think>")
        assert p.has_valid_think
        assert not p.has_valid_answer

    def test_invalid_reference_frame(self):
        resp = "<think>\n[Reference_Frame]: unknown-frame\n[Target_Object]: cup\n[Explanation]: test\n</think>\n<answer>left</answer>"
        p = parse_response(resp)
        assert not p.reference_frame_is_valid
        assert p.reference_frame is None

    def test_empty_explanation(self):
        resp = "<think>\n[Reference_Frame]: camera-centric\n[Target_Object]: cup\n[Explanation]:   \n</think>\n<answer>left</answer>"
        p = parse_response(resp)
        # explanation is whitespace only → non_empty should be False
        assert not p.explanation_non_empty

    def test_chinese_reference_frame(self):
        resp = "<think>\n[Reference_Frame]: 相机参考系\n[Target_Object]: 杯子\n[Explanation]: 首先看图\n</think>\n<answer>左边</answer>"
        p = parse_response(resp)
        assert p.reference_frame == "camera-centric"
        assert p.reference_frame_is_valid

    def test_fullwidth_colon(self):
        resp = "<think>\n[Reference_Frame]：camera-centric\n[Target_Object]：cup\n[Explanation]：test\n</think>\n<answer>left</answer>"
        p = parse_response(resp)
        assert p.reference_frame == "camera-centric"

    def test_object_centric_alias(self):
        resp = "<think>\n[Reference_Frame]: object-based\n[Target_Object]: sofa\n[Explanation]: test\n</think>\n<answer>right</answer>"
        p = parse_response(resp)
        assert p.reference_frame == "object-centric"

    def test_empty_string(self):
        p = parse_response("")
        assert not p.has_valid_think
        assert not p.has_valid_answer

    def test_garbage_input(self):
        p = parse_response("random garbage text with no tags")
        assert not p.has_valid_think
        assert not p.has_valid_answer


# ── normalize_answer ───────────────────────────────────────────────────────────

class TestNormalizeAnswer:
    def test_basic(self):
        assert normalize_answer("Left") == "left"
        assert normalize_answer("  RIGHT  ") == "right"

    def test_alias(self):
        assert normalize_answer("on the left") == "left"
        assert normalize_answer("to the right") == "right"

    def test_punctuation(self):
        assert normalize_answer("left.") == "left"
        assert normalize_answer("yes!") == "yes"

    def test_empty(self):
        assert normalize_answer("") == ""
        assert normalize_answer(None) == ""


# ── answer_match ───────────────────────────────────────────────────────────────

class TestAnswerMatch:
    def test_exact(self):
        assert answer_match("left", {"answer_gt": "left"}) == 1.0

    def test_alias(self):
        assert answer_match("on the left", {"answer_gt": "left", "valid_answers": ["on the left"]}) == 1.0

    def test_wrong(self):
        assert answer_match("right", {"answer_gt": "left"}) == 0.0

    def test_choice_letter(self):
        sample = {
            "answer_gt": "A",
            "valid_answers": [],
            "choice_set": ["A. left", "B. right"],
        }
        assert answer_match("A", sample) == 1.0
        assert answer_match("left", sample) == 1.0

    def test_multiple_valid(self):
        sample = {"answer_gt": "yes", "valid_answers": ["correct", "true"]}
        assert answer_match("true", sample) == 1.0

    def test_empty_pred(self):
        assert answer_match("", {"answer_gt": "left"}) == 0.0


# ── frame_match ────────────────────────────────────────────────────────────────

class TestFrameMatch:
    def test_exact(self):
        assert frame_match("camera-centric", {"reference_frame": "camera-centric"}) == 1.0

    def test_alias_in_data(self):
        # data uses "camera-based", pred uses canonical
        assert frame_match("camera-centric", {"reference_frame": "camera-based"}) == 1.0

    def test_wrong(self):
        assert frame_match("object-centric", {"reference_frame": "camera-centric"}) == 0.0

    def test_none_pred(self):
        assert frame_match(None, {"reference_frame": "camera-centric"}) == 0.0

    def test_no_gt(self):
        assert frame_match("camera-centric", {}) == 0.0


# ── object_match ───────────────────────────────────────────────────────────────

class TestObjectMatch:
    def test_exact(self):
        assert object_match(["cup"], {"target_object": ["cup"]}) == 1.0

    def test_article_removal(self):
        assert object_match(["the cup"], {"target_object": ["cup"]}) == 1.0

    def test_alias_set(self):
        sample = {"target_object": ["cup"], "valid_target_objects": [["cup"], ["the cup"]]}
        assert object_match(["the cup"], sample) == 1.0

    def test_multi_object(self):
        sample = {"target_object": ["cup", "table"]}
        assert object_match(["cup", "table"], sample) == 1.0

    def test_wrong(self):
        assert object_match(["sofa"], {"target_object": ["cup"]}) == 0.0

    def test_none_pred(self):
        assert object_match(None, {"target_object": ["cup"]}) == 0.0

    def test_no_gt(self):
        assert object_match(["cup"], {}) == 0.0

    def test_string_input(self):
        assert object_match("cup", {"target_object": ["cup"]}) == 1.0

    def test_comma_separated(self):
        assert object_match("cup, table", {"target_object": ["cup", "table"]}) == 1.0


# ── logic_word_reward ──────────────────────────────────────────────────────────

class TestLogicWordReward:
    def test_zero_keywords(self):
        assert logic_word_reward("The cup is on the left side.") == 0.0

    def test_one_keyword(self):
        assert logic_word_reward("First, I look at the image.") == 0.5

    def test_two_keywords(self):
        assert logic_word_reward("First I look. Then I see. Therefore left.") == 1.0

    def test_chinese_keywords(self):
        assert logic_word_reward("首先看图，然后判断方向。") == 1.0

    def test_empty(self):
        assert logic_word_reward("") == 0.0
        assert logic_word_reward(None) == 0.0


# ── compute_reward ─────────────────────────────────────────────────────────────

class TestComputeReward:
    def test_perfect_response(self):
        reward, rd = compute_reward(SAMPLE_FULL, FULL_RESPONSE)
        assert rd["r_fmt"] == 1.0
        assert rd["r_ans"] == 1.0
        assert rd["r_think_fmt"] == 1.0
        assert rd["r_frame"] == 1.0
        assert rd["r_object"] == 1.0
        assert rd["r_word"] == 1.0
        # R = 1 + 1 + 0.25*(1+1) + 0.05*1 = 2.55
        assert abs(reward - 2.55) < 1e-6

    def test_missing_think(self):
        reward, rd = compute_reward(SAMPLE_FULL, "<answer>left</answer>")
        assert rd["r_fmt"] == 0.0
        assert rd["r_ans"] == 1.0

    def test_wrong_answer(self):
        resp = FULL_RESPONSE.replace("<answer>left</answer>", "<answer>right</answer>")
        reward, rd = compute_reward(SAMPLE_FULL, resp)
        assert rd["r_ans"] == 0.0

    def test_no_gt_fields(self):
        # Should not raise, r_acc should be 0
        reward, rd = compute_reward(SAMPLE_NO_GT, FULL_RESPONSE)
        assert rd["r_acc"] == 0.0
        assert isinstance(reward, float)

    def test_garbage_response(self):
        reward, rd = compute_reward(SAMPLE_FULL, "I don't know")
        assert rd["r_fmt"] == 0.0
        assert reward == 0.0

    def test_reward_formula(self):
        """Verify formula: R = r_fmt + r_ans + 0.25*(r_think_fmt + r_acc) + 0.05*r_word"""
        reward, rd = compute_reward(SAMPLE_FULL, FULL_RESPONSE)
        expected = (
            rd["r_fmt"]
            + rd["r_ans"]
            + 0.25 * (rd["r_think_fmt"] + rd["r_acc"])
            + 0.05 * rd["r_word"]
        )
        assert abs(reward - expected) < 1e-6

    def test_missing_reference_frame_gt(self):
        sample = {**SAMPLE_FULL, "reference_frame": "", "valid_reference_frames": []}
        reward, rd = compute_reward(sample, FULL_RESPONSE)
        # r_acc should only use r_object
        assert rd["r_acc"] == rd["r_object"]

    def test_missing_target_object_gt(self):
        sample = {**SAMPLE_FULL, "target_object": [], "valid_target_objects": []}
        reward, rd = compute_reward(sample, FULL_RESPONSE)
        # r_acc should only use r_frame
        assert rd["r_acc"] == rd["r_frame"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
