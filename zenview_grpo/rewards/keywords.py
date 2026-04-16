"""Logic keyword lists for R_word reward."""

LOGIC_KEYWORDS_EN = [
    "first", "then", "next", "because", "therefore",
    "however", "so", "finally", "thus", "hence",
    "since", "as a result", "consequently", "moreover",
    "furthermore", "additionally", "in conclusion",
]

LOGIC_KEYWORDS_ZH = [
    "首先", "然后", "接着", "因为", "因此",
    "所以", "但是", "最后", "由于", "从而",
    "进而", "综上", "总结", "另外", "此外",
]

ALL_LOGIC_KEYWORDS = LOGIC_KEYWORDS_EN + LOGIC_KEYWORDS_ZH

# Valid reference frame canonical names
VALID_REFERENCE_FRAMES = {
    "object-centric",
    "camera-centric",
    "direction-centric",
}

# Alias → canonical mapping (lower-cased keys)
REFERENCE_FRAME_ALIASES = {
    # English variants
    "object-centric": "object-centric",
    "object centric": "object-centric",
    "object-based": "object-centric",
    "object based": "object-centric",
    "egocentric": "object-centric",
    "ego-centric": "object-centric",
    "camera-centric": "camera-centric",
    "camera centric": "camera-centric",
    "camera-based": "camera-centric",
    "camera based": "camera-centric",
    "viewer-centric": "camera-centric",
    "viewer centric": "camera-centric",
    "allocentric": "camera-centric",
    "direction-centric": "direction-centric",
    "direction centric": "direction-centric",
    "direction-based": "direction-centric",
    "direction based": "direction-centric",
    "absolute direction": "direction-centric",
    "absolute-direction": "direction-centric",
    # Chinese variants
    "物体参考系": "object-centric",
    "以物体为参考": "object-centric",
    "相机参考系": "camera-centric",
    "观察者参考系": "camera-centric",
    "摄像机参考系": "camera-centric",
    "方向参考系": "direction-centric",
    "绝对方向参考系": "direction-centric",
    "绝对方向": "direction-centric",
}

# Common answer aliases for spatial reasoning
ANSWER_ALIASES = {
    # left variants
    "left": "left",
    "on the left": "left",
    "to the left": "left",
    "to the left of": "left",
    "the left": "left",
    "l": "left",
    # right variants
    "right": "right",
    "on the right": "right",
    "to the right": "right",
    "to the right of": "right",
    "the right": "right",
    "r": "right",
    # front/front variants
    "front": "front",
    "in front": "front",
    "in front of": "front",
    "ahead": "front",
    # back variants
    "back": "back",
    "behind": "back",
    "at the back": "back",
    # yes/no
    "yes": "yes",
    "yeah": "yes",
    "yep": "yes",
    "correct": "yes",
    "true": "yes",
    "no": "no",
    "nope": "no",
    "false": "no",
    "incorrect": "no",
}

STOPWORDS = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "being"}
