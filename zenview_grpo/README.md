# ZenView GRPO

GRPO (Group Relative Policy Optimization) training for ZenView spatial reasoning model.

Trains `Qwen2.5-VL-7B` (or any Qwen2-VL checkpoint) with a structured reward function designed for diagnosable spatial reasoning.

## Reward Function

```
R = R_fmt + R_ans + 0.25 * (R_think_fmt + R_acc) + 0.05 * R_word
```

| Component | Description |
|-----------|-------------|
| `R_fmt` | Hard gate: `<think>` + `<answer>` both present |
| `R_ans` | Final answer correctness |
| `R_think_fmt` | Avg of `[Reference_Frame]`, `[Target_Object]`, `[Explanation]` presence |
| `R_acc` | Intermediate field accuracy (frame + object), degrades gracefully if GT missing |
| `R_word` | Saturating logic-keyword reward (0 / 0.5 / 1.0) |

## Directory Structure

```
zenview_grpo/
  train_grpo.py          # training entry point
  evaluate.py            # standalone evaluation
  trainer/
    grpo_trainer.py      # main GRPO loop
    sampling.py          # group response generation
    advantage.py         # z-score / rank advantage
    losses.py            # policy / KL / entropy losses
  rewards/
    parser.py            # robust <think>/<answer> parser
    normalize.py         # text normalization
    spatial_reward.py    # reward computation
    keywords.py          # logic keywords + frame aliases
  data/
    dataset.py           # ZenView JSONL dataset
    collator.py          # multi-image Qwen2-VL collator
    templates.py         # prompt templates
  utils/
    metrics.py           # MetricsTracker + evaluate_dataset
    distributed.py       # DDP helpers
    io.py                # JSON/JSONL I/O
  configs/
    zenview_grpo.yaml    # default config
  tests/
    test_rewards.py      # unit tests
```

## Quick Start

### Single GPU

```bash
cd /path/to/GeoThinker
python zenview_grpo/train_grpo.py --config zenview_grpo/configs/zenview_grpo.yaml
```

### Multi-GPU (4 GPUs)

```bash
torchrun --nproc_per_node=4 zenview_grpo/train_grpo.py \
    --config zenview_grpo/configs/zenview_grpo.yaml
```

### Evaluation only

```bash
python zenview_grpo/evaluate.py \
    --config zenview_grpo/configs/zenview_grpo.yaml \
    --eval_file /path/to/val.jsonl \
    --output_file results.json
```

### Run unit tests

```bash
cd /path/to/GeoThinker
pytest zenview_grpo/tests/test_rewards.py -v
```

## Config

Edit `configs/zenview_grpo.yaml`. Key fields:

```yaml
model_name_or_path: /path/to/checkpoint
train_file: /path/to/train.jsonl
image_root: /path/to/images/root
group_size: 4          # responses per prompt
kl_coef: 0.02          # KL penalty weight
ppo_clip_range: 0.2
learning_rate: 1.0e-6
```

## Data Format

Reads `zenview_master_filtered.jsonl` natively. Also accepts the task-spec format:

```json
{
  "question": "Is the cup on the left or right?",
  "answer_gt": "left",
  "valid_answers": ["on the left"],
  "reference_type_gt": "camera-based",
  "target_object_gt": ["cup"],
  "resolved_media_paths": ["images/scene0001/400.jpg"]
}
```

## Error Attribution

The evaluator reports three diagnostic rates:

- `error_frame_rate` — model used wrong reference frame (semantic failure)
- `error_object_rate` — model identified wrong target object (grounding failure)
- `error_spatial_only_rate` — frame+object correct but answer wrong (geometric reasoning failure)
