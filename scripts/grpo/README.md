# GRPO For Score-Reason-Rewrite

Script:

- `scripts/grpo/train_score_reason_rewrite_grpo.py`

Task format:

- Input: `{Q + A + D + Cs}`
- Output: `{Score + Reason + Modified Answer}`

## Reward Design

The total reward is a weighted average:

`reward = w_format * format + w_score * score_align + w_reason * reason_sim + w_rewrite * rewrite_gain - penalties`

Components:

- `format`
  Checks whether the model output can be parsed into `Score`, `Reason`, and `Modified Answer`.
- `score_align`
  Uses per-sample teacher-score agreement as a proxy for global score correlation:
  `0.7 * (1 - |pred-target| / 5) + 0.3 * exact_match`
- `reason_sim`
  Embedding similarity between generated `Reason` and the rubric entry selected by `pred_score`
- `rewrite_gain`
  Uses a frozen judge to rescore `Modified Answer`, then measures improvement over the original answer:
  `max(0, judge_score(rewrite) - base_score) / max(1, 5 - base_score)`

Default weights:

- `format_weight=0.10`
- `score_weight=0.40`
- `reason_weight=0.20`
- `rewrite_weight=0.30`

## Input Data

Recommended flat JSONL inputs with teacher annotations:

- `datasets/baseline_new/predictions/openai_cortex-5_judge_outputs_with_reference.jsonl`
- `datasets/0-5/score_0.jsonl`
- `datasets/0-5/score_1.jsonl`
- `datasets/0-5/score_2.jsonl`
- `datasets/0-5/score_3.jsonl`
- `datasets/0-5/score_4.jsonl`
- `datasets/0-5/full_score_dimension_seeds.jsonl`

The script auto-detects common fields:

- `question`
- `answer`
- `dimension_name` or `evaluation_dimension`
- `score_criteria` or `criteria_0..5`
- `reference_score` or `Score` or `score`
- `reference_reason` or `Reason` or `reason`
- `reference_modified_answer` or `modified_answer`

## Example Command

```powershell
python scripts\grpo\train_score_reason_rewrite_grpo.py `
  --input datasets\baseline_new\predictions\openai_cortex-5_judge_outputs_with_reference.jsonl `
  --output-dir outputs\grpo\chaincritic-grpo `
  --model-name-or-path Qwen\Qwen2.5-7B-Instruct `
  --judge-base-url-template http://127.0.0.1:{port}/v1 `
  --judge-ports 8000 8001 `
  --embedding-base-url-template http://127.0.0.1:{port}/v1 `
  --embedding-ports 8004 `
  --judge-model Qwen2.5-32B-Instruct `
  --embedding-model Qwen3-Embedding-4B `
  --use-lora `
  --bf16 `
  --gradient-checkpointing
```

## Dependencies

```powershell
pip install trl transformers accelerate peft
```
