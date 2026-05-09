# Generic QA Evaluation Pipeline

This folder contains a reusable three-stage pipeline for QA answer evaluation.
All model calls use OpenAI-compatible local endpoints, such as vLLM servers.

## Stage 1: Dimensions + Full-Score Criteria

Input: JSONL with QA records. The scripts auto-detect common fields:

- Question: `question`, `prompt`, `instruction`, `query`, `problem`, `orig_instruction`
- Answer: `answer`, `response`, `output`, `candidate_answer`, `model_answer`, `orig_response`
- `messages` format is also supported.

Command:

```powershell
python scripts\evaluation_pipeline\generate_dimensions_full_score.py `
  --input datasets\my_eval\qa.jsonl `
  --output datasets\my_eval\dimensions_full_score.jsonl `
  --ports 8001-8004 `
  --model my-dimension-model
```

Output: one row per QA sample:

```json
{
  "sample_id": "...",
  "question": "...",
  "answer": "...",
  "evaluation_dimensions": [
    {
      "dimension_name": "...",
      "category": "...",
      "full_score_criteria": "..."
    }
  ]
}
```

Use `--question-field` and `--answer-field` when a dataset uses nonstandard field names.

## Stage 2: Complete 0-5 Criteria

Input: Stage 1 output, or any flat JSONL with `question`, `answer`,
`dimension_name`, and `full_score_criteria`.

Command:

```powershell
python scripts\evaluation_pipeline\generate_score_criteria_0_5.py `
  --input datasets\my_eval\dimensions_full_score.jsonl `
  --output datasets\my_eval\score_criteria_0_5.jsonl `
  --ports 8001-8004 `
  --model my-rubric-model
```

Output: one row per QA dimension:

```json
{
  "sample_id": "...:dimension:1",
  "question": "...",
  "answer": "...",
  "dimension_name": "...",
  "full_score_criteria": "...",
  "score_criteria": {
    "0": "...",
    "1": "...",
    "2": "...",
    "3": "...",
    "4": "...",
    "5": "..."
  },
  "criteria_0": "...",
  "criteria_1": "...",
  "criteria_2": "...",
  "criteria_3": "...",
  "criteria_4": "...",
  "criteria_5": "..."
}
```

## Stage 3: Score + Reason + Modified Answer

Input: Stage 2 output.

Command:

```powershell
python scripts\evaluation_pipeline\score_reason_rewrite_local.py `
  --input datasets\my_eval\score_criteria_0_5.jsonl `
  --output datasets\my_eval\score_reason_rewrite.jsonl `
  --ports 8001-8004 `
  --model my-score-rewrite-model `
  --reorder
```

External OpenAI-compatible APIs are also supported:

```powershell
$env:VENDOR_API_KEY="your-key"
python scripts\evaluation_pipeline\score_reason_rewrite_local.py `
  --input datasets\my_eval\score_criteria_0_5.jsonl `
  --output datasets\my_eval\score_reason_rewrite.jsonl `
  --base-url https://vendor.example.com/v1 `
  --api-key-env VENDOR_API_KEY `
  --model vendor-model-name `
  --workers 8 `
  --reorder
```

For direct external `--base-url` endpoints, health check is skipped by default
because many providers require authenticated or provider-specific `/models`
access. Pass `--health-check` to force the `/models` check.

There is also an API-only Stage 3 script with no local vLLM port options:

```powershell
$env:OPENAI_API_KEY="your-key"
python scripts\evaluation_pipeline\score_reason_rewrite_openai_api.py `
  --input datasets\my_eval\score_criteria_0_5.jsonl `
  --output datasets\my_eval\score_reason_rewrite_openai.jsonl `
  --base-url https://api.openai.com/v1 `
  --model gpt-4.1 `
  --workers 8 `
  --reorder
```

For other OpenAI-compatible providers:

```powershell
$env:VENDOR_API_KEY="your-key"
python scripts\evaluation_pipeline\score_reason_rewrite_openai_api.py `
  --input datasets\my_eval\score_criteria_0_5.jsonl `
  --output datasets\my_eval\score_reason_rewrite_vendor.jsonl `
  --base-url https://vendor.example.com/v1 `
  --api-key-env VENDOR_API_KEY `
  --model vendor-model-name `
  --workers 8 `
  --reorder
```

To score a separate QA model output file against previously generated rubrics,
pass `--answers`. The answer file should share an id with the rubric base sample,
such as `sample_id`, `id`, or `unique_id`.

```powershell
python scripts\evaluation_pipeline\score_reason_rewrite_local.py `
  --input datasets\my_eval\score_criteria_0_5.jsonl `
  --answers datasets\my_eval\qa_model_predictions.jsonl `
  --answer-id-field unique_id `
  --answer-field response `
  --output datasets\my_eval\qa_model_score_reason_rewrite.jsonl `
  --ports 8001-8004 `
  --model my-score-rewrite-model `
  --reorder
```

Output: one row per QA dimension:

```json
{
  "sample_id": "...",
  "question": "...",
  "answer": "...",
  "dimension_name": "...",
  "score_criteria": {"0": "...", "1": "...", "2": "...", "3": "...", "4": "...", "5": "..."},
  "predicted_score": 4,
  "predicted_reason": "...",
  "predicted_modified_answer": "..."
}
```

## Common Runtime Options

- `--ports 8000-8003` or `--ports 8000 8001 8002 8003`
- `--base-url-template http://127.0.0.1:{port}/v1`
- `--model ""` lets the script fetch the model id from `/v1/models`
- `--workers N` controls concurrency
- `--overwrite` discards existing output
- Without `--overwrite`, scripts resume by `sample_id`
- `--limit N` runs a small subset for smoke tests
- `--skip-health-check` skips waiting for `/models`

## MATH500-Bench

`datasets/MATH500-Bench/test.jsonl` contains:

- `problem`: math problem text
- `solution`: gold worked solution
- `answer`: gold final answer
- `subject`, `level`, `unique_id`: metadata

For rubric generation, use the worked solution as the reference answer and keep
`unique_id` as the stable sample id:

```powershell
python scripts\evaluation_pipeline\generate_dimensions_full_score.py `
  --input datasets\MATH500-Bench\test.jsonl `
  --output datasets\MATH500-Bench\dimensions_full_score.jsonl `
  --question-field problem `
  --answer-field solution `
  --id-field unique_id `
  --ports 8001-8004 `
  --model my-dimension-model
```

```powershell
python scripts\evaluation_pipeline\generate_score_criteria_0_5.py `
  --input datasets\MATH500-Bench\dimensions_full_score.jsonl `
  --output datasets\MATH500-Bench\score_criteria_0_5.jsonl `
  --ports 8001-8004 `
  --model my-rubric-model
```

If your QA model predictions are stored as JSONL with `unique_id` and `response`,
score them with:

```powershell
python scripts\evaluation_pipeline\score_reason_rewrite_local.py `
  --input datasets\MATH500-Bench\score_criteria_0_5.jsonl `
  --answers datasets\MATH500-Bench\my_qa_model_predictions.jsonl `
  --answer-id-field unique_id `
  --answer-field response `
  --output datasets\MATH500-Bench\my_qa_model_score_reason_rewrite.jsonl `
  --ports 8001-8004 `
  --model my-score-rewrite-model `
  --reorder
```

## Post-Pipeline Evaluation

After Stage 3 produces `score_reason_rewrite.jsonl`, run the following optional
evaluation scripts.

### Answer Relevance

Embedding similarity between generated answers and reference solutions. By
default it compares `predicted_modified_answer` against
`reference_solution/reference_answer`.

```powershell
python scripts\evaluation_pipeline\evaluate_answer_relevance.py `
  --input datasets\MATH500-Bench\my_qa_model_score_reason_rewrite.jsonl `
  --output-dir datasets\MATH500-Bench\answer_relevance `
  --ports 8000-8003 `
  --model my-embedding-model
```

Outputs:

- `answer_relevance\<input_stem>.jsonl`
- `answer_relevance\answer_relevance_summary.csv`
- `answer_relevance\answer_relevance_summary.json`

### Reason Relevance

Embedding similarity between `predicted_reason` and the rubric criterion
corresponding to `predicted_score`.

```powershell
python scripts\evaluation_pipeline\evaluate_reason_relevance.py `
  --input datasets\MATH500-Bench\my_qa_model_score_reason_rewrite.jsonl `
  --output-dir datasets\MATH500-Bench\reason_relevance `
  --ports 8000-8003 `
  --model my-embedding-model
```

Outputs:

- `reason_relevance\<input_stem>.jsonl`
- `reason_relevance\reason_relevance_summary.csv`
- `reason_relevance\reason_relevance_summary.json`

### In-Domain Generation Ability

LLM pairwise judge for generated answers. By default Candidate A is
`predicted_modified_answer`, and Candidate B is
`reference_solution/reference_answer`. The result is WIN/TIE/LOSE from
Candidate A's perspective.

```powershell
python scripts\evaluation_pipeline\evaluate_indomain_generation_ability.py `
  --input datasets\MATH500-Bench\my_qa_model_score_reason_rewrite.jsonl `
  --output-dir datasets\MATH500-Bench\indomain_generation_judge `
  --ports 8000-8003 `
  --model my-judge-model
```

To compare against another model output instead of the reference solution:

```powershell
python scripts\evaluation_pipeline\evaluate_indomain_generation_ability.py `
  --input datasets\MATH500-Bench\my_qa_model_score_reason_rewrite.jsonl `
  --baseline datasets\MATH500-Bench\baseline_score_reason_rewrite.jsonl `
  --output-dir datasets\MATH500-Bench\indomain_generation_vs_baseline `
  --ports 8000-8003 `
  --model my-judge-model
```

Outputs:

- `indomain_generation_judge\<input_stem>.jsonl`
- `indomain_generation_judge\indomain_generation_summary.csv`
- `indomain_generation_judge\indomain_generation_summary.json`
