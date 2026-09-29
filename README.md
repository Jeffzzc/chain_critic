
# ChainCritic: CriticGen

Research code for **CriticGen: Generation-Aware Evaluation as
Actionable Feedback**.

CriticGen is a fine-grained, generation-aware evaluation framework for
reasoning answers. Instead of applying a fixed checklist or returning only a
scalar score, it builds a sample-specific rubric for each question-answer pair
and uses every rubric item to jointly produce:

1. a score,
2. a criterion-grounded reason,
3. an executable refinement suggestion, and
4. a refined answer.

The central idea is simple: evaluation should identify not only **what is
wrong**, but also **why it is wrong, what should change, and how the answer can
be improved**.

## Method

CriticGen contains two separately trained SFT components connected by a rubric
interface.

```text
Input: question q + candidate answer a
                    |
                    v
Stage 1: Dynamic Rubric Induction
G(q, a) -> R(q, a) = {(d_i, C_i)}

  d_i: sample-specific evaluation dimension
  C_i: question- and answer-grounded 0-5 scoring criterion
                    |
                    v
Stage 2: Rubric-Conditioned Evaluation and Refinement
S(q, a, d_i, C_i) -> (s_i, r_i, u_i, a'_i)

  s_i: score in {0, ..., 5}
  r_i: criterion-grounded reason
  u_i: executable refinement suggestion
  a'_i: answer refined under the same rubric item
```

The dynamic rubric is organized under three broad constraint families while
remaining specific to the current answer:

- **Subjective quality:** clarity, coherence, fluency, and readability.
- **Objective correctness:** factual or numerical correctness and explicit
  task constraints.
- **Self-derived constraints:** consistency between reasoning steps, absence of
  contradictions, and whether the conclusion follows from the reasoning.

These families provide coverage, but they are not a fixed nine-dimension
checklist. The model generates only dimensions that are useful for evaluating
the current `(question, answer)` pair.

## Why Generation-Aware Evaluation?

Many evaluators are coarse-grained, rely on task-level criteria, or stop after
describing an error. CriticGen uses the same rubric item throughout diagnosis
and revision:

```text
criterion -> score -> reason -> edit plan -> refined answer
```

The reason and suggestion have different roles. The reason explains the score
using evidence from the answer; the suggestion specifies a concrete edit and
where or how to apply it. This structured trajectory makes the evaluation
signal directly actionable for generation.

## Main Results

The accompanying paper reports the following results:

| Evaluation                                                      |                    Result |
| --------------------------------------------------------------- | ------------------------: |
| Human-rated rubric relevance: static -> CriticGen               |     3.33 ->**3.97** |
| Human-rated rubric coverage: static -> CriticGen                |     4.03 ->**4.24** |
| Best RefineData score agreement, Pearson / Spearman             | **0.9556 / 0.9560** |
| Best reason semantic-unit F1                                    |          **0.7554** |
| Best executable-suggestion semantic-unit F1                     |          **0.7900** |
| CriticGen-Qwen2.5-7B vs. human scores, Pearson / Spearman       | **0.8868 / 0.8969** |
| Gold-answer accuracy before -> after refinement, 5,000 examples | 45.36% ->**91.26%** |
| Wrong -> Right / Right -> Wrong                                 |  **47.30% / 1.40%** |

The score-correlation results on RefineData measure agreement with
consensus-filtered reference scores and should not be interpreted as universal
evaluator superiority. The paper also reports out-of-domain transfer on
Feedback Bench.

## Supervision Data

CriticGen constructs two complementary datasets.

### RubricData

RubricData trains dynamic rubric induction:

```text
(q, a) -> {(d_i, C_i)}
```

Each item contains a sample-specific dimension and six distinguishable score
anchors from 0 to 5.

### RefineData

RefineData trains rubric-conditioned evaluation and refinement:

```text
(q, a, d_i, C_i) -> (s_i, r_i, u_i, a'_i)
```

The supervision data starts from diverse answers to questions drawn from
GSM8K, LIMO, NaturalReasoning, NuminaMath-CoT, and QwQ-LongCoT-130K. Candidate
answers from models with different capabilities provide varied error patterns.
Missing score regions are completed with plausible answers targeted at the
corresponding 0-5 quality levels.

For the evaluation-refinement stage, scores are retained only when three
teacher models agree. Lexical-diversity analysis and near-duplicate filtering
reduce templated supervision.

| Dataset    |     Train |   Test |     Total |
| ---------- | --------: | -----: | --------: |
| RubricData |    43,145 |  8,629 |    51,774 |
| RefineData | 1,180,929 | 23,619 | 1,204,548 |

The datasets themselves are not committed to this repository; the scripts
under `scripts/data/`, `scripts/evaluation_pipeline/`, and `scripts/filter/`
implement the corresponding construction and filtering stages.

## Installation

Python 3.10 or newer is recommended. The repository does not currently ship a
locked environment, so install the dependencies required by your workflow.

```bash
git clone git@github.com:mug2mag/chain_critic.git
cd chain_critic

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip

# Core API pipeline
python3 -m pip install openai python-dotenv requests tqdm
```

Optional dependencies:

```bash
# Concurrent local inference and load balancing
python3 -m pip install httpx aiohttp fastapi uvicorn

# Local model inference and LoRA adapters
python3 -m pip install torch transformers accelerate peft bitsandbytes

# Experimental GRPO workflows
python3 -m pip install trl datasets
```

GPU training and vLLM workflows have additional version and hardware
requirements. Inspect the relevant launch script before running it.

## API Configuration

The modular commands use OpenAI-compatible chat-completions APIs. Put the
configuration in a local `.env` file or export it in the shell:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="your-model-name"
# Optional; defaults to https://api.openai.com/v1
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Other configured provider names include `deepseek_v31`, `deepseek_r1`,
`doubao_seed_16`, `doubao_1.5_lite_32k`, and `doubao_1.5_pro_256k`. They use
matching environment-variable prefixes. For example:

```bash
export LLM_PROVIDER=deepseek_r1
export DEEPSEEK_R1_API_KEY="your-api-key"
export DEEPSEEK_R1_MODEL="your-model-name"
export DEEPSEEK_R1_BASE_URL="https://your-endpoint.example/v1"
```

Never commit API keys or a populated `.env` file.

## Paper-Aligned Evaluation Pipeline

The most direct implementation of the paper's structured interface is under
`scripts/evaluation_pipeline/`. Input files use JSONL; common question and
answer field names are detected automatically.

### Stage 1: induce dimensions and full-score criteria

```bash
python3 scripts/evaluation_pipeline/generate_dimensions_full_score.py \
  --input data/qa.jsonl \
  --output outputs/dimensions_full_score.jsonl \
  --ports 8000-8003 \
  --model your-rubric-model
```

Example output item:

```json
{
  "question": "...",
  "answer": "...",
  "evaluation_dimensions": [
    {
      "dimension_name": "Arithmetic correctness",
      "category": "objective",
      "full_score_criteria": "Every operation and the final result are correct."
    }
  ]
}
```

### Stage 2: complete the 0-5 scoring criterion

```bash
python3 scripts/evaluation_pipeline/generate_score_criteria_0_5.py \
  --input outputs/dimensions_full_score.jsonl \
  --output outputs/score_criteria_0_5.jsonl \
  --ports 8000-8003 \
  --model your-rubric-model
```

This stage expands every rubric item into question- and answer-specific anchors
for scores 0, 1, 2, 3, 4, and 5.

### Stage 3: score, diagnose, plan, and refine

```bash
python3 scripts/evaluation_pipeline/score_reason_rewrite_local.py \
  --input outputs/score_criteria_0_5.jsonl \
  --output outputs/score_reason_rewrite.jsonl \
  --ports 8000-8003 \
  --model your-refinement-model \
  --reorder
```

Each rubric-conditioned output includes:

```json
{
  "predicted_score": 4,
  "predicted_reason": "...",
  "revision_suggestions": "...",
  "predicted_modified_answer": "..."
}
```

External OpenAI-compatible endpoints are supported through `--base-url`,
`--api-key-env`, and `--model`. Jobs are resumable by `sample_id` unless
`--overwrite` is supplied. See
[the evaluation pipeline guide](scripts/evaluation_pipeline/README.md) for the
complete schemas, field mapping, API examples, and post-pipeline metrics.

## Lightweight Modular Workflow

The modules under `src/` provide a simpler API-driven workflow for prototyping.
They expose the main stages separately, but the basic analyzer stores only a
full-score criterion rather than the paper's complete 0-5 anchors. Use the
pipeline above when reproducing the full CriticGen formulation.

Input may be a JSON array or JSONL with `question` and `answer` fields:

```json
{"question":"A shop has 12 apples and sells 5. How many remain?","answer":"12 - 5 = 7. The answer is 7."}
```

Run dimension induction, rating, and revision:

```bash
python3 -m src.analysis.main \
  --input data/examples.jsonl \
  --output outputs/dimensions.json \
  --sample-size 10 \
  --provider openai

python3 -m src.rating.main \
  --input outputs/dimensions.json \
  --output outputs/ratings.json \
  --sample-size 10 \
  --max-score 5 \
  --provider openai

python3 -m src.iteration_generation.main \
  --strategy one-shot \
  --input outputs/ratings.json \
  --output outputs/revised.json \
  --provider openai \
  --min-score 5 \
  --disable-local
```

The code uses `derived_constraint` for the paper's self-derived-constraint
family. The one-shot and iterative revision strategies are useful engineering
variants; the paper's main formulation generates an independent structured
trajectory under each rubric item.

## Local and Multi-GPU Inference

The pipeline scripts work with one or more OpenAI-compatible local servers,
such as vLLM. Port ranges may be passed as `8000-8003`, while several modular
scripts accept comma-separated ports.

For high-throughput dimension induction:

```bash
python3 scripts/parallel_analysis.py \
  --input data/examples.jsonl \
  --output outputs/dimensions.json \
  --host 127.0.0.1 \
  --ports 8000,8001,8002,8003 \
  --model your-served-model-name
```

`scripts/serve/local_lb.py` provides a least-inflight proxy for multiple local
servers. Its `BACKENDS` list defaults to ports 8000-8007 and should be adapted
to the deployment.

## Training

The paper trains both components with standard next-token SFT:

- rubric induction: maximize the likelihood of the serialized dynamic rubric;
- evaluation-refinement: maximize the likelihood of the complete
  score-reason-suggestion-rewrite sequence.

The reported default setup uses AdamW, learning rate `2e-5`, cosine scheduling,
warmup ratio `0.03`, weight decay `0.01`, 3 epochs, bf16, gradient
checkpointing, maximum sequence length 4096, and effective batch size 128.

Training entry points and data builders live under `scripts/train/`,
`scripts/train_rubric/`, and `scripts/data/`. Several shell scripts contain
machine-specific paths and may represent individual experiment variants rather
than the exact reported configuration; update paths and hyperparameters before
launching them.

## Evaluation

The repository includes utilities for the paper's complementary evaluation
protocols:

- rubric relevance and coverage against a static checklist;
- Pearson and Spearman score correlation;
- semantic-unit precision, recall, and F1 for reasons and suggestions;
- rubric-conditioned improvement, unchanged, and degradation rates;
- normalized gold-answer accuracy and Wrong/Right flip analysis;
- answer sensitivity of induced rubrics;
- out-of-domain scoring on Feedback Bench.

Relevant entry points are located in `scripts/evaluation_pipeline/`,
`scripts/embedding/`, `scripts/baseline/`, and `scripts/benchmark/`.
