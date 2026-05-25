#!/usr/bin/env python3
"""Run Feedback-Bench rubric JSONL through local vLLM endpoints.

Defaults:
 - input: datasets/Feedback-Bench/data/score_rubric_0_5.jsonl
 - output dir: evaluation/predictions/Feedback-Bench
 - base_urls: http://127.0.0.1:8001/v1,...,http://127.0.0.1:8007/v1

This reuses the prompt format in the repo's tagged training data and outputs JSONL
rows compatible with other evaluation scripts:
predicted_score / predicted_reason / predicted_revision_suggestions / predicted_modified_answer.
"""

import argparse
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_INPUT = "datasets/Feedback-Bench/data/score_rubric_0_5.jsonl"
DEFAULT_OUTDIR = "datasets/Feedback-Bench/test"
DEFAULT_BASE_URLS = ",".join([f"http://127.0.0.1:{p}/v1" for p in range(8000, 8008)])
DEFAULT_API_KEY = "EMPTY"


def safe_filename(text: str) -> str:
    """Convert model id to a filesystem-safe filename part."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
    return cleaned.strip("._-") or "model"

def read_jsonl(path: str):
    """Read a JSONL file line by line."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_jsonl(path: str, rows: List[Dict[str, Any]]):
    """Append rows to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_criteria_text(sample: Dict[str, Any]) -> str:
    """Build complete 0-5 scoring criteria text from Feedback-Bench row."""
    lines = []

    score_criteria = sample.get("score_criteria")
    if isinstance(score_criteria, dict):
        for k in ["0", "1", "2", "3", "4", "5"]:
            if k in score_criteria and score_criteria[k]:
                lines.append(f"{k}: {score_criteria[k]}")
        if lines:
            return "\n".join(lines)

    for k in range(6):
        key = f"criteria_{k}"
        value = sample.get(key)
        if value:
            lines.append(f"{k}: {value}")

    if lines:
        return "\n".join(lines)

    return str(sample.get("full_score_criteria", "") or "")


def build_messages(sample: Dict[str, Any]):
    """Build OpenAI-style chat messages for one Feedback-Bench sample."""
    system = (
        "You are an AI evaluator-and-rewriter.\n"
        "Evaluate the given answer strictly using ONLY the provided evaluation dimension and the complete 0-5 scoring criteria.\n"
        "Then revise the answer to better satisfy ONLY that dimension.\n"
        "Do not add unsupported facts. If an assumption is necessary, state it minimally and explicitly.\n"
        "Output plain text in exactly 1 line using these tags and no numbering:\n"
        "<s>score</s><r>reason</r><rs>revision suggestions</rs><ra>refined answer</ra>\n"
        "Do not include any extra text, JSON, markdown, bullets, or line breaks inside any field."
    )

    user_parts = [
        "###Task Description:",
        "You are given a question, a response to evaluate, and one evaluation dimension with its complete 0-5 scoring criteria.",
        "1. Write a score that reflects how well the response satisfies the given criteria.",
        "2. Write feedback that assesses the quality relative to the criteria.",
        "3. Provide concise revision suggestions to improve the response for that dimension.",
        "4. Provide a refined answer that better satisfies the dimension (only change what's needed).",
        "---",
        f"Question: {sample.get('question', '')}",
        f"Answer: {sample.get('answer', '')}",
        f"Evaluation dimension: {sample.get('dimension_name', '')}",
        "Scoring criteria:",
        build_criteria_text(sample),
    ]

    user_content = "\n".join(user_parts)

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def fetch_model_id(base_url: str, timeout: int = 30) -> str:
    """Fetch first served model id from /v1/models."""
    import requests

    resp = requests.get(base_url.rstrip("/") + "/models", timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()

    data = payload.get("data")
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict) and first.get("id"):
            return str(first["id"])

    return ""


def resolve_model(base_urls: List[str], model: str, timeout: int = 30) -> str:
    """Use user-provided model id, or fetch from available vLLM endpoint."""
    if model.strip():
        return model.strip()

    for base_url in base_urls:
        try:
            model_id = fetch_model_id(base_url, timeout=timeout)
        except Exception:
            model_id = ""

        if model_id:
            return model_id

    raise ValueError("No --model was provided and no model id could be fetched from /v1/models.")


def call_vllm(
    base_urls: List[str],
    model: str,
    messages: List[Dict[str, str]],
    api_key: str = DEFAULT_API_KEY,
    timeout: int = 30,
    max_tokens: int = 1024,
) -> Tuple[Dict[str, Any], str]:
    """Randomly dispatch one request to one of the local vLLM endpoints."""
    import random
    import requests

    url = random.choice(base_urls)

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(
        url.rstrip("/") + "/chat/completions",
        json=payload,
        headers=headers,
        timeout=timeout,
    )
    resp.raise_for_status()

    return resp.json(), url


def extract_text_from_response(resp: Dict[str, Any]) -> str:
    """Extract text from OpenAI-compatible chat completion response."""
    if not resp:
        return ""

    try:
        choices = resp.get("choices") or []
        if choices:
            c0 = choices[0]
            if isinstance(c0.get("message"), dict):
                return c0["message"].get("content", "").strip()
            if isinstance(c0.get("text"), str):
                return c0.get("text", "").strip()
    except Exception:
        pass

    return str(resp)


def parse_tagged_output(text: str) -> Dict[str, Any]:
    """Parse tagged output.

    Expected:
    <s>score</s><r>reason</r><rs>revision suggestions</rs><ra>refined answer</ra>

    Also tolerates:
    <score>score</score>
    """
    out = {
        "predicted_score": None,
        "predicted_reason": None,
        "predicted_revision_suggestions": None,
        "predicted_modified_answer": None,
        "raw_output": text,
        "parse_error": False,
    }

    def extract(tag: str):
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        if open_tag in text and close_tag in text:
            return text.split(open_tag, 1)[1].split(close_tag, 1)[0].strip()
        return None

    def clean_field(x):
        if x is None:
            return None
        x = str(x).strip()
        prefixes = [
            "feedback>",
            "reason>",
            "revision suggestions>",
            "revision_suggestions>",
            "suggestions>",
        ]
        lower = x.lower()
        for p in prefixes:
            if lower.startswith(p):
                x = x[len(p):].strip()
                break
        return x

    try:
        score = extract("s")
        if score is None:
            score = extract("score")

        if score is not None:
            m = re.search(r"[0-5]", str(score))
            score = int(m.group(0)) if m else None

        out["predicted_score"] = score
        out["predicted_reason"] = clean_field(extract("r"))
        out["predicted_revision_suggestions"] = clean_field(extract("rs"))
        out["predicted_modified_answer"] = clean_field(extract("ra"))

    except Exception:
        out["parse_error"] = True
        return out

    required_fields = [
        out["predicted_score"],
        out["predicted_reason"],
        out["predicted_revision_suggestions"],
        out["predicted_modified_answer"],
    ]

    if any(v is None or str(v).strip() == "" for v in required_fields):
        out["parse_error"] = True

    return out


def run_one(
    sample: Dict[str, Any],
    base_urls: List[str],
    model: str,
    api_key: str,
    timeout: int = 30,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run inference for one sample."""
    messages = build_messages(sample)
    t0 = time.time()

    try:
        resp, endpoint = call_vllm(
            base_urls=base_urls,
            model=model,
            messages=messages,
            api_key=api_key,
            timeout=timeout,
            max_tokens=max_tokens,
        )

        text = extract_text_from_response(resp)
        parsed = parse_tagged_output(text)

        parsed.update(
            {
                "sample_id": sample.get("sample_id"),
                "index": sample.get("index"),
                "question": sample.get("question"),
                "answer": sample.get("answer"),
                "dimension_name": sample.get("dimension_name"),
                "full_score_criteria": sample.get("full_score_criteria"),
                "score_criteria": sample.get("score_criteria"),
                "model": model,
                "endpoint": endpoint,
                "latency": time.time() - t0,
                "raw_response": resp,
            }
        )

        return parsed

    except Exception as e:
        return {
            "sample_id": sample.get("sample_id"),
            "index": sample.get("index"),
            "question": sample.get("question"),
            "answer": sample.get("answer"),
            "dimension_name": sample.get("dimension_name"),
            "model": model,
            "error": str(e),
            "latency": time.time() - t0,
        }


def prepare_samples(input_path: str, limit: int = None) -> List[Dict[str, Any]]:
    """Load samples and attach row index."""
    samples = []

    for i, row in enumerate(read_jsonl(input_path)):
        if limit is not None and i >= limit:
            break
        row["index"] = i
        samples.append(row)

    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--base_urls", default=DEFAULT_BASE_URLS)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--model", default="", help="Served vLLM model id. If omitted, fetched from /v1/models.")
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument(
        "--output-name",
        default="",
        help="Output JSONL filename. If omitted, uses vllm_feedbackbench_outputs__{model}.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    base_urls = [u.strip() for u in args.base_urls.split(",") if u.strip()]
    if not base_urls:
        raise ValueError("No valid base_urls were provided.")

    model = resolve_model(base_urls, args.model, timeout=args.timeout)
    samples = prepare_samples(args.input, limit=args.limit)

    os.makedirs(args.outdir, exist_ok=True)

    output_name = args.output_name.strip()
    if not output_name:
        output_name = f"vllm_feedbackbench_outputs__{safe_filename(model)}.jsonl"

    outpath = os.path.join(args.outdir, output_name)

    if os.path.exists(outpath):
        if args.overwrite:
            os.remove(outpath)
        else:
            raise FileExistsError(
                f"Output file already exists: {outpath}\n"
                f"Use --overwrite to replace it, or pass --output-name to choose another filename."
            )

    print(f"[model] {model}")
    print(f"[base_urls] {base_urls}")
    print(f"[samples] {len(samples)}")
    print(f"[workers] {args.workers}")
    print(f"[output] {outpath}")

    ok_count = 0
    error_count = 0
    parse_fail_count = 0
    batch: List[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                run_one,
                sample,
                base_urls,
                model,
                args.api_key,
                args.timeout,
                args.max_tokens,
            ): sample
            for sample in samples
        }

        progress = (
            tqdm(total=len(futures), desc="vLLM Feedback-Bench", ncols=120)
            if tqdm is not None
            else None
        )

        try:
            for fut in as_completed(futures):
                res = fut.result()
                batch.append(res)

                if res.get("error"):
                    error_count += 1
                elif res.get("parse_error"):
                    parse_fail_count += 1
                else:
                    ok_count += 1

                if len(batch) >= 20:
                    append_jsonl(outpath, batch)
                    batch = []

                if progress is not None:
                    progress.update(1)
                    progress.set_postfix(
                        ok=ok_count,
                        error=error_count,
                        parse_fail=parse_fail_count,
                    )
                else:
                    done = ok_count + error_count + parse_fail_count
                    if done % 100 == 0 or done == len(samples):
                        print(
                            f"[progress] {done}/{len(samples)} "
                            f"ok={ok_count} error={error_count} parse_fail={parse_fail_count}"
                        )

        finally:
            if progress is not None:
                progress.close()

    if batch:
        append_jsonl(outpath, batch)

    print(
        f"[summary] total={len(samples)} "
        f"ok={ok_count} "
        f"error={error_count} "
        f"parse_fail={parse_fail_count}"
    )
    print(f"Wrote outputs to {outpath}")


if __name__ == "__main__":
    main()