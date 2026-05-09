#!/usr/bin/env python
"""Run two-stage inference: dimensions then ratings."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DIM_SYSTEM_PROMPT = (
    "You are an evaluation-dimension generator. "
    "Given a question and its answer, output the evaluation dimensions with full-score criteria."
)
DIM_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Return JSON with the following format:\n"
    "{{\"evaluation_dimensions\": ["
    "{{\"dimension_name\": \"...\", \"full_score_criteria\": \"...\", \"category\": \"subjective|objective|derived_constraint\"}}"
    "]}}"
)

RATING_SYSTEM_PROMPT = (
    "You are a strict evaluator. "
    "Score the answer on each provided dimension and explain each score briefly."
)
RATING_USER_TEMPLATE = (
    "Question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Evaluation dimensions (with full-score criteria):\n{dimensions_json}\n\n"
    "Return JSON with the following format:\n"
    "{{\"ratings\": ["
    "{{\"dimension_name\": \"...\", \"score\": 0, \"reason\": \"...\", \"category\": \"...\"}}"
    "], \"overall_score\": 0}}"
)


def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    raise ValueError(f"Expected list JSON in {path}")


def save_json_or_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def build_dimension_prompt(question: str, answer: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": DIM_SYSTEM_PROMPT},
        {"role": "user", "content": DIM_USER_TEMPLATE.format(question=question, answer=answer)},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_rating_prompt(
    question: str, answer: str, dimensions: List[Dict[str, Any]], tokenizer: AutoTokenizer
) -> str:
    dimensions_json = json.dumps(dimensions, ensure_ascii=False, indent=2)
    messages = [
        {"role": "system", "content": RATING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": RATING_USER_TEMPLATE.format(
                question=question,
                answer=answer,
                dimensions_json=dimensions_json,
            ),
        },
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
        )
    gen_ids = output[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def build_model_and_tokenizer(
    model_name: str,
    dim_adapter: str,
    rating_adapter: Optional[str],
    load_in_4bit: bool,
    bf16: bool,
    fp16: bool,
    attn_implementation: str,
    trust_remote_code: bool,
) -> Tuple[PeftModel, AutoTokenizer, bool]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if bf16:
        torch_dtype = torch.bfloat16
    elif fp16:
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )

    attn_impl = None if attn_implementation == "auto" else attn_implementation
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        dtype=torch_dtype,
        device_map="auto",
        quantization_config=quant_config,
        attn_implementation=attn_impl,
    )
    if attn_impl is not None and hasattr(base_model.config, "attn_implementation"):
        base_model.config.attn_implementation = attn_impl

    model = PeftModel.from_pretrained(base_model, dim_adapter, is_trainable=False)
    has_rating_adapter = False
    if rating_adapter and rating_adapter != dim_adapter:
        model.load_adapter(rating_adapter, adapter_name="rating", is_trainable=False)
        model.set_adapter("default")
        has_rating_adapter = True
    return model, tokenizer, has_rating_adapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage inference for dimensions and ratings.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--dim-adapter", type=str, default=None)
    parser.add_argument("--rating-adapter", type=str, default=None)
    parser.add_argument("--max-new-tokens-dim", type=int, default=512)
    parser.add_argument("--max-new-tokens-rating", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--trust-remote-code", action="store_true")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    print(f"Loading input: {input_path}")
    data = load_json_or_jsonl(input_path)
    print(f"Loaded samples: {len(data)}")

    if args.adapter:
        dim_adapter = args.adapter
        rating_adapter = args.adapter
    else:
        if not args.dim_adapter:
            raise ValueError("Provide --adapter or --dim-adapter.")
        dim_adapter = args.dim_adapter
        rating_adapter = args.rating_adapter or args.dim_adapter

    print("Loading model and adapters...")
    model, tokenizer, has_rating_adapter = build_model_and_tokenizer(
        args.model_name,
        dim_adapter,
        rating_adapter,
        args.load_in_4bit,
        args.bf16,
        args.fp16,
        args.attn_implementation,
        args.trust_remote_code,
    )
    print("Model ready.")

    results = []
    iterator = data
    if tqdm is not None:
        iterator = tqdm(data, desc="Infer", unit="sample")
    else:
        print("tqdm not available; progress will be logged every 10 samples.")
    for idx, sample in enumerate(iterator, 1):
        question = sample.get("question", "").strip()
        answer = sample.get("answer", "").strip()
        if not question or not answer:
            continue

        dim_prompt = build_dimension_prompt(question, answer, tokenizer)
        dim_text = generate_text(
            model,
            tokenizer,
            dim_prompt,
            args.max_new_tokens_dim,
            args.temperature,
        )
        dim_json = extract_json(dim_text) or {}
        dimensions = dim_json.get("evaluation_dimensions", [])

        if has_rating_adapter:
            model.set_adapter("rating")
        rating_prompt = build_rating_prompt(question, answer, dimensions, tokenizer)
        rating_text = generate_text(
            model,
            tokenizer,
            rating_prompt,
            args.max_new_tokens_rating,
            args.temperature,
        )
        rating_json = extract_json(rating_text) or {}

        results.append(
            {
                "question": question,
                "answer": answer,
                "evaluation_dimensions": dimensions,
                "ratings": rating_json.get("ratings", []),
                "overall_score": rating_json.get("overall_score"),
                "raw_dimension_output": dim_text,
                "raw_rating_output": rating_text,
            }
        )

        if has_rating_adapter:
            model.set_adapter("default")

        if tqdm is None and idx % 10 == 0:
            print(f"Processed {idx}/{len(data)} samples", file=sys.stderr)

    print(f"Saving outputs: {output_path}")
    save_json_or_jsonl(output_path, results)
    print("Done.")


if __name__ == "__main__":
    main()
