from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from tqdm import tqdm  # Import tqdm for progress bar

# Standard category definitions (fixed)
STANDARD_CATEGORIES = {
    "subjective": {
        "name": "subjective",
        "display_name": "Subjective (Subjective Dimensions)",
        "description": (
            "These dimensions evaluate the subjective quality of responses, "
            "without comparing to standard answers, but based on human judgment criteria"
        ),
        "examples": ["Language fluency", "Expression naturalness", "Readability", "Comprehensibility"],
    },
    "objective": {
        "name": "objective",
        "display_name": "Objective (Objective Dimensions)",
        "description": (
            "These dimensions evaluate the objective accuracy of responses, "
            "requiring comparison with standard answers, facts, or reality"
        ),
        "examples": ["Factual correctness", "Data accuracy", "Answer completeness"],
    },
    "derived_constraint": {
        "name": "derived_constraint",
        "display_name": "derived_constraint (Derived Constraint Dimensions)",
        "description": (
            "These dimensions have no direct ground truth answers, but need to be verified "
            "through context and logical reasoning"
        ),
        "examples": ["Logical consistency", "Reasoning chain completeness", "Internal coherence", "Argument rigor"],
    },
}

DEFAULT_SYSTEM_MESSAGE = (
    "You are a professional LLM evaluation expert, skilled in analyzing chain-of-thought quality dimensions."
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    data = _read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
        return data["data"]
    raise ValueError("Unsupported input JSON format; expected list or JSONL.")


def load_custom_categories(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Category definitions file not found: {path}")
    data = _read_json(path)
    return data.get("custom_categories", {}) if isinstance(data, dict) else {}


def build_prompt(
    question: str,
    answer: str,
    standard_categories: Dict[str, Dict[str, Any]],
    custom_categories: Dict[str, Dict[str, Any]],
) -> str:
    category_text_parts = []
    category_num = 1

    for _, category_def in standard_categories.items():
        display_name = category_def.get("display_name", "")
        description = category_def.get("description", "")
        examples = category_def.get("examples", [])
        examples_text = ", ".join(examples) if examples else "(Please think based on data characteristics)"
        category_text_parts.append(
            f"{category_num}. **{display_name}** - {description}\n   - Examples: {examples_text}, etc."
        )
        category_num += 1

    for _, category_def in custom_categories.items():
        display_name = category_def.get("display_name", "")
        description = category_def.get("description", "")
        examples = category_def.get("examples", [])
        examples_text = ", ".join(examples) if examples else "(Please think based on data characteristics)"
        category_text_parts.append(
            f"{category_num}. **{display_name}** - {description}\n   - Examples: {examples_text}, etc."
        )
        category_num += 1

    categories_text = "\n\n".join(category_text_parts)

    json_format_parts = []
    for category_key in standard_categories.keys():
        json_format_parts.append(
            f'''    "{category_key}": {{
        "sub_dimensions": [
            {{
                "name": "Dimension Name 1",
                "full_score_criteria": "Description of criteria for full score on this dimension"
            }},
            {{
                "name": "Dimension Name 2",
                "full_score_criteria": "Description of criteria for full score on this dimension"
            }}
        ]
    }}'''
        )

    for category_key in custom_categories.keys():
        json_format_parts.append(
            f'''    "{category_key}": {{
        "sub_dimensions": [
            {{
                "name": "Dimension Name 1",
                "full_score_criteria": "Description of criteria for full score on this dimension"
            }}
        ]
    }}'''
        )

    json_format = "{\n" + ",\n".join(json_format_parts) + "\n}"

    prompt = f"""Please analyze the following question and answer, and identify specific sub-dimensions that need to be evaluated under each category based on the following category framework.

Question: {question}

Answer: {answer}

**Analysis Framework (Category Definitions):**
{categories_text}

**Requirements:**
- Based on this specific question and answer, think about what sub-dimensions need to be evaluated under each major category
- Only list dimensions that are actually needed in the current sample. If a category is not applicable in the current sample, sub_dimensions can be an empty array
- Dimension names should be specific and actionable
- Provide full-score criteria for each dimension, describing the conditions that must be met for full score on this dimension

**Output Requirements:**
Please output in JSON format as follows:
{json_format}

Output only JSON, do not add any other explanations.
"""
    return prompt


def _http_json(method: str, url: str, payload: Optional[Dict[str, Any]], timeout: float) -> Dict[str, Any]:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = Request(url, data=data, headers=headers, method=method.upper())
    with urlopen(req, timeout=timeout) as resp:
        content = resp.read().decode("utf-8")
    return json.loads(content)


def fetch_model_id(host: str, port: int, timeout: float) -> Optional[str]:
    try:
        url = f"http://{host}:{port}/v1/models"
        data = _http_json("GET", url, None, timeout)
        if isinstance(data, dict):
            models = data.get("data", [])
            if models and isinstance(models[0], dict) and "id" in models[0]:
                return models[0]["id"]
    except Exception:
        return None
    return None


def call_llm(
    prompt: str,
    host: str,
    ports: List[int],
    preferred_port_idx: int,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    retry_delay: float,
    system_message: str,
) -> Tuple[str, int]:
    for attempt in range(max_retries):
        port_idx = (preferred_port_idx + attempt) % len(ports)
        port = ports[port_idx]
        url = f"http://{host}:{port}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        try:
            data = _http_json("POST", url, payload, timeout)
            content = data["choices"][0]["message"]["content"]
            return content, port
        except (HTTPError, URLError, KeyError, IndexError, json.JSONDecodeError) as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"LLM request failed after {max_retries} attempts: {exc}") from exc


def parse_analysis_result(result_text: str) -> Dict[str, Any]:
    json_match = re.search(r"\{[\s\S]*\}", result_text)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {"raw_output": result_text, "parse_error": "Failed to parse JSON, returning raw output"}


def extract_dimensions(analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
    dimensions_list: List[Dict[str, str]] = []
    if "parse_error" in analysis_result or "raw_output" in analysis_result:
        return dimensions_list

    for category_name, category_info in analysis_result.items():
        if not isinstance(category_info, dict):
            continue
        sub_dimensions = category_info.get("sub_dimensions", [])
        for dim in sub_dimensions:
            if isinstance(dim, dict):
                dim_name = dim.get("name", "")
                criteria = dim.get("full_score_criteria", "")
            elif isinstance(dim, str):
                dim_name = dim
                criteria = ""
            else:
                continue
            if dim_name:
                dimensions_list.append(
                    {
                        "dimension_name": dim_name,
                        "full_score_criteria": criteria,
                        "category": category_name,
                    }
                )
    return dimensions_list


def write_json_output(path: Path, data: List[Dict[str, Any]]) -> None:
    if path.suffix.lower() == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        return
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel local-GPU inference for evaluation dimensions")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset path (JSON/JSONL) or a directory containing multiple files",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path (single file mode only)")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (directory mode). Defaults to <input_dir>/out",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="_dimensions.json",
        help="Output filename suffix for directory mode",
    )
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--ports", type=str, default="8008,8009,8010,8011", help="Comma-separated ports")
    parser.add_argument("--host", type=str, default="localhost", help="Host for local LLM servers")
    parser.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--model", type=str, default=None, help="Model name for OpenAI-compatible server")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Retry delay seconds")
    parser.add_argument("--raw-output", type=str, default=None, help="Optional raw output path (single file)")
    parser.add_argument(
        "--raw-output-dir",
        type=str,
        default=None,
        help="Optional raw output directory (directory mode)",
    )
    parser.add_argument("--category-definitions", type=str, default=None, help="Custom category definitions JSON")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_MESSAGE, help="System prompt")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    raw_output_path = Path(args.raw_output) if args.raw_output else None
    ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
    if not ports:
        raise ValueError("No ports provided.")

    def _iter_input_files(path: Path) -> List[Path]:
        if path.is_file():
            return [path]
        if not path.exists():
            raise FileNotFoundError(f"Input path not found: {path}")
        files = []
        for suffix in (".json", ".jsonl"):
            files.extend(sorted(path.glob(f"*{suffix}")))
        return files

    input_files = _iter_input_files(input_path)
    if not input_files:
        raise RuntimeError(f"No JSON/JSONL files found in: {input_path}")

    custom_categories = load_custom_categories(Path(args.category_definitions)) if args.category_definitions else {}

    model = args.model
    if not model:
        for port in ports:
            model = fetch_model_id(args.host, port, timeout=5.0)
            if model:
                break
    if not model:
        model = "local-model"

    num_workers = args.num_workers or len(ports)
    num_workers = max(1, num_workers)

    def _resolve_output_paths(in_file: Path) -> Tuple[Path, Optional[Path]]:
        if input_path.is_file():
            if not output_path:
                raise ValueError("--output is required when --input is a file")
            return output_path, raw_output_path
        out_dir = Path(args.out_dir) if args.out_dir else input_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{in_file.stem}{args.output_suffix}"
        if args.raw_output_dir:
            raw_dir = Path(args.raw_output_dir)
        elif raw_output_path:
            raw_dir = raw_output_path
        else:
            raw_dir = None
        raw_file = None
        if raw_dir:
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_file = raw_dir / f"{in_file.stem}.jsonl"
        return out_file, raw_file

    def _process_one(index: int, sample: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        question = sample.get("question", "")
        answer = sample.get("answer", "")
        prompt = build_prompt(question, answer, STANDARD_CATEGORIES, custom_categories)
        try:
            result_text, port = call_llm(
                prompt=prompt,
                host=args.host,
                ports=ports,
                preferred_port_idx=index % len(ports),
                model=model,
                temperature=args.temperature,
                timeout=args.timeout,
                max_retries=args.max_retries,
                retry_delay=args.retry_delay,
                system_message=args.system_prompt,
            )
            parsed = parse_analysis_result(result_text)
            dimensions = extract_dimensions(parsed)
            output_item = {
                "question": question,
                "answer": answer,
                "evaluation_dimensions": dimensions,
            }
            raw_item = {
                "index": index,
                "port": port,
                "raw_output": result_text,
                "parse_error": parsed.get("parse_error"),
            }
            return index, output_item, raw_item
        except Exception as exc:
            output_item = {
                "question": question,
                "answer": answer,
                "evaluation_dimensions": [],
                "error": str(exc),
            }
            raw_item = {
                "index": index,
                "port": None,
                "raw_output": None,
                "parse_error": str(exc),
            }
            return index, output_item, raw_item

    for file_index, in_file in enumerate(input_files, 1):
        dataset = load_dataset(in_file)
        if args.sample_size is not None:
            dataset = dataset[: min(args.sample_size, len(dataset))]

        results: List[Optional[Dict[str, Any]]] = [None] * len(dataset)
        raw_records: List[Dict[str, Any]] = []

        out_file, raw_file = _resolve_output_paths(in_file)

        print("=" * 60)
        print(f"[{file_index}/{len(input_files)}] Processing: {in_file}")
        print(f"Total samples: {len(dataset)}")
        print(f"Ports: {ports}")
        print(f"Workers: {num_workers}")
        print(f"Model: {model}")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_one, idx, sample) for idx, sample in enumerate(dataset)]
            completed = 0
            # Adding progress bar using tqdm
            with tqdm(total=len(dataset), desc="Processing", ncols=100, dynamic_ncols=False) as pbar:
                for fut in as_completed(futures):
                    idx, output_item, raw_item = fut.result()
                    results[idx] = output_item
                    if raw_file:
                        raw_records.append(raw_item)
                    completed += 1
                    pbar.update(1)  # Update progress bar by 1 for each completed task

        final_results = [item for item in results if item is not None]
        out_file.parent.mkdir(parents=True, exist_ok=True)
        write_json_output(out_file, final_results)
        print(f"Saved: {out_file}")

        if raw_file:
            with raw_file.open("w", encoding="utf-8") as f:
                for item in raw_records:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Raw outputs: {raw_file}")


if __name__ == "__main__":
    main()
