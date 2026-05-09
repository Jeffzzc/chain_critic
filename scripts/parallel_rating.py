import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen

from openai import OpenAI

# Add src directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.common.llm_client import DEFAULT_SYSTEM_MESSAGE
from src.rating.rater import LLMRater


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
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    system_message: str,
) -> str:
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as exc:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                continue
            raise RuntimeError(f"LLM request failed after {max_retries} attempts: {exc}") from exc


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    """Load dataset from JSON/JSONL file."""
    if path.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_results(output_path: Path, results: List[Dict[str, Any]]) -> None:
    """Save rating results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Results saved to {output_path}")


def build_output_path(
    input_file: Path,
    output_dir: Path,
    output_suffix: str,
    start_line: int,
    end_line: int,
) -> Path:
    """Create a unique output file path for each shard."""
    suffix = output_suffix or "_scores.json"
    if "{start}" in suffix or "{end}" in suffix:
        try:
            suffix = suffix.format(start=start_line, end=end_line)
        except Exception as exc:
            raise ValueError("Invalid output_suffix format.") from exc
        return output_dir / f"{input_file.stem}{suffix}"

    if not (suffix.startswith("_") or suffix.startswith(".")):
        suffix = f"_{suffix}"

    return output_dir / f"{input_file.stem}_lines_{start_line}_{end_line}{suffix}"


def _build_output_item(
    sample: Dict[str, Any],
    dimensions: List[Dict[str, Any]],
    ratings: List[Dict[str, Any]],
    overall_score: Optional[float],
    error: Optional[str] = None,
    raw_output: Optional[str] = None,
) -> Dict[str, Any]:
    item = {
        "question": sample.get("question", ""),
        "answer": sample.get("answer", ""),
        "evaluation_dimensions": dimensions,
        "ratings": ratings,
        "overall_score": overall_score,
    }
    if error:
        item["error"] = error
    if raw_output:
        item["raw_output"] = raw_output
    return item


def rate_single_sample(
    index: int,
    sample: Dict[str, Any],
    rater: LLMRater,
    client: OpenAI,
    model: str,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    system_message: str,
) -> Tuple[int, Dict[str, Any]]:
    dimensions = sample.get("evaluation_dimensions", [])
    if not dimensions:
        return index, _build_output_item(sample, [], [], None)

    prompt = rater._build_single_sample_prompt(sample, dimensions)
    try:
        result_text = call_llm(
            client=client,
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            retry_delay=retry_delay,
            system_message=system_message,
        )
        parsed = rater._parse_rating_result(result_text)
        if "parse_error" in parsed or "raw_output" in parsed:
            return index, _build_output_item(
                sample,
                dimensions,
                [],
                None,
                error=parsed.get("parse_error", "Failed to parse LLM output"),
                raw_output=parsed.get("raw_output"),
            )
        ratings = rater._extract_ratings_from_result(parsed, dimensions)
        overall_score = rater._compute_overall_score(ratings)
        return index, _build_output_item(sample, dimensions, ratings, overall_score)
    except Exception as exc:
        return index, _build_output_item(sample, dimensions, [], None, error=str(exc))


def parallel_rating(
    input_file: Path,
    output_dir: Path,
    num_workers: int,
    sample_size: Optional[int],
    gpu_count: int,
    ports: List[int],
    output_suffix: str,
    host: str,
    model: Optional[str],
    timeout: float,
    temperature: float,
    max_retries: int,
    retry_delay: float,
    system_prompt: str,
    start_line: int,
    end_line: int,
) -> None:
    """Rate a dataset shard in parallel."""
    rater = LLMRater(provider=None)

    dataset = load_dataset(input_file)
    total_samples = len(dataset)
    if start_line >= total_samples:
        print(f"Start line {start_line} is out of range for dataset size {total_samples}. Skipping.")
        return

    end_line = min(end_line, total_samples - 1)
    subset = dataset[start_line : end_line + 1]
    if sample_size is not None:
        subset = subset[: min(sample_size, len(subset))]

    print(f"Total samples to rate: {len(subset)}")
    if not subset:
        print("No samples in this shard. Skipping output.")
        return

    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive.")
    if not ports:
        raise ValueError("ports list is empty.")
    if gpu_count > len(ports):
        raise ValueError("gpu_count cannot exceed the number of ports.")

    if not model:
        for port in ports:
            model = fetch_model_id(host, port, timeout=min(5.0, timeout))
            if model:
                break
    if not model:
        model = os.getenv("OPENAI_MODEL", "local-model")

    clients_by_port = {
        port: OpenAI(api_key="EMPTY", base_url=f"http://{host}:{port}/v1", timeout=timeout)
        for port in ports
    }

    results: List[Optional[Dict[str, Any]]] = [None] * len(subset)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for i, sample in enumerate(subset):
            gpu_id = i % gpu_count
            port = ports[gpu_id]
            client = clients_by_port[port]
            futures.append(
                executor.submit(
                    rate_single_sample,
                    i,
                    sample,
                    rater,
                    client,
                    model,
                    temperature,
                    max_retries,
                    retry_delay,
                    system_prompt,
                )
            )

        from tqdm import tqdm

        with tqdm(total=len(subset), desc="Processing Samples", ncols=100, dynamic_ncols=False) as pbar:
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result
                pbar.update(1)

    final_results = [item for item in results if item is not None]
    if not final_results:
        print("No results produced for this shard. Skipping output.")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = build_output_path(input_file, output_dir, output_suffix, start_line, end_line)
    save_results(output_file, final_results)


def _parse_ports(ports_str: str) -> List[int]:
    ports = [int(p.strip()) for p in ports_str.split(",") if p.strip()]
    return ports


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel LLM Rating on Dataset")

    # Input and output paths
    parser.add_argument("--input", type=str, required=True, help="Input dataset file path (JSON/JSONL)")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for results")

    # Parallel worker count
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for parallel processing")
    parser.add_argument("--sample-size", type=int, default=None, help="Number of samples to rate. If None, rate all.")

    # GPU count
    parser.add_argument("--gpu-count", type=int, default=8, help="Number of GPUs (typically 8 for H20)")

    # Shard range
    parser.add_argument("--start-line", type=int, required=True, help="Start line for the task")
    parser.add_argument("--end-line", type=int, required=True, help="End line for the task")

    # Ports and output naming
    parser.add_argument("--ports", type=str, required=True, help="Comma-separated list of ports")
    parser.add_argument("--output-suffix", type=str, required=True, help="Suffix for the output file")
    parser.add_argument("--host", type=str, default="localhost", help="Host for local LLM servers")
    parser.add_argument("--model", type=str, default=None, help="Model name for OpenAI-compatible server")
    parser.add_argument("--timeout", type=float, default=120.0, help="Request timeout seconds")
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per request")
    parser.add_argument("--retry-delay", type=float, default=3.0, help="Retry delay seconds")
    parser.add_argument("--system-prompt", type=str, default=DEFAULT_SYSTEM_MESSAGE, help="System prompt")

    args = parser.parse_args()

    input_file = Path(args.input)
    output_dir = Path(args.out_dir)
    ports = _parse_ports(args.ports)

    parallel_rating(
        input_file=input_file,
        output_dir=output_dir,
        num_workers=args.num_workers,
        sample_size=args.sample_size,
        gpu_count=args.gpu_count,
        ports=ports,
        output_suffix=args.output_suffix,
        host=args.host,
        model=args.model,
        timeout=args.timeout,
        temperature=args.temperature,
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        system_prompt=args.system_prompt,
        start_line=args.start_line,
        end_line=args.end_line,
    )
