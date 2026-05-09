"""Command-line entry for iterative answer improvement."""

import argparse
from pathlib import Path

try:
    from ..common.llm_client import list_providers
    from .iterative_generator import IterativeGenerator
    from .one_shot_generator import OneShotGenerator
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.llm_client import list_providers
    from src.iteration_generation.iterative_generator import IterativeGenerator
    from src.iteration_generation.one_shot_generator import OneShotGenerator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ChainCritic - Answer improvement using dimension scores"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="one-shot",
        choices=["iterative", "one-shot"],
        help="Improvement strategy: 'iterative' (fix dimension by dimension) or 'one-shot' (fix all at once)",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset file path (JSON or JSONL) with ratings",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (JSON or JSONL)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=list_providers(),
        help=f"LLM provider name. Supported providers: {', '.join(list_providers())}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to process (default: all)",
    )
    parser.add_argument(
        "--max-score",
        type=float,
        default=5.0,
        help="Maximum score for each dimension (default 5.0)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Only improve dimensions with score < min-score (default: max-score)",
    )
    parser.add_argument(
        "--max-dimensions",
        type=int,
        default=None,
        help="Maximum number of dimensions to iterate per sample",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="score_asc",
        choices=["score_asc", "score_desc", "original"],
        help="Dimension order for iteration (one-shot also uses this ordering)",
    )
    parser.add_argument(
        "--exclude-no-score",
        action="store_true",
        help="Skip dimensions without a score",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature parameter for LLM (default 0.2)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds (between dimensions for iterative, or between samples for one-shot)",
    )
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Store intermediate answers",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Parallel workers across samples (default 1)",
    )
    parser.add_argument(
        "--local-host",
        type=str,
        default="localhost",
        help="Host for local OpenAI-compatible servers",
    )
    parser.add_argument(
        "--local-ports",
        type=str,
        default="",
        help="Comma-separated local ports, e.g. 8000,8001,8002,8003",
    )
    parser.add_argument(
        "--local-model",
        type=str,
        default=None,
        help="Model name for local endpoints. If omitted, fetch from /v1/models.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Timeout for local requests in seconds",
    )
    parser.add_argument(
        "--disable-local",
        action="store_true",
        help="Disable local endpoint mode and only use API provider",
    )
    parser.add_argument(
        "--rerate-after-iteration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-score the final iterated answer and output final_ratings/final_overall_score",
    )
    parser.add_argument(
        "--rating-temperature",
        type=float,
        default=None,
        help="Temperature used for re-rating final answers (default: same as --temperature)",
    )
    parser.add_argument(
        "--output-mode",
        type=str,
        default="simplified",
        choices=["simplified", "full"],
        help="Output format: simplified(question/answer/modified_answer/modified_dimension_scores/per_dimension_results) or full",
    )

    args = parser.parse_args()
    local_ports = []
    if args.local_ports.strip():
        local_ports = [int(p.strip()) for p in args.local_ports.split(",") if p.strip()]

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}")
        return

    if args.strategy == "one-shot" and OneShotGenerator is None:
        print("Error: one-shot strategy requested, but one_shot_generator.py is missing.")
        print("Please add src/iteration_generation/one_shot_generator.py or use --strategy iterative.")
        return

    print(f"\n[Initializing] Using provider: {args.provider or 'auto-detect'}")
    print(f"[Strategy] Selected strategy: {args.strategy.upper()}")

    init_args = {
        "provider": args.provider,
        "max_score": args.max_score,
        "local_host": args.local_host,
        "local_ports": local_ports,
        "local_model": args.local_model,
        "request_timeout": args.request_timeout,
        "prefer_local": not args.disable_local,
    }

    if args.strategy == "one-shot":
        generator = OneShotGenerator(**init_args)
    else:
        generator = IterativeGenerator(**init_args)

    print("\n[Processing] Improving dataset answers...")

    output_file = generator.generate_dataset(
        input_file=str(input_path),
        output_file=args.output,
        sample_size=args.sample_size,
        min_score_to_improve=args.min_score,
        max_dimensions=args.max_dimensions,
        order=args.order,
        include_no_score=not args.exclude_no_score,
        temperature=args.temperature,
        delay_between_requests=args.delay,
        keep_intermediate=args.keep_intermediate,
        num_workers=max(1, args.num_workers),
        rerate_after_iteration=args.rerate_after_iteration,
        rating_temperature=args.rating_temperature,
        output_mode=args.output_mode,
    )

    print(f"\nIteration complete! Output saved to: {output_file}")


if __name__ == "__main__":
    main()
