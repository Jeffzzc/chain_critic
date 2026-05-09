"""Main program entry - Rate QA pairs based on evaluation dimensions."""

import argparse
from pathlib import Path

try:
    from ..common.data_loader import DatasetLoader
    from ..common.llm_client import list_providers
    from .rater import LLMRater
    from .formatter import RatingFormatter
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.data_loader import DatasetLoader
    from src.common.llm_client import list_providers
    from src.rating.rater import LLMRater
    from src.rating.formatter import RatingFormatter


def main():
    """Main function - Implement rating workflow."""
    parser = argparse.ArgumentParser(
        description="ChainCritic - Rate QA pairs using evaluation dimensions"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path (JSON or JSONL) with evaluation_dimensions"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="rating_results.json",
        help="Output rating file path"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples to rate (default 10)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=list_providers(),
        help=f"LLM provider name. Supported providers: {', '.join(list_providers())}"
    )
    parser.add_argument(
        "--max-score",
        type=int,
        default=5,
        help="Maximum score for each dimension (default 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API requests (default 0.5)"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}")
        return

    print("\n[Step 1] Loading dataset...")
    loader = DatasetLoader()
    try:
        dataset = loader.load_from_json(str(input_path))
        stats = loader.get_statistics()
        print(f"Successfully loaded dataset, {stats['total']} samples")
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        return

    print("\n[Step 2] Rating samples with LLM...")
    rater = LLMRater(provider=args.provider, max_score=args.max_score)
    try:
        rating_results = rater.rate_dataset(
            dataset,
            sample_size=args.sample_size,
            delay_between_requests=args.delay
        )
        print(f"\nLLM rating completed, rated {len(rating_results)} samples")
    except Exception as e:
        print(f"Error: LLM rating failed: {e}")
        return

    print("\n[Step 3] Formatting rating results...")
    try:
        formatter = RatingFormatter(rating_results)
        formatter.print_summary()
        formatter.export_to_json(args.output)
        print(f"\nRating results saved to: {args.output}")
    except Exception as e:
        print(f"Error: Failed to save results: {e}")
        return

    print("\n" + "=" * 60)
    print("Rating completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
