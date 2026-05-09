"""Command-line script for generating chain-of-thought answers using LLM"""

import argparse
from pathlib import Path

try:
    from .generator import DataGenerator
    from ..common.llm_client import list_providers
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.generation.generator import DataGenerator
    from src.common.llm_client import list_providers


def main():
    """Main function for data generation"""
    parser = argparse.ArgumentParser(
        description="ChainCritic - Generate chain-of-thought answers for datasets using LLM"
    )
    
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset file path (JSONL format)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path. If not specified, auto-generates from input filename and provider"
    )
    
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=list_providers(),
        help=f"LLM provider name. Supported providers: {', '.join(list_providers())}"
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of samples to process (default: process all)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature parameter for LLM (default: 0.3)"
    )
    
    parser.add_argument(
        "--no-reference-format",
        action="store_true",
        help="Do not use reference answer format in prompt"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between API requests (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    # Initialize generator
    print(f"\n[Initializing] Using provider: {args.provider or 'auto-detect'}")
    generator = DataGenerator(provider=args.provider)
    
    # Generate dataset
    print(f"\n[Generating] Processing dataset...")
    output_file = generator.generate_dataset(
        input_file=str(input_path),
        output_file=args.output,
        sample_size=args.sample_size,
        use_reference_format=not args.no_reference_format,
        temperature=args.temperature,
        delay_between_requests=args.delay
    )
    
    print(f"\n✓ Generation complete! Output saved to: {output_file}")


if __name__ == "__main__":
    main()

