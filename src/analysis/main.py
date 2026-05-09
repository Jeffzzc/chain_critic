"""Main program entry - Implement complete dataset analysis workflow"""

import argparse
import json
from pathlib import Path

try:
    from ..common.data_loader import DatasetLoader
    from .analyzer import LLMAnalyzer
    from .formatter import ResultFormatter
    from ..common.llm_client import list_providers
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.data_loader import DatasetLoader
    from src.analysis.analyzer import LLMAnalyzer
    from src.analysis.formatter import ResultFormatter
    from src.common.llm_client import list_providers


def main():
    """Main function - Implement the complete workflow shown in the diagram"""
    parser = argparse.ArgumentParser(
        description="ChainCritic - LLM Chain-of-Thought Evaluation Dimension Analysis System"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input dataset file path (supports JSON array format or JSONL format, one JSON object per line)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_dimensions.json",
        help="Output evaluation dimension configuration file path"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10,
        help="Number of samples for analysis (default 10)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        choices=list_providers(),
        help=f"LLM provider name. Configuration is read from provider-specific environment variables. "
             f"Supported providers: {', '.join(list_providers())}"
    )
    parser.add_argument(
        "--category-definitions",
        type=str,
        default=None,
        help="Custom category definitions file path (JSON format, refer to category_definitions.example.json)"
    )
    
    args = parser.parse_args()
    
    # Load custom category definitions
    custom_category_definitions = None
    if args.category_definitions:
        category_def_path = Path(args.category_definitions)
        if category_def_path.exists():
            try:
                with open(category_def_path, 'r', encoding='utf-8') as f:
                    category_config = json.load(f)
                    custom_category_definitions = category_config.get("custom_categories", {})
                    print(f"✓ Loaded custom category definitions: {len(custom_category_definitions)} categories")
            except Exception as e:
                print(f"⚠ Warning: Failed to load custom category definitions: {e}, will use default settings")
        else:
            print(f"⚠ Warning: Custom category definitions file does not exist: {args.category_definitions}, will use default settings")
    

    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    loader = DatasetLoader()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    try:
        dataset = loader.load_from_json(str(input_path))
        stats = loader.get_statistics()
        print(f"✓ Successfully loaded dataset, {stats['total']} samples")
    except Exception as e:
        print(f"Error: Failed to load dataset: {e}")
        return
    
    # Step 2: LLM analysis (analyze samples one by one)
    print(f"\n[Step 2] Using LLM to analyze dataset samples one by one (sample size: {min(args.sample_size, len(dataset))})...")

    if custom_category_definitions:
        print(f"   Using custom categories: {', '.join(custom_category_definitions.keys())}")
    try:
        analyzer = LLMAnalyzer(
            custom_category_definitions=custom_category_definitions,
            provider=args.provider
        )
        analysis_results = analyzer.analyze_dataset(dataset, sample_size=args.sample_size)
        print(f"\n✓ LLM analysis completed, analyzed {len(analysis_results)} samples")
        
    except Exception as e:
        print(f"Error: LLM analysis failed: {e}")
        return
    
    # Step 3: Format and save results
    print("\n[Step 3] Formatting analysis results...")
    try:
        formatter = ResultFormatter(analysis_results)
        formatter.print_summary()
        formatter.export_to_json(args.output)
        print(f"\n✓ Analysis results saved to: {args.output}")
    except Exception as e:
        print(f"Error: Failed to save results: {e}")
        return
    
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run main program
    main()

