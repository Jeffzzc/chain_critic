"""Data Generation Module - Generate chain-of-thought answers for questions using LLM"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    from ..common.llm_client import ask_llm
    from ..common.data_loader import DatasetLoader
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.llm_client import ask_llm
    from src.common.data_loader import DatasetLoader


class DataGenerator:
    """Data generator - Generate chain-of-thought answers using LLM"""
    
    def __init__(self, provider: Optional[str] = None):
        """Initialize data generator
        
        Args:
            provider: LLM provider name. If not provided, auto-detects from environment variables.
        """
        self.provider = provider
    
    def _build_prompt(self, question: str, reference_answer: Optional[str] = None) -> str:
        """Build prompt for generating chain-of-thought answer
        
        Args:
            question: Question to answer
            reference_answer: Optional reference answer to guide the format
        
        Returns:
            Formatted prompt string
        """
        prompt = f"""Please solve the following question step by step, showing your reasoning process clearly.

Question: {question}

Requirements:
1. Show your step-by-step reasoning process
2. For each calculation step, use the format: step description = <<calculation>>result
3. End your answer with #### followed by the final numerical answer (if applicable)
4. If the answer is not a number, end with #### followed by your final answer

"""
        
        if reference_answer:
            prompt += f"""Please follow a similar format to this example:
Example:
{reference_answer}

"""
        
        prompt += "Now please provide your answer:"
        
        return prompt
    
    def generate_answer(self, question: str, reference_answer: Optional[str] = None, temperature: float = 0.3) -> str:
        """Generate chain-of-thought answer for a single question
        
        Args:
            question: Question to answer
            reference_answer: Optional reference answer to guide the format
            temperature: Temperature parameter for LLM (default 0.3)
        
        Returns:
            Generated answer string
        """
        prompt = self._build_prompt(question, reference_answer)
        
        answer = ask_llm(
            prompt=prompt,
            provider=self.provider
        )
        
        return answer.strip()
    
    def generate_dataset(
        self,
        input_file: str,
        output_file: Optional[str] = None,
        sample_size: Optional[int] = None,
        use_reference_format: bool = True,
        temperature: float = 0.3,
        delay_between_requests: float = 0.5
    ) -> str:
        """Generate answers for entire dataset
        
        Args:
            input_file: Path to input JSONL file
            output_file: Path to output JSONL file. If None, auto-generates from input_file and provider
            sample_size: Number of samples to process. If None, process all
            use_reference_format: Whether to use reference answer format in prompt
            temperature: Temperature parameter for LLM (default 0.3)
            delay_between_requests: Delay in seconds between API requests (default 0.5)
        
        Returns:
            Path to output file
        """
        # Load input dataset
        loader = DatasetLoader()
        dataset = loader.load_from_json(input_file)
        
        # Limit sample size if specified
        if sample_size is not None:
            dataset = dataset[:min(sample_size, len(dataset))]
        
        print(f"Loaded {len(dataset)} samples from {input_file}")
        
        # Generate output file path if not specified
        if output_file is None:
            input_path = Path(input_file)
            # Use provider name or "auto" if not specified
            provider_suffix = f"_{self.provider}" if self.provider else "_auto"
            output_file = str(input_path.parent / f"{input_path.stem}{provider_suffix}{input_path.suffix}")
        
        # Process each sample
        results = []
        successful = 0
        failed = 0
        
        for idx, sample in enumerate(dataset, 1):
            question = sample.get("question", "")
            reference_answer = sample.get("answer", "") if use_reference_format else None
            
            if not question:
                print(f"Warning: Sample {idx} has no question, skipping")
                failed += 1
                continue
            
            try:
                print(f"Processing sample {idx}/{len(dataset)}...", end=" ", flush=True)
                
                # Generate answer
                generated_answer = self.generate_answer(
                    question=question,
                    reference_answer=reference_answer,
                    temperature=temperature
                )
                
                # Create output item
                result_item = {
                    "question": question,
                    "answer": generated_answer
                }
                
                # Preserve other fields from original sample if any
                for key, value in sample.items():
                    if key not in ["question", "answer"]:
                        result_item[key] = value
                
                results.append(result_item)
                successful += 1
                print(f"✓")
                
                # Delay between requests to avoid rate limiting
                if idx < len(dataset) and delay_between_requests > 0:
                    time.sleep(delay_between_requests)
                    
            except Exception as e:
                print(f"✗ Error: {e}")
                failed += 1
                # Save original sample with error marker
                error_item = sample.copy()
                error_item["generation_error"] = str(e)
                results.append(error_item)
                continue
        
        # Save results to output file
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n{'='*60}")
        print(f"Generation completed!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Output file: {output_file}")
        print(f"{'='*60}")
        
        return output_file

