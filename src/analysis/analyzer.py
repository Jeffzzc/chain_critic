"""LLM Analysis Module - Analyze datasets based on category framework, identify sub-dimensions under each category"""

from typing import List, Dict, Any, Optional

try:
    from ..common.llm_client import ask_llm
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from src.common.llm_client import ask_llm


# Standard category definitions (fixed)
STANDARD_CATEGORIES = {
    "subjective": {
        "name": "subjective",
        "display_name": "Subjective (Subjective Dimensions)",
        "description": "These dimensions evaluate the subjective quality of responses, without comparing to standard answers, but based on human judgment criteria",
        # "examples": ["Language fluency", "Expression naturalness", "Readability", "Comprehensibility"]
        "examples": []
    },
    "objective": {
        "name": "objective",
        "display_name": "Objective (Objective Dimensions)",
        "description": "These dimensions evaluate the objective accuracy of responses, requiring comparison with standard answers, facts, or reality",
        # "examples": ["Factual correctness", "Data accuracy", "Answer completeness"]
        "examples": []
    },
    "derived_constraint": {
        "name": "derived_constraint",
        "display_name": "derived_constraint (Derived Constraint Dimensions)",
        "description": "These dimensions have no direct ground truth answers, but need to be verified through context and logical reasoning",
        # "examples": ["Logical consistency", "Reasoning chain completeness", "Internal coherence", "Argument rigor"]
        "examples": []
    }
}


class LLMAnalyzer:
    """LLM Analyzer - Analyze datasets based on category framework, identify sub-dimensions under each category
    
    Standard categories (fixed):
    - Subjective (Subjective Dimensions)
    - Objective (Objective Dimensions)
    - derived_constraint (Derived Constraint Dimensions)
    
    Custom categories (optional):
    - Can be passed via custom_category_definitions parameter
    - Example: format_structure (Format/Structure Type), etc.
    """
    
    def __init__(
        self, 
        custom_category_definitions: Optional[Dict[str, Dict[str, Any]]] = None,
        provider: Optional[str] = None
    ):
        """Initialize LLM analyzer
        
        Args:
            custom_category_definitions: Custom category definitions dictionary, format:
                {
                    "category_key": {
                        "name": "category_key",
                        "display_name": "Display Name",
                        "description": "Category description",
                        "examples": ["Example 1", "Example 2"]
                    }
                }
            provider: LLM provider name. If not provided, uses the first available provider from env.example
                Supported providers: openai, deepseek_v31, deepseek_r1, doubao_seed_1.6, doubao_1.5_lite_32k, doubao_1.5_pro_256k
                Configuration is read from provider-specific environment variables (e.g., DEEPSEEK_V31_API_KEY, DEEPSEEK_V31_MODEL, etc.)
        """
        self.provider = provider
        self.custom_category_definitions = custom_category_definitions or {}
    
    def analyze_single_sample(
        self,
        sample: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze a single sample data
        
        Args:
            sample: Single sample, should contain 'question' and 'answer' fields
        
        Returns:
            Analysis result for a single sample
        """
        # Build prompt for single sample
        prompt = self._build_single_sample_prompt(sample, STANDARD_CATEGORIES, self.custom_category_definitions)
        
        # Call LLM using ask_llm function
        result_text = ask_llm(
            prompt=prompt,
            provider=self.provider
        )
        
        # Parse result
        return self._parse_analysis_result(result_text)
    
    def _build_single_sample_prompt(
        self,
        sample: Dict[str, Any],
        standard_categories: Dict[str, Dict[str, Any]],
        custom_categories: Dict[str, Dict[str, Any]]
    ) -> str:
        """Build analysis prompt for single sample
        
        Args:
            sample: Single sample data
            standard_categories: Standard category definitions
            custom_categories: Custom category definitions
        """
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        
        # Build category description text
        category_text_parts = []
        category_num = 1
        
        # Add standard categories
        for category_key, category_def in standard_categories.items():
            display_name = category_def.get("display_name", category_key)
            description = category_def.get("description", "")
            examples = category_def.get("examples", [])
            examples_text = ", ".join(examples) if examples else "(Please think based on data characteristics)"
            
            category_text_parts.append(
                f"{category_num}. **{display_name}** - {description}\n"
                f"   - Examples: {examples_text}, etc."
            )
            category_num += 1
        
        # Add custom categories
        for category_key, category_def in custom_categories.items():
            display_name = category_def.get("display_name", category_key)
            description = category_def.get("description", "")
            examples = category_def.get("examples", [])
            examples_text = ", ".join(examples) if examples else "(Please think based on data characteristics)"
            
            category_text_parts.append(
                f"{category_num}. **{display_name}** - {description}\n"
                f"   - Examples: {examples_text}, etc."
            )
            category_num += 1
        
        categories_text = "\n\n".join(category_text_parts)
        
        # Build JSON output format example
        json_format_parts = []
        for category_key in standard_categories.keys():
            json_format_parts.append(f'''    "{category_key}": {{
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
    }}''')
        
        for category_key in custom_categories.keys():
            json_format_parts.append(f'''    "{category_key}": {{
        "sub_dimensions": [
            {{
                "name": "Dimension Name 1",
                "full_score_criteria": "Description of criteria for full score on this dimension"
            }}
        ]
    }}''')
        
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
    
    def analyze_dataset(
        self, 
        dataset: List[Dict[str, Any]], 
        sample_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Analyze dataset and generate evaluation dimensions for each sample
        
        Analyze each sample one by one, generate corresponding evaluation dimension list for each question-answer pair.
        
        Args:
            dataset: Dataset list, each element should contain 'question' and 'answer' fields
            sample_size: Number of samples for analysis, if None, analyze all data
        
        Returns:
            Analysis results list, each element contains question, answer and corresponding evaluation dimension list
        """
        # Get sample data
        if sample_size is not None:
            samples = dataset[:min(sample_size, len(dataset))]
        else:
            samples = dataset
        
        # Store analysis results for all samples
        results_with_samples = []
        
        # Analyze each sample one by one
        for idx, sample in enumerate(samples, 1):
            try:
                # Analyze single sample
                analysis_result = self.analyze_single_sample(sample)
                
                # Extract dimension list
                evaluation_dimensions = self._extract_dimensions_from_result(analysis_result)
                
                # Combine question, answer with dimensions
                result_item = {
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "evaluation_dimensions": evaluation_dimensions
                }
                
                results_with_samples.append(result_item)
                print(f"  Sample {idx}/{len(samples)} analysis completed, generated {len(evaluation_dimensions)} dimensions")
                
            except Exception as e:
                print(f"Warning: Sample {idx} analysis failed: {e}")
                # Even if analysis fails, keep question and answer
                results_with_samples.append({
                    "question": sample.get("question", ""),
                    "answer": sample.get("answer", ""),
                    "evaluation_dimensions": [],
                    "error": str(e)
                })
                continue
        
        return results_with_samples
    
    def _extract_dimensions_from_result(self, analysis_result: Dict[str, Any]) -> List[Dict[str, str]]:
        """Extract dimension list from analysis result
        
        Args:
            analysis_result: LLM analysis result, contains sub_dimensions under each category
        
        Returns:
            Dimension list, format: [{"dimension_name": "...", "full_score_criteria": "...", "category": "..."}, ...]
        """
        dimensions_list = []
        
        # Skip error results
        if "parse_error" in analysis_result or "raw_output" in analysis_result:
            return dimensions_list
        
        # Iterate through all categories
        for category_name, category_info in analysis_result.items():
            if not isinstance(category_info, dict):
                continue
            
            sub_dimensions = category_info.get("sub_dimensions", [])
            
            # Process each dimension
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
                    dimensions_list.append({
                        "dimension_name": dim_name,
                        "full_score_criteria": criteria,
                        "category": category_name
                    })
        
        return dimensions_list

    def _parse_analysis_result(self, result_text: str) -> Dict[str, Any]:
        """Parse analysis result returned by LLM"""
        import json
        import re
        
        # Try to extract JSON part
        json_match = re.search(r'\{[\s\S]*\}', result_text)
        if json_match:
            try:
                result = json.loads(json_match.group())
                return result
            except json.JSONDecodeError:
                pass
        
        # If parsing fails, return raw text
        return {
            "raw_output": result_text,
            "parse_error": "Failed to parse JSON, returning raw output"
        }
