"""Result formatting module - Format analysis results into final output format"""

from typing import List, Dict, Any
import json


class ResultFormatter:
    """Result formatter - Convert analysis results to final output format"""
    
    def __init__(self, analysis_results: List[Dict[str, Any]]):
        """Initialize result formatter
        
        Args:
            analysis_results: Analysis results list, each element contains question, answer and evaluation_dimensions
        """
        self.analysis_results = analysis_results
    
    def format_output(self) -> List[Dict[str, Any]]:
        """Format output results
        
        Returns:
            Formatted results list
        """
        formatted_results = []
        
        for result in self.analysis_results:
            formatted_item = {
                "question": result.get("question", ""),
                "answer": result.get("answer", ""),
                "evaluation_dimensions": result.get("evaluation_dimensions", [])
            }
            
            # If analysis failed, keep error information
            if "error" in result:
                formatted_item["error"] = result["error"]
            
            formatted_results.append(formatted_item)
        
        return formatted_results
    
    def export_to_json(self, output_path: str):
        """Export results to JSON file
        
        Args:
            output_path: Output file path
        """
        formatted_results = self.format_output()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)
    
    def print_summary(self):
        """Print results summary"""
        formatted_results = self.format_output()
        
        total_samples = len(formatted_results)
        successful_samples = sum(1 for r in formatted_results if "error" not in r)
        total_dimensions = sum(len(r.get("evaluation_dimensions", [])) for r in formatted_results)
        
        print("=" * 60)
        print("Analysis Results Summary")
        print("=" * 60)
        print(f"Total samples: {total_samples}")
        print(f"Successful analysis: {successful_samples}")
        print(f"Failed analysis: {total_samples - successful_samples}")
        print(f"Total dimensions generated: {total_dimensions}")
        
        if formatted_results:
            print(f"\nAverage dimensions per sample: {total_dimensions / successful_samples:.2f}" if successful_samples > 0 else "")
            
        print("=" * 60)
