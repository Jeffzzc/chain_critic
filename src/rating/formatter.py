"""Rating result formatter - Format rating results into final output format."""

from typing import List, Dict, Any
import json


class RatingFormatter:
    """Rating formatter - Convert rating results to final output format."""

    def __init__(self, rating_results: List[Dict[str, Any]]):
        """Initialize rating formatter.

        Args:
            rating_results: Rating results list.
        """
        self.rating_results = rating_results

    def format_output(self) -> List[Dict[str, Any]]:
        """Format output results.

        Returns:
            Formatted results list.
        """
        formatted_results = []

        for result in self.rating_results:
            formatted_item = {
                "question": result.get("question", ""),
                "answer": result.get("answer", ""),
                "evaluation_dimensions": result.get("evaluation_dimensions", []),
                "ratings": result.get("ratings", []),
                "overall_score": result.get("overall_score", None)
            }

            if "error" in result:
                formatted_item["error"] = result["error"]
            if "raw_output" in result:
                formatted_item["raw_output"] = result["raw_output"]

            formatted_results.append(formatted_item)

        return formatted_results

    def export_to_json(self, output_path: str):
        """Export results to JSON file.

        Args:
            output_path: Output file path.
        """
        formatted_results = self.format_output()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(formatted_results, f, ensure_ascii=False, indent=2)

    def print_summary(self):
        """Print results summary."""
        formatted_results = self.format_output()

        total_samples = len(formatted_results)
        successful_samples = sum(1 for r in formatted_results if "error" not in r)
        total_ratings = sum(len(r.get("ratings", [])) for r in formatted_results)

        all_scores = []
        for r in formatted_results:
            for rating in r.get("ratings", []):
                score = rating.get("score")
                if isinstance(score, (int, float)):
                    all_scores.append(score)

        average_score = sum(all_scores) / len(all_scores) if all_scores else None

        print("=" * 60)
        print("Rating Results Summary")
        print("=" * 60)
        print(f"Total samples: {total_samples}")
        print(f"Successful ratings: {successful_samples}")
        print(f"Failed ratings: {total_samples - successful_samples}")
        print(f"Total dimensions scored: {total_ratings}")
        if average_score is not None:
            print(f"Average score: {average_score:.2f}")
        print("=" * 60)
