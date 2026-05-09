"""Data loading module - Process Q&A datasets"""

from typing import List, Dict, Any
import json
from pathlib import Path


class DatasetLoader:
    """Dataset loader"""
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
    
    def load_from_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON or JSONL file
        
        Supports two formats:
        1. JSON array format: [{"question": "...", "answer": "..."}, ...]
        2. JSONL format (one JSON object per line):
           {"question": "...", "answer": "..."}
           {"question": "...", "answer": "..."}
        
        Args:
            file_path: Path to JSON or JSONL file
        
        Returns:
            Dataset list
        """
        path = Path(file_path)
        
        # Determine format by file extension
        if path.suffix.lower() == '.jsonl':
            return self.load_from_jsonl(file_path)
        
        # Try to load as JSON array
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            
        # If content is empty, return empty list
        if not content:
            self.data = []
            return []
        
        # Try to parse as JSON array
        try:
            data = json.loads(content)
            if isinstance(data, list):
                self.data = data
                return data
        except json.JSONDecodeError:
            # If not a valid JSON array, try as JSONL
            pass
        
        # If not a JSON array, try as JSONL
        # Check if first line is JSON object format
        first_line = content.split('\n', 1)[0].strip()
        if first_line and first_line.startswith('{') and first_line.endswith('}'):
            # Looks like JSONL format
            return self.load_from_jsonl(file_path)
        
        # If neither matches, raise exception
        raise ValueError(f"Cannot parse file {file_path}, please ensure it's a valid JSON array or JSONL format")
    
    def load_from_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Load dataset from JSONL file (one JSON object per line)
        
        Args:
            file_path: Path to JSONL file
        
        Returns:
            Dataset list
        """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError as e:
                    raise ValueError(f"JSON parsing error at line {line_num} in file {file_path}: {e}")
        
        self.data = data
        return data
    
    def load_from_dict(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Load dataset from dictionary list
        
        Args:
            data: Dataset list, each element should contain 'question' and 'answer' fields
        
        Returns:
            Dataset list
        """
        self.data = data
        return data
    
    def get_sample(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get sample data for analysis
        
        Args:
            n: Number of samples
        
        Returns:
            Sample data list
        """
        if not self.data:
            return []
        return self.data[:min(n, len(self.data))]
    
    def get_all(self) -> List[Dict[str, Any]]:
        """Get all data"""
        return self.data
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self.data:
            return {"total": 0}
        
        return {
            "total": len(self.data),
            "sample_questions": [item.get("question", "")[:50] + "..." 
                                for item in self.data[:3]]
        }
