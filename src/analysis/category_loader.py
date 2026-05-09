"""Category definition loading module - Load and manage custom category definitions"""

from typing import Dict, Any, Optional
import json
from pathlib import Path


def load_category_definitions(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load custom category definitions from JSON file
    
    Args:
        file_path: JSON file path, format should contain "custom_categories" field
    
    Returns:
        Custom category definitions dictionary
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config.get("custom_categories", {})


def get_default_category_definitions() -> Dict[str, Dict[str, Any]]:
    """Get default custom category definitions (format_structure as example)
    
    Returns:
        Default custom category definitions dictionary
    """
    return {
        "format_structure": {
            "name": "format_structure",
            "display_name": "format_structure (Format/Structure Type)",
            "description": "These dimensions evaluate the format specifications and structural organization of responses",
            "examples": [
                "Whether the opening is clear",
                "Whether the explanation is complete",
                "Whether the summary is appropriate",
                "Whether the paragraph structure is reasonable"
            ]
        }
    }
