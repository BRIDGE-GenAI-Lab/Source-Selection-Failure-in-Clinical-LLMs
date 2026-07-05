"""
Load and validate cases from JSON file.
"""

import json
from pathlib import Path
from typing import Any

from .schemas import Case


def load_cases(filepath: str | Path) -> list[Case]:
    """
    Load cases from JSON file and validate structure.
    
    Args:
        filepath: Path to the JSON file containing case list.
        
    Returns:
        List of validated Case objects.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If validation fails.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Input file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(raw_data, dict):
        # Check for nested structure: {"cases_with_real_tools": {"cases": [...]}}
        if "cases_with_real_tools" in raw_data:
            items = raw_data["cases_with_real_tools"].get("cases", [])
        elif "cases" in raw_data:
            items = raw_data["cases"]
        else:
            raise ValueError("Expected 'cases' or 'cases_with_real_tools' key in dict")
    elif isinstance(raw_data, list):
        # Old format or nested batches
        items = []
        for item in raw_data:
            if isinstance(item, dict) and "cases" in item:
                items.extend(item["cases"])
            else:
                items.append(item)
    else:
        raise ValueError("Expected a JSON list or dict with 'cases' key")
    
    cases = []
    errors = []
    
    for i, item in enumerate(items):
        try:
            # Ensure eval structure exists (for new format compatibility)
            _ensure_eval_structure(item)
            
            # Validate required fields exist
            _validate_required_fields(item, i)
            
            # Parse into Case model
            case = Case.model_validate(item)
            cases.append(case)
            
        except Exception as e:
            errors.append(f"Case {i} ({item.get('case_id', 'unknown')}): {e}")
    
    if errors:
        error_summary = "\n".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_summary += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(f"Validation errors:\n{error_summary}")
    
    return cases


def _ensure_eval_structure(item: dict) -> None:
    """Add eval structure if missing, inferring preferred_tool from tools."""
    if "eval" not in item:
        item["eval"] = {}
    
    if "preferred_tool" not in item["eval"]:
        # Tool T is always the "true" tool, S is the sham
        item["eval"]["preferred_tool"] = "T"
    
    if "sham_trap_type" not in item["eval"]:
        # Try to get sham_type from Tool S metadata
        tools = item.get("tools", {})
        tool_s = tools.get("S", {})
        tool_s_meta = tool_s.get("metadata", {})
        sham_type = tool_s_meta.get("sham_type", "unknown")
        item["eval"]["sham_trap_type"] = sham_type


def _validate_required_fields(item: dict[str, Any], index: int) -> None:
    """Validate that required fields exist in a case dict."""
    required_paths = [
        ("case_id",),
        ("tools", "T", "excerpt_text"),
        ("tools", "S", "excerpt_text"),
        ("eval", "preferred_tool"),
    ]
    
    for path in required_paths:
        obj = item
        for key in path:
            if not isinstance(obj, dict) or key not in obj:
                path_str = ".".join(path)
                raise ValueError(f"Missing required field: {path_str}")
            obj = obj[key]


def load_cases_limited(filepath: str | Path, max_cases: int | None = None) -> list[Case]:
    """
    Load cases with optional limit.
    
    Args:
        filepath: Path to the JSON file.
        max_cases: Maximum number of cases to load (None for all).
        
    Returns:
        List of validated Case objects.
    """
    cases = load_cases(filepath)
    
    if max_cases is not None and max_cases > 0:
        cases = cases[:max_cases]
    
    return cases
