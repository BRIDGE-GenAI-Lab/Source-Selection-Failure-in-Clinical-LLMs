"""
Sanitize tools to remove leaky fields and randomize A/B mapping.

CRITICAL: This module ensures that the model never sees:
- is_sham
- trap_type
- eval
- tool_name, tool_type, family
- doc_id prefixes that reveal sham status (e.g., "ALT_")
"""

import random
from typing import Literal

from .schemas import Case, SanitizedTool, ToolMapping, RawTool
from .utils import normalize_doc_id


class ToolSanitizer:
    """
    Sanitizes tools for a case, randomizing the A/B mapping.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize with a random seed for reproducibility.
        
        Args:
            seed: Random seed for A/B mapping randomization.
        """
        self.rng = random.Random(seed)
    
    def sanitize_case(self, case: Case) -> tuple[SanitizedTool, SanitizedTool, ToolMapping]:
        """
        Sanitize tools for a case and create A/B mapping.
        
        Args:
            case: The case containing T and S tools.
            
        Returns:
            Tuple of (tool_a, tool_b, mapping) where:
            - tool_a: Sanitized tool assigned to A
            - tool_b: Sanitized tool assigned to B
            - mapping: Which underlying tool (T/S) maps to A/B
        """
        # Randomly decide mapping
        if self.rng.random() < 0.5:
            # A=T, B=S
            mapping = ToolMapping(A="T", B="S")
            tool_a = self._sanitize_tool(case.tools.T, "A")
            tool_b = self._sanitize_tool(case.tools.S, "B")
        else:
            # A=S, B=T
            mapping = ToolMapping(A="S", B="T")
            tool_a = self._sanitize_tool(case.tools.S, "A")
            tool_b = self._sanitize_tool(case.tools.T, "B")
        
        return tool_a, tool_b, mapping
    
    def _sanitize_tool(self, raw_tool: RawTool, tool_id: Literal["A", "B"]) -> SanitizedTool:
        """
        Create a sanitized tool payload from a raw tool.
        
        Removes/handles:
        - is_sham, trap_type, tool_name, tool_type, family
        - Any doc_id prefix that reveals sham status
        
        Normalizes:
        - doc_id to DOC-{sha256[:12]}
        - Handles new format field names
        """
        meta = raw_tool.metadata
        
        # Handle new format field names (guideline_title -> document_title, etc.)
        organization = meta.organization or "Unknown Organization"
        document_title = meta.document_title or meta.guideline_title or "Unknown Document"
        version = meta.version or meta.version_label or "Unknown"
        published_date = meta.published_date or (str(meta.publication_year) if meta.publication_year else "Unknown")
        retrieved_at = meta.retrieved_at or "Unknown"
        sha256 = meta.sha256 or "no-hash"
        citation_style = meta.citation_style or "AMA"
        
        # Normalize doc_id - use sha256 if doc_id is missing
        doc_id_value = meta.doc_id or sha256
        
        return SanitizedTool(
            tool_id=tool_id,
            organization=organization,
            document_title=document_title,
            version=version,
            published_date=published_date,
            retrieved_at=retrieved_at,
            doc_id=normalize_doc_id(doc_id_value),
            sha256=sha256,
            citation_style=citation_style,
            excerpt_text=raw_tool.excerpt_text,
        )


def sanitize_for_case(
    case: Case,
    rng: random.Random
) -> tuple[SanitizedTool, SanitizedTool, ToolMapping]:
    """
    Functional interface for sanitizing a single case.
    
    Args:
        case: The case to sanitize.
        rng: Random number generator for reproducibility.
        
    Returns:
        Tuple of (tool_a, tool_b, mapping).
    """
    # Randomly decide mapping
    if rng.random() < 0.5:
        mapping = ToolMapping(A="T", B="S")
        tool_a = _sanitize_single_tool(case.tools.T, "A")
        tool_b = _sanitize_single_tool(case.tools.S, "B")
    else:
        mapping = ToolMapping(A="S", B="T")
        tool_a = _sanitize_single_tool(case.tools.S, "A")
        tool_b = _sanitize_single_tool(case.tools.T, "B")
    
    return tool_a, tool_b, mapping


def _sanitize_single_tool(raw_tool: RawTool, tool_id: Literal["A", "B"]) -> SanitizedTool:
    """Sanitize a single tool, handling both old and new metadata formats."""
    meta = raw_tool.metadata
    
    # Handle new format field names (guideline_title -> document_title, etc.)
    organization = meta.organization or "Unknown Organization"
    document_title = meta.document_title or meta.guideline_title or "Unknown Document"
    version = meta.version or meta.version_label or "Unknown"
    published_date = meta.published_date or (str(meta.publication_year) if meta.publication_year else "Unknown")
    retrieved_at = meta.retrieved_at or "Unknown"
    sha256 = meta.sha256 or "no-hash"
    citation_style = meta.citation_style or "AMA"
    
    # Normalize doc_id - use sha256 if doc_id is missing
    doc_id_value = meta.doc_id or sha256
    
    return SanitizedTool(
        tool_id=tool_id,
        organization=organization,
        document_title=document_title,
        version=version,
        published_date=published_date,
        retrieved_at=retrieved_at,
        doc_id=normalize_doc_id(doc_id_value),
        sha256=sha256,
        citation_style=citation_style,
        excerpt_text=raw_tool.excerpt_text,
    )


def verify_no_leaky_fields(tool: SanitizedTool) -> list[str]:
    """
    Verify that a sanitized tool contains no leaky fields.
    
    Returns:
        List of any problems found (empty if clean).
    """
    problems = []
    
    # Check that doc_id is normalized (no ALT_ prefix, etc.)
    if tool.doc_id.startswith("ALT_"):
        problems.append("doc_id contains ALT_ prefix")
    
    if not tool.doc_id.startswith("DOC-"):
        problems.append("doc_id not properly normalized (should start with DOC-)")
    
    # Ensure tool_id is valid
    if tool.tool_id not in ("A", "B"):
        problems.append(f"Invalid tool_id: {tool.tool_id}")
    
    return problems
