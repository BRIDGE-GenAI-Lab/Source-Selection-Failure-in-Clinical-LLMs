"""
Pydantic schemas for the safe-guideline-tooling-eval project.
"""

from typing import Literal, Optional, Any
from pydantic import BaseModel, Field


# ============================================================================
# Input Data Schemas (from JSON file)
# ============================================================================

class Demographics(BaseModel):
    """Patient demographics."""
    age: int
    sex: str
    weight_kg: Optional[float] = None


class Vitals(BaseModel):
    """Patient vital signs."""
    temp_c: Optional[float] = None
    hr: Optional[int] = None
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    rr: Optional[int] = None
    spo2: Optional[int] = None


class KeyLabs(BaseModel):
    """Key laboratory values."""
    wbc: Optional[float] = None
    hgb: Optional[float] = None
    plt: Optional[int] = None
    na: Optional[int] = None
    k: Optional[float] = None
    creatinine: Optional[float] = None
    egfr: Optional[int] = None
    lactate: Optional[float] = None
    crp: Optional[float] = None
    inr: Optional[float] = None
    trop: Optional[float | str] = None  # Can be numeric or qualitative (e.g., "Rising")
    glucose: Optional[int] = None


class ECG(BaseModel):
    """ECG findings."""
    qtc_ms: Optional[int] = None
    rhythm_note: Optional[str] = None


class ToolMetadata(BaseModel):
    """Metadata for a guideline tool (flexible schema for old and new formats)."""
    organization: Optional[str] = None
    document_title: Optional[str] = None
    guideline_title: Optional[str] = None  # New format uses this
    version: Optional[str] = None
    version_label: Optional[str] = None  # New format uses this
    publication_year: Optional[int] = None  # New format
    published_date: Optional[str] = None
    retrieved_at: Optional[str] = None
    doc_id: Optional[str] = None
    sha256: Optional[str] = None
    source_pdf_path: Optional[str] = None  # New format
    page_anchors: Optional[list] = None  # New format
    citation_style: Optional[str] = None
    source_status: Optional[str] = None  # New format: 'updated_from_pdf' or 'unmapped_no_pdf'
    notes: Optional[str] = None  # New format
    trap_type: Optional[str] = None  # Legacy: only present in sham tools
    sham_type: Optional[str] = None  # New format: sham type identifier
    modification_log: Optional[list[str]] = None  # New format: sham modifications


class RawTool(BaseModel):
    """Raw tool data from input JSON (flexible for old and new formats)."""
    # Legacy fields (optional for new format)
    tool_name: Optional[str] = None
    tool_type: Optional[str] = None
    is_sham: Optional[bool] = None
    family: Optional[str] = None
    
    # Core fields (required in both formats)
    excerpt_text: str
    metadata: ToolMetadata
    
    # New format fields
    tool_id: Optional[str] = None  # 'T' or 'S'
    safety_guardrails: Optional[list[str]] = None
    applicability: Optional[list[str]] = None
    exclusions_or_cautions: Optional[list[str]] = None


class Tools(BaseModel):
    """Container for T (true) and S (sham) tools."""
    T: RawTool
    S: RawTool


class Eval(BaseModel):
    """Evaluation ground truth."""
    preferred_tool: Literal["T", "S"]
    sham_trap_type: str


class Case(BaseModel):
    """Complete case from input JSON."""
    case_id: str
    setting: Optional[str] = None
    demographics: Optional[Demographics] = None
    chief_complaint: Optional[str] = None
    history_present_illness: Optional[str] = None
    pmh: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    vitals: Optional[Vitals] = None
    key_labs: Optional[KeyLabs] = None
    ecg: Optional[ECG] = None
    imaging: list[str] = Field(default_factory=list)
    assessment_context: Optional[str] = None
    clinical_question: Optional[str] = None
    case_flags: list[str] = Field(default_factory=list)
    tool_family: Optional[str] = None
    tools: Tools
    eval: Eval


# ============================================================================
# Sanitized Tool Schema (what the model sees)
# ============================================================================

class SanitizedTool(BaseModel):
    """
    Tool payload with leaky fields removed.
    This is what gets returned to the model when it calls guideline_a or guideline_b.
    """
    tool_id: Literal["A", "B"]
    organization: str
    document_title: str
    version: str
    published_date: str
    retrieved_at: str
    doc_id: str  # Normalized to DOC-{sha256[:12]}
    sha256: str
    citation_style: str
    excerpt_text: str


# ============================================================================
# Model Output Schema (structured output)
# ============================================================================

class ModelDecision(BaseModel):
    """
    The structured output schema the model must produce.
    Used with OpenAI's structured outputs or validated via Pydantic.
    """
    case_id: str = Field(description="The case identifier")
    selected_tool: Literal["A", "B", "unclear"] = Field(
        description="Which tool (A or B) is more trustworthy - should be A or B, unclear is fallback only"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence in the selection (0.0 to 1.0)"
    )
    trust_rationale: str = Field(
        description="Brief rationale for the trust decision (2-4 sentences)"
    )
    final_answer: str = Field(
        description="Short clinical recommendation with 'research-only; not medical advice' footer"
    )
    safety_notes: list[str] = Field(
        default_factory=list,
        description="Notes about suspicious content in tools (e.g., 'override text ignored')"
    )
    tool_calls_made: list[Literal["guideline_a", "guideline_b"]] = Field(
        default_factory=list,
        description="List of tools that were called"
    )
    contradictions_found: list[str] = Field(
        default_factory=list,
        description="Contradictions found between tools or with case facts"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="Information that would be helpful but was not available"
    )


# ============================================================================
# Result Schemas (for scoring and output)
# ============================================================================

class ToolMapping(BaseModel):
    """Maps A/B to T/S for a specific case."""
    A: Literal["T", "S"]
    B: Literal["T", "S"]


class CaseResult(BaseModel):
    """Complete result for a single case, used for scoring."""
    case_id: str
    mapping: ToolMapping
    sham_trap_type: str
    tool_payload_hashes: dict[str, str]  # {"A": sha256, "B": sha256}
    model_decision: Optional[ModelDecision] = None
    selected_tool_correct: Optional[bool] = None
    errors: list[str] = Field(default_factory=list)
    raw_responses: list[str] = Field(default_factory=list)  # Filenames of raw response files
    usage: Optional[dict[str, Any]] = None
    latency_ms: Optional[float] = None


class ExperimentConfig(BaseModel):
    """Configuration for an experiment run."""
    model: str
    seed: int
    input_file: str
    input_file_hash: str
    prompt_variant: Literal["baseline", "defended"]
    require_both_tools: bool
    max_cases: Optional[int] = None
    timestamp: str
    git_commit: Optional[str] = None


class Metrics(BaseModel):
    """Computed metrics for scoring."""
    accuracy_overall: float
    accuracy_by_trap: dict[str, float]
    unclear_rate: float
    tool_call_compliance: float
    injection_susceptibility: Optional[float] = None
    total_cases: int
    successful_cases: int
    failed_cases: int
