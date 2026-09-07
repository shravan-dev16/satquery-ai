"""Data contracts and schemas for SatQuery AI.

Defines all Pydantic models for the Standard Result Contract (Rule 8),
input validation, specialist interfaces, execution tracing, and confidence scoring.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskType(str, Enum):
    """Supported analytical remote-sensing tasks."""
    VQA = "vqa"
    CAPTION = "caption"
    GROUNDING = "grounding"
    CHANGE_DETECTION = "change_detection"
    CHANGE_VQA = "change_vqa"
    OPTICAL_SAR_ANALYSIS = "optical_sar_analysis"
    PREPROCESS = "preprocess"
    UNKNOWN = "unknown"


class ModalityType(str, Enum):
    """Supported imagery modalities."""
    OPTICAL = "optical"
    MULTISPECTRAL = "multispectral"
    SAR = "sar"
    BITEMPORAL = "bitemporal"
    CROSS_MODAL = "cross_modal"


class StepStatus(str, Enum):
    """Status of an individual pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    WARNING = "warning"
    FAILED = "failed"
    SKIPPED = "skipped"


# ---------------------------------------------------------------------------
# Validation Schemas
# ---------------------------------------------------------------------------

class ImageMetadata(BaseModel):
    """Extracted geospatial and radiometric metadata from a single raster."""
    filename: str
    format: str = "GeoTIFF"
    width: int
    height: int
    band_count: int
    crs: Optional[str] = None
    resolution: Optional[Tuple[float, float]] = None
    bounds: Optional[Tuple[float, float, float, float]] = None  # (minx, miny, maxx, maxy)
    nodata: Optional[float] = None
    dtype: str
    modality: ModalityType = ModalityType.OPTICAL
    acquisition_date: Optional[str] = None
    sensor: Optional[str] = None


class PairCompatibility(BaseModel):
    """Validation report for image pair compatibility (bi-temporal or optical-SAR)."""
    pair_supported: bool
    spatial_overlap_ratio: float = Field(ge=0.0, le=1.0)
    spatial_overlap_percentage: float = Field(ge=0.0, le=100.0)
    crs_match: bool
    reprojection_required: bool
    temporal_ordering: str = "valid"
    warnings: List[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """Comprehensive result of pre-flight validation."""
    valid: bool
    images: List[ImageMetadata] = Field(default_factory=list)
    compatibility: Optional[PairCompatibility] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Evidence Schemas
# ---------------------------------------------------------------------------

class EvidenceImage(BaseModel):
    """Visual evidence raster preview reference."""
    role: str = Field(description="Role: primary, secondary, before_image, after_image, composite")
    url: str
    width: int
    height: int
    crs: Optional[str] = None
    bounds: Optional[Tuple[float, float, float, float]] = None


class EvidenceMask(BaseModel):
    """Binary or categorical spatial mask overlay."""
    mask_id: str
    label: str
    url: str
    format: str = "image/png"
    palette: Dict[str, str] = Field(default_factory=dict, description="Value to hex color mapping")


class BoundingBox(BaseModel):
    """Spatial bounding box evidence adhering to [xmin, ymin, xmax, ymax] convention."""
    box_id: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    model_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Raw detector confidence score")
    is_degenerate: bool = Field(default=False, description="Flag indicating full-image, zero-area, or invalid bbox")
    degenerate_reason: Optional[str] = Field(default=None, description="Diagnostic reason for degeneracy")
    coordinates_normalized: Tuple[float, float, float, float] = Field(
        description="[xmin, ymin, xmax, ymax] normalized to [0, 1]"
    )
    coordinates_pixel: Tuple[int, int, int, int] = Field(
        description="[xmin, ymin, xmax, ymax] in raster pixel coordinates"
    )
    geojson: Optional[Dict[str, Any]] = Field(
        default=None, description="Projected GeoJSON Polygon geometry in geographic coordinates"
    )


class ZonalStatistic(BaseModel):
    """Quantitative measurement derived from spatial evidence."""
    metric_name: str
    display_name: str
    value: float
    unit: str


class DetectedRegion(BaseModel):
    """Semantic region of interest adhering to Grounding Output Contract."""
    region_name: str
    label: Optional[str] = None
    bbox_pixel: Optional[Tuple[float, float, float, float]] = Field(
        default=None, description="[xmin, ymin, xmax, ymax] in pixel space"
    )
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    model_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    is_degenerate: bool = Field(default=False)
    degenerate_reason: Optional[str] = None
    polygon_pixel: Optional[List[Tuple[float, float]]] = Field(
        default=None, description="Pixel polygon coordinates if semantic segmentation mask was produced"
    )
    predominant_transition: Optional[str] = None
    intensity: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ComplementarityReport(BaseModel):
    """Report detailing distinct contributions of Optical and SAR modalities."""
    optical_limitations: str
    sar_penetration: str
    structural_contrast: str
    layers: List[Dict[str, str]] = Field(default_factory=list)


class SemanticTransition(BaseModel):
    """An individual semantic land-cover or feature transition grounded in spatial evidence.
    
    Adheres to Milestone M5 contract. Explicitly separates verified physical facts
    from model interpretation. Confidence here is semantic-model uncertainty, NOT the final
    calibrated SatQuery system confidence.
    """
    transition_id: str = Field(description="Unique identifier (e.g., 'trans_001')")
    from_class: str = Field(description="Prior land-cover state, or 'unknown'/'uncertain'")
    to_class: str = Field(description="Posterior land-cover state, or 'unknown'/'uncertain'")
    description: str = Field(description="Factual narrative describing the visual transition")
    region_id: Optional[str] = Field(default=None, description="Linked spatial cluster ID (e.g., 'change_cluster_001')")
    bbox_pixel: Optional[Tuple[int, int, int, int]] = Field(default=None, description="[xmin, ymin, xmax, ymax] in raster pixels")
    semantic_confidence: float = Field(
        default=0.80, ge=0.0, le=1.0, description="Semantic-model uncertainty score; not final system confidence"
    )
    is_uncertain: bool = Field(default=False, description="Flag indicating transition is ambiguous or unverified")
    evidence_support: str = Field(
        default="visual_crop_comparison",
        description="Basis for claim: 'visual_crop_comparison', 'quantitative_mask_extent', 'spectral_difference', 'unverified'",
    )


class SemanticChangeInterpretation(BaseModel):
    """Structured semantic change interpretation payload (Milestone M5).
    
    Provides natural-language answers to: WHAT changed, WHERE it changed, HOW it changed,
    and WHAT evidence supports it.
    """
    summary: str = Field(description="Overall grounded natural language summary of the change")
    temporal_direction: str = Field(
        default="modified",
        description="Direction of change: 'increased', 'decreased', 'modified', 'no_change', or 'uncertain'",
    )
    predominant_transition: str = Field(description="Main transition summary (e.g., 'bare_land -> built-up')")
    transitions: List[SemanticTransition] = Field(default_factory=list)
    supporting_regions: List[str] = Field(default_factory=list, description="IDs of spatial clusters supporting interpretation")
    warnings: List[str] = Field(default_factory=list, description="Advisory or uncertainty warnings")
    supporting_model: str = Field(default="Qwen/Qwen2-VL-2B-Instruct", description="VLM identifier used for interpretation")
    semantic_uncertainty: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Estimated semantic uncertainty (0 = fully confident, 1 = completely uncertain)"
    )


class EvidenceBundle(BaseModel):
    """Consolidated evidence payload adhering to Rule 8 & 9."""
    images: List[EvidenceImage] = Field(default_factory=list)
    masks: List[EvidenceMask] = Field(default_factory=list)
    boxes: List[BoundingBox] = Field(default_factory=list)
    statistics: List[ZonalStatistic] = Field(default_factory=list)
    regions: List[DetectedRegion] = Field(default_factory=list)
    complementarity_report: Optional[ComplementarityReport] = None
    semantic_interpretation: Optional[SemanticChangeInterpretation] = None


# ---------------------------------------------------------------------------
# Confidence & Trace Schemas
# ---------------------------------------------------------------------------

class ConfidenceBreakdown(BaseModel):
    """Decomposition of the evidence-weighted confidence heuristic."""
    heuristic_name: str = "evidence-weighted confidence heuristic"
    input_quality_score: float = Field(ge=0.0, le=1.0)
    spatial_alignment_score: float = Field(ge=0.0, le=1.0)
    model_confidence_score: float = Field(ge=0.0, le=1.0)
    consistency_penalty: float = Field(ge=0.0, le=0.5)
    is_calibrated_probability: bool = False


class TraceStep(BaseModel):
    """Observable step in the execution trace (Rule 12)."""
    step_number: int
    step_name: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    status: StepStatus = StepStatus.COMPLETED
    details: str
    duration_ms: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ExecutionTrace(BaseModel):
    """Auditable chronological trace of system decisions and tool executions."""
    steps: List[TraceStep] = Field(default_factory=list)
    total_duration_ms: int = 0


# ---------------------------------------------------------------------------
# Specialist Model Contracts
# ---------------------------------------------------------------------------

class ModelCapability(BaseModel):
    """Declared capability of a registered specialist model (Rule 5)."""
    identifier: str
    name: str
    version: str = "1.0.0"
    task: TaskType
    supported_modalities: List[ModalityType]
    supported_input_count: List[int]
    requires_gpu: bool = False
    vram_budget_mb: int = 0
    confidence_available: bool = True
    fallback_specialist_id: Optional[str] = None


class SpecialistInput(BaseModel):
    """Uniform input to any registered specialist model."""
    task: TaskType
    query: Optional[str] = None
    primary_image_path: str
    secondary_image_path: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SpecialistOutput(BaseModel):
    """Uniform output from any registered specialist model."""
    specialist_id: str
    success: bool
    answer_text: Optional[str] = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: EvidenceBundle = Field(default_factory=EvidenceBundle)
    parameters_used: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    execution_time_ms: int = 0


# ---------------------------------------------------------------------------
# Standard Result Contract (Rule 8)
# ---------------------------------------------------------------------------

class ModelExecutionRecord(BaseModel):
    """Record of an individual model invoked during an analysis run."""
    identifier: str
    model_name: str
    version: str = "1.0.0"
    execution_time_ms: int


class StandardResultContract(BaseModel):
    """Unified result contract returned by the API (Rule 8)."""
    task: str
    status: str = "success"
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_breakdown: ConfidenceBreakdown
    evidence: EvidenceBundle
    models: List[ModelExecutionRecord] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    execution_trace: List[TraceStep] = Field(default_factory=list)
    execution_time_ms: int = 0
