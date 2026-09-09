"""M11 Reporting Schemas for SatQuery AI.

Adheres to SIH26167 and AGENTS.md Rule 8 (Standard Result Contract), Rule 11 (Confidence Principle),
Rule 12 (Auditable Execution Trace), and M11 Evidence Packaging requirements.

Packages existing outputs produced by M7 (orchestration), M8 (fusion/consistency),
M9 (system confidence), and specialist models into a unified, machine-readable,
auditable analyst report.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.agent.schema import (
    BoundingBox,
    ConfidenceBreakdown,
    ConsistencyReport,
    ConsistencyStatus,
    EvidenceConflict,
    TraceStep,
    ZonalStatistic,
)


class ReportMetadata(BaseModel):
    """Administrative and provenance envelope for an analyst report."""
    report_id: str = Field(description="Unique report identifier")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="UTC generation timestamp in ISO-8601 format",
    )
    mission: str = Field(
        default="Smart India Hackathon 2026 / SIH26167",
        description="Governing mission identifier",
    )
    organization: str = Field(
        default="Indian Space Research Organisation (ISRO)",
        description="Domain authority",
    )
    app_version: str = Field(default="0.3.0", description="SatQuery AI release version")
    execution_time_ms: int = Field(default=0, description="Total analysis runtime in milliseconds")


class RasterMetadataSummary(BaseModel):
    """Structured summary of an ingested raster input."""
    filename: str
    role: str = Field(description="Input role: 'primary', 'secondary', 't1', 't2', 'optical', 'sar'")
    dimensions: Optional[Tuple[int, int]] = Field(default=None, description="(width, height) in pixels")
    band_count: Optional[int] = Field(default=None, description="Number of spectral/polarimetric bands")
    crs: Optional[str] = Field(default=None, description="Coordinate Reference System string or EPSG code")
    resolution: Optional[Tuple[float, float]] = Field(default=None, description="Spatial resolution (dx, dy) in meters/units")
    modality: Optional[str] = Field(default=None, description="Sensor modality (optical, sar, multispectral)")
    bounds: Optional[Tuple[float, float, float, float]] = Field(default=None, description="Geographic bounds [minx, miny, maxx, maxy]")


class InputSummary(BaseModel):
    """Consolidated summary of all inputs submitted for analysis."""
    query: str = Field(description="Original natural-language query")
    input_count: int = Field(default=1, description="Number of image inputs provided")
    images: List[RasterMetadataSummary] = Field(default_factory=list, description="Metadata for each input raster")
    temporal_ordering: Optional[str] = Field(default=None, description="Chronological relationship for multi-date inputs")
    spatial_overlap_percentage: Optional[float] = Field(default=None, description="Estimated spatial overlap across inputs (0-100%)")


class VisualEvidenceReference(BaseModel):
    """Pointer to a concrete visual artifact generated during analysis."""
    id: str = Field(description="Artifact reference ID (e.g., 'vis_mask_001')")
    type: str = Field(description="Visual type: 'mask', 'overlay', 'composite', 'source_image', 'preview'")
    label: str = Field(description="Human-readable title for UI/document rendering")
    path_or_url: str = Field(description="Relative web URL or static file path to preview")
    description: str = Field(description="Analytical explanation of what the visual depicts")
    format: str = Field(default="image/png", description="MIME type or image format")


class QuantitativeStatistics(BaseModel):
    """Consolidated quantitative metrics extracted across specialists."""
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Key-value numerical indicators (pixel counts, hectares, %)")
    zonal_statistics: List[ZonalStatistic] = Field(default_factory=list, description="Formal zonal statistics from spatial evidence")
    summary_notes: List[str] = Field(default_factory=list, description="Interpreted quantitative bullet points")


class ConfidenceSummary(BaseModel):
    """Defensible confidence disclosure adhering strictly to Milestone M9."""
    score: float = Field(ge=0.0, le=1.0, description="Heuristic system confidence score (M9)")
    level: str = Field(description="Tier: HIGH, MEDIUM, LOW, UNSUPPORTED")
    is_calibrated_probability: bool = Field(
        default=False,
        description="Always False: Heuristic score; NOT statistically calibrated probability (Rule 11)",
    )
    method: str = Field(
        default="evidence-weighted confidence heuristic",
        description="Defensible multi-factor scoring methodology",
    )
    supporting_factors: List[str] = Field(default_factory=list, description="Observable factors reinforcing confidence")
    warnings: List[str] = Field(default_factory=list, description="Dampening factors or operational penalties")
    breakdown: Optional[ConfidenceBreakdown] = Field(default=None, description="Full M9 factor breakdown")


class ConsistencySummary(BaseModel):
    """Observable multi-source consistency disclosure adhering to Milestone M8."""
    status: str = Field(description="Consistency tier: CONSISTENT, PARTIALLY_CONSISTENT, UNCERTAIN, CONTRADICTORY, INSUFFICIENT_EVIDENCE")
    is_gated: bool = Field(default=False, description="True if answer was modified by the reliability gater")
    gating_action: str = Field(default="allow", description="Action taken: allow, qualify, flag_contradiction, state_insufficient")
    conflicts_count: int = Field(default=0, description="Total conflicts detected across evidence sources")
    conflicts: List[EvidenceConflict] = Field(default_factory=list, description="Detailed list of detected evidence discrepancies")
    summary_narrative: str = Field(default="Evidence is consistent across invoked specialists.")


class SpecialistProvenanceEntry(BaseModel):
    """Audit record mapping an analytical claim or finding to its originating tool."""
    evidence_id: str = Field(description="Linked EvidenceItem or specialist output ID")
    source_specialist: str = Field(description="Specialist identifier (e.g. 'CHANGE_DETECT', 'RS_GROUND', 'OPTICAL_SAR_FUSION')")
    model_name: Optional[str] = Field(default=None, description="Specialist model name")
    version: Optional[str] = Field(default="1.0.0", description="Model version")
    claim: str = Field(description="Factual claim or finding produced")
    evidence_type: str = Field(description="Evidence classification")
    input_files: List[str] = Field(default_factory=list, description="Input files consumed by this specialist")
    execution_stage: str = Field(default="specialist_execution", description="Pipeline stage where evidence was generated")
    raw_confidence: Optional[float] = Field(default=None, description="Raw model confidence prior to system synthesis")
    provenance_details: Dict[str, Any] = Field(default_factory=dict, description="Additional telemetry (sensor, CRS, resolution)")


class AnalystReport(BaseModel):
    """Canonical M11 Structured Analyst Report.
    
    Unifies task intent, inputs, factual answers, quantitative statistics,
    visual evidence links, M8 consistency, M9 system confidence, and an
    auditable execution trace into an immutable presentation document.
    """
    metadata: ReportMetadata
    task: str = Field(description="Canonical task identifier")
    answer: str = Field(description="Grounded, evidence-backed answer")
    input_summary: InputSummary
    visual_evidence: List[VisualEvidenceReference] = Field(default_factory=list)
    statistics: QuantitativeStatistics
    confidence: ConfidenceSummary
    consistency: ConsistencySummary
    provenance: List[SpecialistProvenanceEntry] = Field(default_factory=list)
    execution_trace: List[TraceStep] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize report to formatted machine-readable JSON."""
        return self.model_dump_json(indent=indent)

    def to_markdown(self) -> str:
        """Serialize report to human-readable Markdown briefing."""
        from backend.reports.markdown import generate_markdown_report
        return generate_markdown_report(self)
