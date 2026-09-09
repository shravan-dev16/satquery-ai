"""M11 Reporting and Evidence Packaging Package for SatQuery AI.

Provides canonical structured AnalystReport, ReportBuilder packaging engine,
and Markdown export utilities adhering to SIH26167 and AGENTS.md rules.
"""

from backend.reports.builder import ReportBuilder
from backend.reports.markdown import generate_markdown_report
from backend.reports.schema import (
    AnalystReport,
    ConfidenceSummary,
    ConsistencySummary,
    InputSummary,
    QuantitativeStatistics,
    RasterMetadataSummary,
    ReportMetadata,
    SpecialistProvenanceEntry,
    VisualEvidenceReference,
)

__all__ = [
    "ReportBuilder",
    "AnalystReport",
    "ReportMetadata",
    "RasterMetadataSummary",
    "InputSummary",
    "VisualEvidenceReference",
    "QuantitativeStatistics",
    "ConfidenceSummary",
    "ConsistencySummary",
    "SpecialistProvenanceEntry",
    "generate_markdown_report",
]
