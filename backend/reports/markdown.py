"""Human-Readable Markdown Report Generator for SatQuery AI (Milestone M11).

Converts canonical AnalystReport into an executive briefing document suitable
for display in terminals, GitHub markdown viewers, UI modals, or export.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.reports.schema import AnalystReport


def generate_markdown_report(report: "AnalystReport") -> str:
    """Renders an AnalystReport into an executive Markdown briefing."""
    meta = report.metadata
    inp = report.input_summary
    conf = report.confidence
    cons = report.consistency
    stats = report.statistics

    lines = []

    # Title & Header
    lines.append(f"# SatQuery AI — Executive Analysis Report")
    lines.append(f"**Mission:** {meta.mission}  ")
    lines.append(f"**Authority:** {meta.organization}  ")
    lines.append(f"**Report ID:** `{meta.report_id}` | **Generated:** {meta.timestamp} | **Runtime:** {meta.execution_time_ms} ms  ")
    lines.append(f"**Engine Version:** v{meta.app_version}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Executive Summary
    lines.append("## 1. Executive Summary")
    lines.append(f"- **Task:** `{report.task}`")
    lines.append(f"- **Query:** *\"{inp.query}\"*")
    lines.append(f"- **Primary Finding:**")
    lines.append(f"  > {report.answer}")
    lines.append("")

    # 2. Defensible System Confidence & Reliability
    lines.append("## 2. System Confidence & Consistency Audit")
    conf_badge = f"**{conf.level} ({conf.score:.2f})**"
    lines.append(f"- **System Confidence:** {conf_badge} *(Heuristic multi-factor score; not calibrated probability)*")
    lines.append(f"- **Consistency Status:** `{cons.status}` (Gating action: `{cons.gating_action}`)")
    lines.append(f"- **Consistency Narrative:** {cons.summary_narrative}")

    if conf.supporting_factors:
        lines.append("")
        lines.append("### Supporting Confidence Factors")
        for factor in conf.supporting_factors:
            lines.append(f"- [x] {factor}")

    if conf.warnings or cons.conflicts:
        lines.append("")
        lines.append("### Identified Risk & Discrepancy Factors")
        for w in conf.warnings:
            lines.append(f"- [!] {w}")
        for c in cons.conflicts:
            lines.append(f"- [!] **Conflict ({c.severity.upper()} - {c.rule_violated}):** {c.description} (Sources: {', '.join(c.conflicting_sources)})")
    lines.append("")

    # 3. Input Imagery Verification
    lines.append("## 3. Ingested Imagery & Pre-Flight Validation")
    if inp.images:
        lines.append("| Role | Filename | Modality | Dimensions | CRS | Resolution |")
        lines.append("|---|---|---|---|---|---|")
        for img in inp.images:
            dims = f"{img.dimensions[0]}x{img.dimensions[1]}" if img.dimensions else "N/A"
            res = f"{img.resolution[0]:.1f}m" if img.resolution else "N/A"
            crs_val = img.crs or "Unprojected"
            mod_val = img.modality or "Optical"
            lines.append(f"| {img.role} | `{img.filename}` | {mod_val} | {dims} | {crs_val} | {res} |")
        lines.append("")
        if inp.temporal_ordering:
            lines.append(f"- **Temporal Ordering:** {inp.temporal_ordering}")
        if inp.spatial_overlap_percentage is not None:
            lines.append(f"- **Spatial Co-Registration Overlap:** {inp.spatial_overlap_percentage:.1f}%")
        lines.append("")
    else:
        lines.append("*No raster metadata recorded.*")
        lines.append("")

    # 4. Quantitative Statistics
    if stats.metrics or stats.zonal_statistics:
        lines.append("## 4. Quantitative Measurements & Spatial Statistics")
        if stats.metrics:
            lines.append("| Metric | Measured Value |")
            lines.append("|---|---|")
            for k, v in stats.metrics.items():
                fmt_val = f"{v:.4f}" if isinstance(v, float) else str(v)
                lines.append(f"| `{k}` | **{fmt_val}** |")
            lines.append("")
        if stats.zonal_statistics:
            lines.append("### Formal Zonal Indicators")
            for z in stats.zonal_statistics:
                lines.append(f"- **{z.display_name}:** {z.value} {z.unit} (`{z.metric_name}`)")
            lines.append("")

    # 5. Visual Evidence Assets
    if report.visual_evidence:
        lines.append("## 5. Visual Evidence References")
        lines.append("| ID | Type | Label | Link / URL | Analytical Description |")
        lines.append("|---|---|---|---|---|")
        for vis in report.visual_evidence:
            link = f"[{vis.label}]({vis.path_or_url})"
            lines.append(f"| `{vis.id}` | `{vis.type}` | {vis.label} | {link} | {vis.description} |")
        lines.append("")

    # 6. Specialist Provenance Ledger
    if report.provenance:
        lines.append("## 6. Specialist Model Provenance Ledger")
        lines.append("| Evidence ID | Specialist Tool | Claim / Output | Modality | Raw Conf. |")
        lines.append("|---|---|---|---|---|")
        for p in report.provenance:
            raw_c = f"{p.raw_confidence:.2f}" if p.raw_confidence is not None else "N/A"
            lines.append(f"| `{p.evidence_id}` | `{p.source_specialist}` | {p.claim} | {p.evidence_type} | {raw_c} |")
        lines.append("")

    # 7. Auditable Execution Trace
    if report.execution_trace:
        lines.append("## 7. Auditable Execution Trace")
        lines.append("| # | Step | Status | Duration | Operational Details |")
        lines.append("|---|---|---|---|---|")
        for st in report.execution_trace:
            st_name = st.step_name
            st_stat = st.status.value if hasattr(st.status, "value") else str(st.status)
            dur = f"{st.duration_ms} ms"
            lines.append(f"| {st.step_number} | `{st_name}` | {st_stat} | {dur} | {st.details} |")
        lines.append("")

    # 8. Warnings & Operational Limitations
    if report.warnings:
        lines.append("## 8. Operational Limitations & Warnings")
        for w in report.warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    lines.append("---")
    lines.append("*Generated by SatQuery AI — Autonomous Multimodal Remote Sensing Intelligence Engine.*")
    return "\n".join(lines)
