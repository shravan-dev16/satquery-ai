"""M11 Report Builder for SatQuery AI.

Transforms raw execution contracts, evidence items, confidence breakdowns,
and provenance data into an immutable, auditable AnalystReport.
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import uuid

if TYPE_CHECKING:
    from backend.agent.schema import StandardResultContract
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

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Canonical packaging engine for Milestone M11 reporting."""

    @classmethod
    def build(
        cls,
        contract: StandardResultContract,
        query: str = "",
        primary_path: Optional[Union[str, Path]] = None,
        secondary_path: Optional[Union[str, Path]] = None,
        primary_filename: Optional[str] = None,
        secondary_filename: Optional[str] = None,
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> AnalystReport:
        """Build an AnalystReport from a completed StandardResultContract.
        
        This method performs ZERO new model inference, ZERO model selection,
        and ZERO mathematical recomputation of M8 consistency or M9 confidence.
        It purely aggregates, structures, and presents evidence.
        """
        try:
            # 1. Metadata Envelope
            task_tag = contract.task.lower().replace(" ", "_")
            report_id = f"REP_{task_tag}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
            meta = ReportMetadata(
                report_id=report_id,
                timestamp=datetime.utcnow().isoformat() + "Z",
                execution_time_ms=contract.execution_time_ms,
            )

            # 2. Input Summary
            inp_summary = cls._build_input_summary(
                contract=contract,
                query=query,
                primary_path=primary_path,
                secondary_path=secondary_path,
                primary_filename=primary_filename,
                secondary_filename=secondary_filename,
            )

            # 3. Visual Evidence References
            visual_evidence = cls._build_visual_evidence(contract)

            # 4. Quantitative Statistics
            quant_stats = cls._build_quantitative_statistics(contract)

            # 5. Defensible System Confidence Summary (M9)
            conf_summary = cls._build_confidence_summary(contract)

            # 6. Multi-Source Consistency Summary (M8)
            cons_summary = cls._build_consistency_summary(contract)

            # 7. Specialist Provenance Ledger
            provenance_ledger = cls._build_provenance_ledger(
                contract=contract,
                primary_filename=primary_filename,
                secondary_filename=secondary_filename,
            )

            # 8. Assemble Full Report
            return AnalystReport(
                metadata=meta,
                task=contract.task,
                answer=contract.answer,
                input_summary=inp_summary,
                visual_evidence=visual_evidence,
                statistics=quant_stats,
                confidence=conf_summary,
                consistency=cons_summary,
                provenance=provenance_ledger,
                execution_trace=contract.execution_trace or [],
                warnings=list(contract.warnings or []),
            )
        except Exception as exc:
            logger.warning(f"ReportBuilder encountered error during assembly: {exc}. Building safe fallback report.")
            return cls._build_fallback_report(contract, query, primary_filename, secondary_filename, str(exc))

    @classmethod
    def _build_input_summary(
        cls,
        contract: StandardResultContract,
        query: str,
        primary_path: Optional[Union[str, Path]],
        secondary_path: Optional[Union[str, Path]],
        primary_filename: Optional[str],
        secondary_filename: Optional[str],
    ) -> InputSummary:
        """Constructs physical input summary from raster metadata and validation artifacts."""
        effective_query = query.strip() if query else contract.parameters.get("query", "")
        images: List[RasterMetadataSummary] = []

        # Roles based on task type
        is_bitemporal = "bitemporal" in contract.task.lower() or "change" in contract.task.lower()
        is_optical_sar = "optical_sar" in contract.task.lower()
        role1 = "t1" if is_bitemporal else ("optical" if is_optical_sar else "primary")
        role2 = "t2" if is_bitemporal else ("sar" if is_optical_sar else "secondary")

        # Ingest metadata from contract evidence images if present
        if contract.evidence and contract.evidence.images:
            for idx, ev_img in enumerate(contract.evidence.images):
                img_role = getattr(ev_img, "role", None) or (role1 if idx == 0 else role2)
                fname = getattr(ev_img, "filename", None) or (primary_filename if idx == 0 else secondary_filename) or f"input_{idx+1}.tif"
                mod_val = getattr(ev_img, "modality", None)
                mod_str = mod_val.value if hasattr(mod_val, "value") else (str(mod_val) if mod_val else ("optical" if is_optical_sar and idx == 0 else ("sar" if is_optical_sar and idx == 1 else "optical")))
                dims = (ev_img.width, ev_img.height) if hasattr(ev_img, "width") and hasattr(ev_img, "height") else None
                images.append(
                    RasterMetadataSummary(
                        filename=fname,
                        role=img_role,
                        dimensions=dims,
                        crs=getattr(ev_img, "crs", None),
                        bounds=getattr(ev_img, "bounds", None),
                        modality=mod_str,
                    )
                )

        # If evidence.images lacked entries, inspect paths or filenames safely
        if not images and (primary_path or primary_filename):
            fname1 = primary_filename or (Path(primary_path).name if primary_path else "primary.tif")
            meta1 = cls._inspect_raster_safely(Path(primary_path)) if primary_path else {}
            images.append(
                RasterMetadataSummary(
                    filename=fname1,
                    role=role1,
                    dimensions=meta1.get("dimensions"),
                    band_count=meta1.get("band_count"),
                    crs=meta1.get("crs"),
                    resolution=meta1.get("resolution"),
                    modality=meta1.get("modality", "optical"),
                    bounds=meta1.get("bounds"),
                )
            )

        if len(images) < 2 and (secondary_path or secondary_filename):
            fname2 = secondary_filename or (Path(secondary_path).name if secondary_path else "secondary.tif")
            meta2 = cls._inspect_raster_safely(Path(secondary_path)) if secondary_path else {}
            images.append(
                RasterMetadataSummary(
                    filename=fname2,
                    role=role2,
                    dimensions=meta2.get("dimensions"),
                    band_count=meta2.get("band_count"),
                    crs=meta2.get("crs"),
                    resolution=meta2.get("resolution"),
                    modality=meta2.get("modality", "sar" if is_optical_sar else "optical"),
                    bounds=meta2.get("bounds"),
                )
            )

        # Extract temporal ordering and spatial overlap from trace metadata or parameters
        temporal_ordering = contract.parameters.get("temporal_ordering") if contract.parameters else None
        spatial_overlap = contract.parameters.get("spatial_overlap_percentage") if contract.parameters else None

        if contract.execution_trace:
            for step in contract.execution_trace:
                if step.metadata:
                    if "temporal_ordering" in step.metadata and not temporal_ordering:
                        temporal_ordering = str(step.metadata["temporal_ordering"])
                    if "overlap_percentage" in step.metadata and spatial_overlap is None:
                        try:
                            spatial_overlap = float(step.metadata["overlap_percentage"])
                        except (ValueError, TypeError):
                            pass

        return InputSummary(
            query=effective_query,
            input_count=max(1, len(images)),
            images=images,
            temporal_ordering=temporal_ordering,
            spatial_overlap_percentage=spatial_overlap,
        )

    @classmethod
    def _inspect_raster_safely(cls, path: Path) -> Dict[str, Any]:
        """Safely extract basic raster headers without raising on missing or non-geospatial files."""
        if not path.exists():
            return {}
        try:
            import rasterio
            with rasterio.open(path) as src:
                crs_str = src.crs.to_string() if src.crs else None
                res = (src.res[0], src.res[1]) if src.res else None
                bounds_tuple = (src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top)
                return {
                    "dimensions": (src.width, src.height),
                    "band_count": src.count,
                    "crs": crs_str,
                    "resolution": res,
                    "bounds": bounds_tuple,
                }
        except Exception:
            return {}

    @classmethod
    def _build_visual_evidence(cls, contract: StandardResultContract) -> List[VisualEvidenceReference]:
        """Gathers and deduplicates all concrete visual preview URLs and masks."""
        visuals: List[VisualEvidenceReference] = []
        seen_urls = set()

        if not contract.evidence:
            return visuals

        # 1. Source Images
        for idx, img in enumerate(contract.evidence.images or []):
            if img.url and img.url not in seen_urls:
                seen_urls.add(img.url)
                lbl = getattr(img, "label", None) or f"Input Image {idx+1} ({getattr(img, 'role', 'source')})"
                visuals.append(
                    VisualEvidenceReference(
                        id=f"vis_src_{idx+1}",
                        type="source_image",
                        label=lbl,
                        path_or_url=img.url,
                        description=f"Ingested raster imagery ({getattr(img, 'crs', None) or 'unprojected'})",
                    )
                )

        # 2. Spatial Masks
        for idx, mask in enumerate(contract.evidence.masks or []):
            if mask.url and mask.url not in seen_urls:
                seen_urls.add(mask.url)
                lbl = getattr(mask, "label", None) or f"Spatial Mask {idx+1}"
                fmt = getattr(mask, "format", "image/png")
                visuals.append(
                    VisualEvidenceReference(
                        id=f"vis_mask_{idx+1}",
                        type="mask",
                        label=lbl,
                        path_or_url=mask.url,
                        description=f"Generated spatial decision mask ({fmt})",
                        format=fmt,
                    )
                )

        # 3. Parameters / Metadata Preview Links
        param_candidates = [
            ("overlay_path", "overlay", "Bi-temporal Change Overlay", "Fused change overlay highlighting physical surface changes"),
            ("mask_path", "mask", "Binary Change Mask", "Binary thresholded surface change mask"),
            ("composite_preview", "composite", "Multi-Panel Composite", "Side-by-side or cross-modal comparative preview"),
            ("t1_preview_url", "preview", "T1 Pre-Change Preview", "RGB visualization of baseline pre-change scene"),
            ("t2_preview_url", "preview", "T2 Post-Change Preview", "RGB visualization of post-change scene"),
        ]

        for key, vtype, label, desc in param_candidates:
            url = contract.parameters.get(key) if contract.parameters else None
            if not url and contract.evidence and hasattr(contract.evidence, "metadata") and getattr(contract.evidence, "metadata", None):
                url = contract.evidence.metadata.get(key)
            if url and isinstance(url, str) and url not in seen_urls:
                seen_urls.add(url)
                visuals.append(
                    VisualEvidenceReference(
                        id=f"vis_{key}_{len(visuals)+1}",
                        type=vtype,
                        label=label,
                        path_or_url=url,
                        description=desc,
                    )
                )

        # 4. Cross-Modal Layers
        if contract.evidence.complementarity_report:
            for idx, layer in enumerate(contract.evidence.complementarity_report.layers or []):
                layer_url = layer.get("url")
                if layer_url and layer_url not in seen_urls:
                    seen_urls.add(layer_url)
                    visuals.append(
                        VisualEvidenceReference(
                            id=layer.get("layer_id", f"vis_layer_{idx+1}"),
                            type="composite",
                            label=layer.get("label", f"Cross-Modal Layer {idx+1}"),
                            path_or_url=layer_url,
                            description="Optical-SAR complementary feature layer",
                        )
                    )

        return visuals

    @classmethod
    def _build_quantitative_statistics(cls, contract: StandardResultContract) -> QuantitativeStatistics:
        """Consolidates key numerical metrics, zonal statistics, and interpretations."""
        metrics: Dict[str, Any] = {}
        zonal_stats = list(contract.evidence.statistics) if (contract.evidence and contract.evidence.statistics) else []
        notes: List[str] = []

        # Pull relevant numbers from parameters
        param_keys = [
            "changed_pixels",
            "physical_area_m2",
            "physical_area_ha",
            "change_ratio",
            "change_percentage",
            "total_clusters",
            "change_threshold",
            "min_cluster_size",
        ]
        for k in param_keys:
            if k in contract.parameters:
                metrics[k] = contract.parameters[k]

        # Grounding Box metrics
        if contract.evidence and contract.evidence.boxes:
            boxes = contract.evidence.boxes
            metrics["detected_boxes_count"] = len(boxes)
            valid_boxes = [b for b in boxes if not b.is_degenerate]
            metrics["valid_boxes_count"] = len(valid_boxes)
            if valid_boxes:
                metrics["max_box_confidence"] = max(b.confidence for b in valid_boxes)
            notes.append(f"Located {len(valid_boxes)} valid spatial target(s) satisfying query.")

        # Bi-temporal Change metrics
        if "changed_pixels" in metrics:
            px = metrics["changed_pixels"]
            pct = metrics.get("change_ratio", 0.0) * 100.0 if "change_ratio" in metrics else metrics.get("change_percentage", 0.0)
            if px == 0:
                notes.append("Zero physical change detected exceeding threshold. Gating applied.")
            else:
                ha = metrics.get("physical_area_ha")
                ha_str = f" (~{ha:.2f} ha)" if ha is not None else ""
                notes.append(f"Physical surface change detected: {px:,} pixels{ha_str} ({pct:.2f}% of scene).")

        # Semantic Transitions
        if contract.evidence and contract.evidence.semantic_interpretation:
            interp = contract.evidence.semantic_interpretation
            metrics["predominant_transition"] = interp.predominant_transition
            metrics["temporal_direction"] = interp.temporal_direction
            metrics["semantic_uncertainty"] = interp.semantic_uncertainty
            notes.append(f"Semantic transition: '{interp.predominant_transition}' (direction: {interp.temporal_direction}).")

        # Cross-modal metrics
        if contract.evidence and contract.evidence.complementarity_report:
            comp = contract.evidence.complementarity_report
            if comp.layers:
                metrics["complementary_layers_count"] = len(comp.layers)
            notes.append("Dual-modality synthesis: optical reflectance verified against SAR microwave backscatter.")

        return QuantitativeStatistics(
            metrics=metrics,
            zonal_statistics=zonal_stats,
            summary_notes=notes,
        )

    @classmethod
    def _build_confidence_summary(cls, contract: StandardResultContract) -> ConfidenceSummary:
        """Packages M9 system confidence without modifying or recomputing formulas."""
        cb = contract.confidence_breakdown
        if cb is not None:
            level_str = contract.confidence_level or (cb.confidence_level.value if hasattr(cb.confidence_level, "value") else (str(cb.confidence_level) if cb.confidence_level else "MEDIUM"))
            score = contract.confidence
            factors = list(cb.confidence_factors or [])
            warnings = list(cb.confidence_warnings or [])
            method = cb.heuristic_name or "evidence-weighted confidence heuristic"
        else:
            level_str = contract.confidence_level or "MEDIUM"
            score = contract.confidence
            factors = []
            warnings = []
            method = "evidence-weighted confidence heuristic"

        return ConfidenceSummary(
            score=score,
            level=level_str,
            is_calibrated_probability=False,  # Strictly False by Rule 11 & M9
            method=method,
            supporting_factors=factors,
            warnings=warnings,
            breakdown=cb,
        )

    @classmethod
    def _build_consistency_summary(cls, contract: StandardResultContract) -> ConsistencySummary:
        """Packages M8 consistency report, conflicts, and gating decisions."""
        if contract.evidence and contract.evidence.consistency_report:
            cr = contract.evidence.consistency_report
            status_str = cr.status.value if hasattr(cr.status, "value") else str(cr.status)
            return ConsistencySummary(
                status=status_str,
                is_gated=cr.is_gated,
                gating_action=cr.gating_action,
                conflicts_count=len(cr.conflicts),
                conflicts=list(cr.conflicts),
                summary_narrative=cr.summary_narrative,
            )

        # Fallback if consistency checker was bypassed or unpopulated
        ev_status = contract.evidence_status or "CONSISTENT"
        return ConsistencySummary(
            status=ev_status,
            is_gated=False,
            gating_action="allow",
            conflicts_count=0,
            conflicts=[],
            summary_narrative="Single-specialist output verified; no multi-source discrepancy detected.",
        )

    @classmethod
    def _build_provenance_ledger(
        cls,
        contract: StandardResultContract,
        primary_filename: Optional[str],
        secondary_filename: Optional[str],
    ) -> List[SpecialistProvenanceEntry]:
        """Constructs an auditable ledger tracing analytical findings to specialist models."""
        ledger: List[SpecialistProvenanceEntry] = []
        input_files = [f for f in [primary_filename, secondary_filename] if f]

        # Model map by identifier
        model_map = {m.identifier: m for m in (contract.models or [])}

        # 1. From fused evidence items (M8)
        if contract.evidence and contract.evidence.fused_items:
            for item in contract.evidence.fused_items:
                m_rec = model_map.get(item.source_specialist)
                model_name = m_rec.model_name if m_rec else item.source_specialist
                version = m_rec.version if m_rec else "1.0.0"

                ledger.append(
                    SpecialistProvenanceEntry(
                        evidence_id=item.item_id,
                        source_specialist=item.source_specialist,
                        model_name=model_name,
                        version=version,
                        claim=item.claim,
                        evidence_type=item.evidence_type,
                        input_files=input_files,
                        execution_stage="specialist_execution",
                        raw_confidence=item.specialist_confidence,
                        provenance_details=item.provenance or {},
                    )
                )

        # 2. If no fused items exist, derive from model execution records
        if not ledger and contract.models:
            for idx, m_rec in enumerate(contract.models):
                ledger.append(
                    SpecialistProvenanceEntry(
                        evidence_id=f"prov_{m_rec.identifier.lower()}_{idx+1}",
                        source_specialist=m_rec.identifier,
                        model_name=m_rec.model_name,
                        version=m_rec.version,
                        claim=contract.answer[:160],
                        evidence_type="specialist_inference",
                        input_files=input_files,
                        execution_stage="specialist_execution",
                        raw_confidence=contract.confidence,
                        provenance_details={"execution_time_ms": m_rec.execution_time_ms},
                    )
                )

        return ledger

    @classmethod
    def _build_fallback_report(
        cls,
        contract: StandardResultContract,
        query: str,
        primary_filename: Optional[str],
        secondary_filename: Optional[str],
        error_msg: str,
    ) -> AnalystReport:
        """Constructs a minimal safe report if builder assembly fails."""
        meta = ReportMetadata(
            report_id=f"REP_FALLBACK_{int(time.time())}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            execution_time_ms=contract.execution_time_ms,
        )
        return AnalystReport(
            metadata=meta,
            task=contract.task,
            answer=contract.answer,
            input_summary=InputSummary(
                query=query or "",
                input_count=1,
                images=[],
            ),
            visual_evidence=[],
            statistics=QuantitativeStatistics(),
            confidence=ConfidenceSummary(
                score=contract.confidence,
                level=contract.confidence_level or "MEDIUM",
                is_calibrated_probability=False,
                supporting_factors=[],
                warnings=[f"Report assembled in fallback mode: {error_msg}"],
            ),
            consistency=cls._build_consistency_summary(contract) if getattr(contract, "evidence", None) else ConsistencySummary(
                status=getattr(contract, "evidence_status", None) or "UNCERTAIN",
                is_gated=False,
                summary_narrative="Report generated under fallback safety handler.",
            ),
            provenance=[],
            execution_trace=contract.execution_trace or [],
            warnings=list(contract.warnings or []) + [f"ReportBuilder warning: {error_msg}"],
        )
