"""Evidence Normalization and Fusion Subsystem for SatQuery AI (Milestone M8).

Adheres to SIH26167 and AGENTS.md:
- Rule 1 (Evidence-first remote sensing AI analyst)
- Rule 8 (Standard Result Contract)
- Rule 18 (Evidence Provenance preservation)
- Rule 10 (Multi-source evidence collection)

Normalizes heterogeneous specialist outputs into standardized EvidenceItem
records and aggregates them into unified EvidenceBundle payloads while
preserving source identity, spatial geometry, temporal context, and raw specialist confidence.
"""

import logging
from typing import Any, Dict, List, Optional
import uuid

from backend.agent.schema import (
    EvidenceBundle,
    EvidenceItem,
    ModalityType,
    SpecialistOutput,
)

logger = logging.getLogger(__name__)


class EvidenceNormalizer:
    """Normalizes specialist outputs into standard EvidenceItem records."""

    @staticmethod
    def _create_item_id(prefix: str = "ev") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @classmethod
    def normalize_vqa(
        cls,
        output: SpecialistOutput,
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Normalizes single-image VQA output."""
        meta = metadata or {}
        items: List[EvidenceItem] = []

        answer_text = output.answer_text or "No VQA answer generated."
        items.append(
            EvidenceItem(
                item_id=cls._create_item_id("vqa"),
                source_specialist="RS_VQA",
                evidence_type="visual_narrative",
                modality=ModalityType.OPTICAL,
                claim=f"Visual assessment: {answer_text}",
                region_id=None,
                geometry=None,
                metrics={},
                specialist_confidence=output.confidence,
                is_uncertain=output.confidence < 0.50,
                uncertainty_rationale="Low VQA model confidence" if output.confidence < 0.50 else None,
                provenance={
                    "query": query,
                    "filename": meta.get("filename", "unknown"),
                    "specialist_id": output.specialist_id,
                    "execution_time_ms": output.execution_time_ms,
                },
            )
        )
        return items

    @classmethod
    def normalize_grounding(
        cls,
        output: SpecialistOutput,
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Normalizes text-guided spatial grounding output."""
        meta = metadata or {}
        items: List[EvidenceItem] = []

        for b in output.evidence.boxes:
            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("box"),
                    source_specialist="RS_GROUND",
                    evidence_type="bounding_box",
                    modality=ModalityType.OPTICAL,
                    claim=f"Localized '{b.label}' at pixel bounds [{b.coordinates_pixel[0]}, {b.coordinates_pixel[1]}, {b.coordinates_pixel[2]}, {b.coordinates_pixel[3]}].",
                    region_id=b.box_id,
                    geometry={
                        "coordinates_pixel": list(b.coordinates_pixel),
                        "coordinates_normalized": list(b.coordinates_normalized),
                        "geojson": b.geojson,
                    },
                    metrics={
                        "confidence": b.confidence,
                        "model_score": b.model_score or b.confidence,
                        "is_degenerate": b.is_degenerate,
                    },
                    specialist_confidence=b.confidence,
                    is_uncertain=b.is_degenerate or b.confidence < 0.35,
                    uncertainty_rationale=b.degenerate_reason if b.is_degenerate else (
                        "Low detector confidence" if b.confidence < 0.35 else None
                    ),
                    provenance={
                        "query": query,
                        "filename": meta.get("filename", "unknown"),
                        "box_id": b.box_id,
                        "crs": meta.get("crs"),
                    },
                )
            )

        # If zero boxes detected for referring expression, emit negative evidence item
        if not output.evidence.boxes:
            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("ground_empty"),
                    source_specialist="RS_GROUND",
                    evidence_type="negative_detection",
                    modality=ModalityType.OPTICAL,
                    claim=f"No spatial regions matching referring expression '{query}' were detected.",
                    region_id=None,
                    geometry=None,
                    metrics={"detected_boxes_count": 0},
                    specialist_confidence=output.confidence,
                    is_uncertain=True,
                    uncertainty_rationale="Zero spatial candidates satisfied detection thresholds.",
                    provenance={"query": query, "filename": meta.get("filename", "unknown")},
                )
            )

        return items

    @classmethod
    def normalize_change_detection(
        cls,
        output: SpecialistOutput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Normalizes bi-temporal change detection output (TinyCD / CVA)."""
        meta = metadata or {}
        params = output.parameters_used
        items: List[EvidenceItem] = []

        ch_px = params.get("changed_pixels", 0)
        ch_ratio = params.get("change_ratio_pct", 0.0)
        clusters = params.get("total_clusters", 0)

        # 1. Spatial mask summary item
        items.append(
            EvidenceItem(
                item_id=cls._create_item_id("cd_mask"),
                source_specialist="CHANGE_DETECT",
                evidence_type="spatial_mask",
                modality=ModalityType.BITEMPORAL,
                claim=f"Binary change detection detected {ch_px:,} changed pixels ({ch_ratio:.2f}% of AOI) across {clusters} spatial clusters.",
                region_id=None,
                geometry=None,
                metrics={
                    "changed_pixels": ch_px,
                    "change_ratio_pct": ch_ratio,
                    "total_clusters": clusters,
                    "width": params.get("width"),
                    "height": params.get("height"),
                },
                specialist_confidence=output.confidence,
                is_uncertain=params.get("diagnostics", {}).get("is_extreme_change_ratio", False),
                uncertainty_rationale="Extreme change ratio detected (potential seasonal/illumination shift)"
                if params.get("diagnostics", {}).get("is_extreme_change_ratio", False)
                else None,
                provenance={
                    "filename_t1": meta.get("filename_t1", "unknown"),
                    "filename_t2": meta.get("filename_t2", "unknown"),
                    "candidate": params.get("candidate", "tinycd"),
                    "mask_path": params.get("mask_path"),
                },
            )
        )

        # 2. Zonal statistics / physical area item
        for stat in output.evidence.statistics:
            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("stat"),
                    source_specialist="CHANGE_DETECT",
                    evidence_type="zonal_statistics",
                    modality=ModalityType.BITEMPORAL,
                    claim=f"Physical measurement: {stat.display_name} = {stat.value:.2f} {stat.unit}.",
                    region_id=None,
                    geometry=None,
                    metrics={"metric_name": stat.metric_name, "value": stat.value, "unit": stat.unit},
                    specialist_confidence=output.confidence,
                    is_uncertain=False,
                    provenance={
                        "metric_name": stat.metric_name,
                        "crs": meta.get("crs", "UTM/projected"),
                    },
                )
            )

        # 3. Discrete spatial regions
        for r in output.evidence.regions:
            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("reg"),
                    source_specialist="CHANGE_DETECT",
                    evidence_type="spatial_cluster",
                    modality=ModalityType.BITEMPORAL,
                    claim=f"Detected change cluster '{r.region_name}' at pixel bbox {r.bbox_pixel}.",
                    region_id=r.region_name,
                    geometry={"bbox_pixel": list(r.bbox_pixel) if r.bbox_pixel else None},
                    metrics={
                        "area_pixels": r.details.get("area_pixels", 0),
                        "cluster_index": r.details.get("cluster_index", 0),
                    },
                    specialist_confidence=r.confidence or output.confidence,
                    is_uncertain=False,
                    provenance={
                        "region_name": r.region_name,
                        "filename_t1": meta.get("filename_t1"),
                        "filename_t2": meta.get("filename_t2"),
                    },
                )
            )

        return items

    @classmethod
    def normalize_change_vqa(
        cls,
        output: SpecialistOutput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Normalizes grounded semantic change interpretation output."""
        meta = metadata or {}
        items: List[EvidenceItem] = []
        interp = output.evidence.semantic_interpretation

        if not interp:
            return items

        # 1. Overall semantic interpretation item
        items.append(
            EvidenceItem(
                item_id=cls._create_item_id("vqa_summary"),
                source_specialist="CHANGE_VQA",
                evidence_type="semantic_interpretation",
                modality=ModalityType.BITEMPORAL,
                claim=f"Semantic change interpretation: {interp.summary} (Direction: {interp.temporal_direction}, Predominant: {interp.predominant_transition}).",
                region_id=None,
                geometry=None,
                metrics={
                    "temporal_direction": interp.temporal_direction,
                    "predominant_transition": interp.predominant_transition,
                    "semantic_uncertainty": interp.semantic_uncertainty,
                    "transition_count": len(interp.transitions),
                },
                specialist_confidence=round(1.0 - interp.semantic_uncertainty, 4),
                is_uncertain=interp.semantic_uncertainty > 0.40 or interp.temporal_direction == "uncertain",
                uncertainty_rationale="High semantic model uncertainty" if interp.semantic_uncertainty > 0.40 else None,
                provenance={
                    "supporting_model": interp.supporting_model,
                    "supporting_regions": interp.supporting_regions,
                    "filename_t1": meta.get("filename_t1"),
                    "filename_t2": meta.get("filename_t2"),
                },
            )
        )

        # 2. Individual transitions
        for t in interp.transitions:
            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("trans"),
                    source_specialist="CHANGE_VQA",
                    evidence_type="semantic_transition",
                    modality=ModalityType.BITEMPORAL,
                    claim=f"Transition '{t.from_class} -> {t.to_class}': {t.description}.",
                    region_id=t.region_id,
                    geometry={"bbox_pixel": list(t.bbox_pixel) if t.bbox_pixel else None},
                    metrics={
                        "from_class": t.from_class,
                        "to_class": t.to_class,
                        "semantic_confidence": t.semantic_confidence,
                    },
                    specialist_confidence=t.semantic_confidence,
                    is_uncertain=t.is_uncertain,
                    uncertainty_rationale="Ambiguous transition signatures" if t.is_uncertain else None,
                    provenance={
                        "transition_id": t.transition_id,
                        "evidence_support": t.evidence_support,
                        "region_id": t.region_id,
                    },
                )
            )

        return items

    @classmethod
    def normalize_optical_sar(
        cls,
        output: SpecialistOutput,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[EvidenceItem]:
        """Normalizes cross-modal Optical + SAR joint analysis output."""
        meta = metadata or {}
        params = output.parameters_used
        items: List[EvidenceItem] = []

        # 1. Joint classification summary
        dist = params.get("class_distribution", {})
        items.append(
            EvidenceItem(
                item_id=cls._create_item_id("opt_sar_joint"),
                source_specialist="OPTICAL_SAR_FUSION",
                evidence_type="cross_modal_joint_class",
                modality=ModalityType.CROSS_MODAL,
                claim=output.answer_text or "Cross-modal Optical + SAR feature fusion completed.",
                region_id=None,
                geometry=None,
                metrics={
                    "class_distribution": dist,
                    "predominant_class": max(dist, key=dist.get) if dist else "unknown",
                },
                specialist_confidence=output.confidence,
                is_uncertain=dist.get("unknown", 0.0) > 25.0,
                uncertainty_rationale="Substantial cross-modal ambiguity (>25% unknown class area)"
                if dist.get("unknown", 0.0) > 25.0
                else None,
                provenance={
                    "filename_optical": meta.get("filename_optical", "unknown"),
                    "filename_sar": meta.get("filename_sar", "unknown"),
                    "method": "dual_stream_raster_fusion",
                },
            )
        )

        # 2. Complementarity report item
        comp = output.evidence.complementarity_report
        if comp:
            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("comp"),
                    source_specialist="OPTICAL_SAR_FUSION",
                    evidence_type="cross_modal_complementarity",
                    modality=ModalityType.CROSS_MODAL,
                    claim=f"Cross-sensor complementarity: Optical provides visible spectral context ({comp.optical_limitations}); SAR provides physical backscatter and all-weather penetration ({comp.sar_penetration}).",
                    region_id=None,
                    geometry=None,
                    metrics={"layer_count": len(comp.layers)},
                    specialist_confidence=output.confidence,
                    is_uncertain=False,
                    provenance={
                        "optical_layers": [l.get("layer") for l in comp.layers if l.get("sensor") == "Optical"],
                        "sar_layers": [l.get("layer") for l in comp.layers if l.get("sensor") == "SAR"],
                    },
                )
            )

        # 3. Discrete spatial regions with dual-sensor cues
        for r in output.evidence.regions:
            opt_cues = r.details.get("optical_evidence", {})
            sar_cues = r.details.get("sar_evidence", {})
            joint_cues = r.details.get("joint_evidence", {})

            if isinstance(opt_cues, dict):
                opt_desc = f"Optical lightness={opt_cues.get('mean_brightness', 0.0):.2f}, ExG={opt_cues.get('mean_exg', 0.0):.3f}"
            else:
                opt_desc = f"Optical cues: {opt_cues}"

            if isinstance(sar_cues, dict):
                sar_desc = f"SAR backscatter={sar_cues.get('mean_backscatter', 0.0):.2f}, roughness={sar_cues.get('roughness', 0.0):.3f}"
            else:
                sar_desc = f"SAR cues: {sar_cues}"

            items.append(
                EvidenceItem(
                    item_id=cls._create_item_id("opt_sar_reg"),
                    source_specialist="OPTICAL_SAR_FUSION",
                    evidence_type="cross_modal_region",
                    modality=ModalityType.CROSS_MODAL,
                    claim=f"Grounded region '{r.region_name}' ({r.label}): {opt_desc}; {sar_desc}.",
                    region_id=r.region_name,
                    geometry={"bbox_pixel": list(r.bbox_pixel) if r.bbox_pixel else None},
                    metrics={
                        "label": r.label,
                        "area_pixels": r.details.get("area_pixels", 0),
                        "optical_evidence": opt_cues,
                        "sar_evidence": sar_cues,
                        "joint_evidence": joint_cues,
                    },
                    specialist_confidence=r.confidence or output.confidence,
                    is_uncertain=r.label == "unknown",
                    uncertainty_rationale="Conflicting optical vs SAR signature" if r.label == "unknown" else None,
                    provenance={
                        "region_name": r.region_name,
                        "filename_optical": meta.get("filename_optical"),
                        "filename_sar": meta.get("filename_sar"),
                    },
                )
            )

        return items


class EvidenceFuser:
    """Aggregates and indexes normalized EvidenceItem records."""

    @classmethod
    def fuse(
        cls,
        evidence_bundle: EvidenceBundle,
        new_items: List[EvidenceItem],
    ) -> EvidenceBundle:
        """Appends new normalized items into the EvidenceBundle while deduplicating."""
        existing_ids = {item.item_id for item in evidence_bundle.fused_items}
        for item in new_items:
            if item.item_id not in existing_ids:
                evidence_bundle.fused_items.append(item)
                existing_ids.add(item.item_id)
        return evidence_bundle

    @classmethod
    def get_items_by_specialist(cls, bundle: EvidenceBundle, specialist_id: str) -> List[EvidenceItem]:
        return [i for i in bundle.fused_items if i.source_specialist == specialist_id]

    @classmethod
    def get_items_by_region(cls, bundle: EvidenceBundle, region_id: str) -> List[EvidenceItem]:
        return [i for i in bundle.fused_items if i.region_id == region_id]

    @classmethod
    def get_items_by_type(cls, bundle: EvidenceBundle, evidence_type: str) -> List[EvidenceItem]:
        return [i for i in bundle.fused_items if i.evidence_type == evidence_type]
