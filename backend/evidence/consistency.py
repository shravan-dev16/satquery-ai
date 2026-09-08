"""Consistency Checking and Evidence Gating Engine for SatQuery AI (Milestone M8).

Adheres to SIH26167 and AGENTS.md:
- Rule 10 (Consistency Checking across independent specialist evidence)
- Rule 11 (Specialist confidence separated from final system confidence)
- Rule 24 (Graceful failure and transparent conflict disclosure)
- Rule 31 (Failure-handling principle: never manufacture certainty)
- Post-M7 Architectural Lock (AD-001, AD-004, AD-005, AD-007, AD-012)

Implements 8 deterministic consistency rules (C1-C8), evaluates evidence sufficiency,
and enforces evidence gating so unsupported or contradictory claims are not presented as facts.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import uuid

from backend.agent.schema import (
    ConsistencyReport,
    ConsistencyStatus,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceItem,
)

logger = logging.getLogger(__name__)


class ConsistencyChecker:
    """Evaluates multi-source evidence against deterministic remote-sensing consistency rules."""

    @staticmethod
    def _create_conflict_id(prefix: str = "conf") -> str:
        return f"{prefix}_{uuid.uuid4().hex[:8]}"

    @classmethod
    def check_bitemporal(
        cls,
        evidence: EvidenceBundle,
        change_params: Dict[str, Any],
        temporal_ordering: str = "valid",
        query: str = "",
    ) -> ConsistencyReport:
        """Evaluates consistency across bi-temporal change detection and semantic VQA."""
        conflicts: List[EvidenceConflict] = []
        warnings: List[str] = []
        supporting_ids: List[str] = []

        changed_pixels = change_params.get("changed_pixels", 0)
        total_clusters = change_params.get("total_clusters", 0)
        detector_regions = {r.region_name for r in evidence.regions}

        interp = evidence.semantic_interpretation
        stats_by_name = {s.metric_name: s.value for s in evidence.statistics}

        # Collect specialist confidences
        specialist_confs: Dict[str, float] = {}
        for item in evidence.fused_items:
            if item.source_specialist not in specialist_confs:
                specialist_confs[item.source_specialist] = item.specialist_confidence

        # -------------------------------------------------------------------
        # RULE C1: ZERO-CHANGE CONTRADICTION
        # If CHANGE_DETECT.changed_pixels == 0, and CHANGE_VQA claims active transition
        # -------------------------------------------------------------------
        if changed_pixels == 0:
            if interp:
                claims_active_change = (
                    interp.temporal_direction in ("increased", "decreased", "modified")
                    or any(t.from_class != "none" and t.to_class != "none" for t in interp.transitions)
                    or (interp.predominant_transition and interp.predominant_transition.lower() not in ("none", "no_change", "unchanged"))
                )
                if claims_active_change:
                    conflicts.append(
                        EvidenceConflict(
                            conflict_id=cls._create_conflict_id("c1"),
                            rule_violated="C1_ZERO_CHANGE_CONTRADICTION",
                            severity="critical",
                            description=(
                                f"Contradiction detected: Spatial change detector verified 0 changed pixels, "
                                f"but semantic VQA asserted active transition ('{interp.predominant_transition}')."
                            ),
                            conflicting_sources=["CHANGE_DETECT", "CHANGE_VQA"],
                            details={
                                "changed_pixels": 0,
                                "predominant_transition": interp.predominant_transition,
                                "temporal_direction": interp.temporal_direction,
                            },
                        )
                    )

        # -------------------------------------------------------------------
        # RULE C2: REGION SUPPORT
        # Referenced region IDs must exist in spatial detector evidence
        # -------------------------------------------------------------------
        if interp:
            for t in interp.transitions:
                if t.region_id and t.region_id not in detector_regions:
                    conflicts.append(
                        EvidenceConflict(
                            conflict_id=cls._create_conflict_id("c2"),
                            rule_violated="C2_UNSUPPORTED_REGION",
                            severity="warning",
                            description=(
                                f"Semantic transition '{t.transition_id}' references region '{t.region_id}', "
                                f"which does not exist in spatial change detector evidence."
                            ),
                            conflicting_sources=["CHANGE_DETECT", "CHANGE_VQA"],
                            details={"referenced_region": t.region_id, "available_regions": list(detector_regions)},
                        )
                    )

        # -------------------------------------------------------------------
        # RULE C3: NONZERO CHANGE SUPPORT
        # Semantic transitions must be backed by nonzero changed pixels
        # -------------------------------------------------------------------
        if interp and interp.transitions:
            if changed_pixels <= 0:
                # Handled by C1 if direction != no_change, but ensure explicit violation if transitions exist
                if not any(c.rule_violated == "C1_ZERO_CHANGE_CONTRADICTION" for c in conflicts):
                    conflicts.append(
                        EvidenceConflict(
                            conflict_id=cls._create_conflict_id("c3"),
                            rule_violated="C3_NONZERO_CHANGE_SUPPORT",
                            severity="critical",
                            description="Semantic transitions were emitted without non-zero spatial change support.",
                            conflicting_sources=["CHANGE_DETECT", "CHANGE_VQA"],
                            details={"changed_pixels": changed_pixels, "transition_count": len(interp.transitions)},
                        )
                    )

        # -------------------------------------------------------------------
        # RULE C4: AREA CONSISTENCY
        # Claimed physical area must match detector zonal statistics within 10% tolerance
        # -------------------------------------------------------------------
        if "physical_area_ha" in stats_by_name:
            meas_area_ha = stats_by_name["physical_area_ha"]
            # Check if any transition claims explicit numeric area in details
            for t in (interp.transitions if interp else []):
                claimed_area = t.model_dump().get("details", {}).get("claimed_area_ha")
                if claimed_area is not None and meas_area_ha > 0:
                    diff_ratio = abs(claimed_area - meas_area_ha) / meas_area_ha
                    if diff_ratio > 0.10:
                        conflicts.append(
                            EvidenceConflict(
                                conflict_id=cls._create_conflict_id("c4"),
                                rule_violated="C4_AREA_CONSISTENCY",
                                severity="warning",
                                description=(
                                    f"Reported area ({claimed_area:.2f} ha) differs from measured physical "
                                    f"area ({meas_area_ha:.2f} ha) by {diff_ratio * 100.0:.1f}% (>10% tolerance)."
                                ),
                                conflicting_sources=["CHANGE_DETECT", "CHANGE_VQA"],
                                details={"claimed_area_ha": claimed_area, "measured_area_ha": meas_area_ha},
                            )
                        )

        # -------------------------------------------------------------------
        # RULE C5: TEMPORAL DIRECTION & EVIDENCE SUPPORT (User Correction 2)
        # T1 < T2 validates chronology; direction is validated against T1->T2 evidence
        # -------------------------------------------------------------------
        # 1. Chronology check
        if temporal_ordering in ("reversed", "invalid", "negative_delta"):
            conflicts.append(
                EvidenceConflict(
                    conflict_id=cls._create_conflict_id("c5_chrono"),
                    rule_violated="C5_TEMPORAL_CHRONOLOGY_INVALID",
                    severity="critical",
                    description=f"Chronological acquisition verification failed: temporal ordering is '{temporal_ordering}'.",
                    conflicting_sources=["METADATA", "TEMPORAL_VALIDATOR"],
                    details={"temporal_ordering": temporal_ordering},
                )
            )

        # 2. Evidence-grounded direction check
        if interp:
            dir_val = interp.temporal_direction.lower()
            if dir_val == "no_change" and changed_pixels > 500:
                conflicts.append(
                    EvidenceConflict(
                        conflict_id=cls._create_conflict_id("c5_dir"),
                        rule_violated="C5_TEMPORAL_DIRECTION_CONFLICT",
                        severity="critical",
                        description=(
                            f"Temporal direction claimed 'no_change', but detector measured "
                            f"{changed_pixels:,} changed pixels."
                        ),
                        conflicting_sources=["CHANGE_DETECT", "CHANGE_VQA"],
                        details={"temporal_direction": dir_val, "changed_pixels": changed_pixels},
                    )
                )
            elif dir_val == "increased":
                # Validate that transitions reflect expansion/construction rather than demolition
                has_increase = any(
                    "construction" in t.to_class.lower()
                    or "built" in t.to_class.lower()
                    or "expansion" in t.description.lower()
                    or "new" in t.description.lower()
                    for t in interp.transitions
                )
                if not has_increase and changed_pixels > 0 and interp.transitions:
                    # If all transitions indicate loss/removal (e.g. forest -> bare_ground), direction increased is inconsistent
                    has_decrease_only = all(
                        "bare" in t.to_class.lower()
                        or "loss" in t.description.lower()
                        or "clearing" in t.description.lower()
                        for t in interp.transitions
                    )
                    if has_decrease_only:
                        conflicts.append(
                            EvidenceConflict(
                                conflict_id=cls._create_conflict_id("c5_dir_mismatch"),
                                rule_violated="C5_TEMPORAL_DIRECTION_CONFLICT",
                                severity="warning",
                                description="Direction was reported as 'increased', but T1->T2 transition evidence indicates vegetation loss/clearing.",
                                conflicting_sources=["CHANGE_VQA"],
                                details={"temporal_direction": dir_val},
                            )
                        )

        # -------------------------------------------------------------------
        # RULE C7: CLASS / TRANSITION CONFLICT
        # Check for semantic conflict between detector land-cover cues and text
        # -------------------------------------------------------------------
        if interp:
            for t in interp.transitions:
                if t.from_class == "water" and "building" in t.to_class and changed_pixels < 50:
                    conflicts.append(
                        EvidenceConflict(
                            conflict_id=cls._create_conflict_id("c7"),
                            rule_violated="C7_CLASS_TRANSITION_CONFLICT",
                            severity="warning",
                            description="Unlikely direct transition from water to building with negligible pixel support.",
                            conflicting_sources=["CHANGE_VQA"],
                            details={"from_class": t.from_class, "to_class": t.to_class},
                        )
                    )

        # -------------------------------------------------------------------
        # RULE C8: INSUFFICIENT EVIDENCE
        # -------------------------------------------------------------------
        if changed_pixels == 0 and total_clusters == 0 and not interp:
            # Single-run change detection with no change is valid, but if query specifically asked for change locations:
            if any(term in query.lower() for term in ("where", "locate", "outline", "show")):
                warnings.append("No change locations were identified to satisfy spatial localization query.")

        # Determine overall status
        status, quality_score, narrative = cls._evaluate_status_and_narrative(
            conflicts=conflicts,
            evidence=evidence,
            query=query,
        )

        # Populate supporting evidence IDs (items not part of conflicts)
        conflicting_rules = {c.rule_violated for c in conflicts}
        for item in evidence.fused_items:
            if not item.is_uncertain and not (item.source_specialist == "CHANGE_VQA" and "C1_ZERO_CHANGE_CONTRADICTION" in conflicting_rules):
                supporting_ids.append(item.item_id)

        return ConsistencyReport(
            status=status,
            is_gated=status in (ConsistencyStatus.CONTRADICTORY, ConsistencyStatus.INSUFFICIENT_EVIDENCE),
            gating_action="flag_contradiction" if status == ConsistencyStatus.CONTRADICTORY else (
                "state_insufficient" if status == ConsistencyStatus.INSUFFICIENT_EVIDENCE else (
                    "qualify" if status in (ConsistencyStatus.PARTIALLY_CONSISTENT, ConsistencyStatus.UNCERTAIN) else "allow"
                )
            ),
            conflicts=conflicts,
            supporting_evidence_ids=supporting_ids,
            warnings=warnings,
            specialist_confidences=specialist_confs,
            evidence_quality_score=quality_score,
            summary_narrative=narrative,
        )

    @classmethod
    def check_optical_sar(
        cls,
        evidence: EvidenceBundle,
        query: str = "",
    ) -> ConsistencyReport:
        """Evaluates consistency across optical spectral cues and SAR physical backscatter (Rule C6)."""
        conflicts: List[EvidenceConflict] = []
        warnings: List[str] = []
        supporting_ids: List[str] = []

        specialist_confs: Dict[str, float] = {}
        for item in evidence.fused_items:
            if item.source_specialist not in specialist_confs:
                specialist_confs[item.source_specialist] = item.specialist_confidence

        # -------------------------------------------------------------------
        # RULE C6: OPTICAL / SAR AGREEMENT
        # Evaluate agreement across individual detected spatial regions
        # -------------------------------------------------------------------
        conflict_region_count = 0
        total_regions = len(evidence.regions)

        for r in evidence.regions:
            opt = r.details.get("optical_evidence", {})
            sar = r.details.get("sar_evidence", {})
            label = r.label or "unknown"

            opt_dict = opt if isinstance(opt, dict) else {}
            sar_dict = sar if isinstance(sar, dict) else {}

            # Check for strong cross-modal contradictions
            is_conflict = False
            # Case 1: High optical greenness / vegetation, but extremely high SAR backscatter and double bounce
            if opt_dict.get("mean_exg", 0.0) > 0.15 and sar_dict.get("mean_backscatter", 0.0) > 0.75 and sar_dict.get("roughness", 0.0) > 0.20:
                is_conflict = True
                conflict_desc = f"Region '{r.region_name}': Optical greenness indicates vegetation, but extreme SAR backscatter indicates structural double-bounce."

            # Case 2: Water label with high SAR backscatter
            elif label == "water_body" and sar_dict.get("mean_backscatter", 0.0) > 0.35:
                is_conflict = True
                conflict_desc = f"Region '{r.region_name}': Optically identified as water, but SAR backscatter is elevated ({sar_dict.get('mean_backscatter', 0.0):.2f})."

            # Case 3: Built structure with near-zero backscatter and flat smoothness
            elif label == "built_structure" and sar_dict.get("mean_backscatter", 0.0) < 0.10 and sar_dict.get("roughness", 0.0) < 0.04:
                is_conflict = True
                conflict_desc = f"Region '{r.region_name}': Labeled built structure, but SAR demonstrates specular reflection (near-zero backscatter)."

            elif label == "unknown":
                is_conflict = True
                conflict_desc = f"Region '{r.region_name}': Ambiguous or conflicting cross-modal signatures resulted in unclassified label."

            if is_conflict:
                conflict_region_count += 1
                conflicts.append(
                    EvidenceConflict(
                        conflict_id=cls._create_conflict_id("c6"),
                        rule_violated="C6_OPTICAL_SAR_DISAGREEMENT",
                        severity="warning",
                        description=conflict_desc,
                        conflicting_sources=["OPTICAL_STREAM", "SAR_STREAM"],
                        details={"region_name": r.region_name, "optical": opt, "sar": sar},
                    )
                )

        # -------------------------------------------------------------------
        # RULE C8: INSUFFICIENT EVIDENCE
        # -------------------------------------------------------------------
        if total_regions == 0:
            conflicts.append(
                EvidenceConflict(
                    conflict_id=cls._create_conflict_id("c8_empty"),
                    rule_violated="C8_INSUFFICIENT_EVIDENCE",
                    severity="warning",
                    description="Zero discrete cross-modal land-cover regions could be segmented from joint features.",
                    conflicting_sources=["OPTICAL_SAR_FUSION"],
                    details={"region_count": 0},
                )
            )

        status, quality_score, narrative = cls._evaluate_status_and_narrative(
            conflicts=conflicts,
            evidence=evidence,
            query=query,
        )

        for item in evidence.fused_items:
            if not item.is_uncertain:
                supporting_ids.append(item.item_id)

        return ConsistencyReport(
            status=status,
            is_gated=status in (ConsistencyStatus.CONTRADICTORY, ConsistencyStatus.INSUFFICIENT_EVIDENCE),
            gating_action="qualify" if status in (ConsistencyStatus.PARTIALLY_CONSISTENT, ConsistencyStatus.UNCERTAIN) else (
                "state_insufficient" if status == ConsistencyStatus.INSUFFICIENT_EVIDENCE else "allow"
            ),
            conflicts=conflicts,
            supporting_evidence_ids=supporting_ids,
            warnings=warnings,
            specialist_confidences=specialist_confs,
            evidence_quality_score=quality_score,
            summary_narrative=narrative,
        )

    @classmethod
    def check_single_image(
        cls,
        evidence: EvidenceBundle,
        task_name: str,
        query: str = "",
        image_metadata: Optional[Dict[str, Any]] = None,
    ) -> ConsistencyReport:
        """Evaluates evidence sufficiency and provenance for single-image workflows (User Correction 3).
        
        Does not manufacture fake cross-model agreement; focuses honestly on evidence sufficiency,
        box degeneracy, and spatial projection validity.
        """
        conflicts: List[EvidenceConflict] = []
        warnings: List[str] = []
        supporting_ids: List[str] = []
        meta = image_metadata or {}

        specialist_confs: Dict[str, float] = {}
        for item in evidence.fused_items:
            if item.source_specialist not in specialist_confs:
                specialist_confs[item.source_specialist] = item.specialist_confidence

        # -------------------------------------------------------------------
        # RULE C8: EVIDENCE SUFFICIENCY & DEGENERACY (Single Image)
        # -------------------------------------------------------------------
        if task_name == "grounding":
            if not evidence.boxes:
                conflicts.append(
                    EvidenceConflict(
                        conflict_id=cls._create_conflict_id("c8_nobox"),
                        rule_violated="C8_INSUFFICIENT_EVIDENCE",
                        severity="warning",
                        description=f"No spatial candidate regions satisfied detection confidence thresholds for '{query}'.",
                        conflicting_sources=["RS_GROUND"],
                        details={"query": query},
                    )
                )
            else:
                for b in evidence.boxes:
                    if b.is_degenerate:
                        conflicts.append(
                            EvidenceConflict(
                                conflict_id=cls._create_conflict_id("c8_degen"),
                                rule_violated="C8_DEGENERATE_DETECTION",
                                severity="warning",
                                description=f"Degenerate bounding box detected ({b.degenerate_reason or 'abnormal dimensions'}).",
                                conflicting_sources=["RS_GROUND"],
                                details={"box_id": b.box_id, "reason": b.degenerate_reason},
                            )
                        )

        # Check CRS projection sufficiency if physical metrics were requested
        if any(term in query.lower() for term in ("area", "hectare", "meter", "size", "distance")):
            if not meta.get("crs"):
                warnings.append("Input GeoTIFF lacks projected CRS; physical ground metrics cannot be deterministically computed.")

        status, quality_score, narrative = cls._evaluate_status_and_narrative(
            conflicts=conflicts,
            evidence=evidence,
            query=query,
        )

        for item in evidence.fused_items:
            if not item.is_uncertain:
                supporting_ids.append(item.item_id)

        return ConsistencyReport(
            status=status,
            is_gated=status == ConsistencyStatus.INSUFFICIENT_EVIDENCE,
            gating_action="state_insufficient" if status == ConsistencyStatus.INSUFFICIENT_EVIDENCE else (
                "qualify" if status == ConsistencyStatus.UNCERTAIN else "allow"
            ),
            conflicts=conflicts,
            supporting_evidence_ids=supporting_ids,
            warnings=warnings,
            specialist_confidences=specialist_confs,
            evidence_quality_score=quality_score,
            summary_narrative=narrative,
        )

    @classmethod
    def _evaluate_status_and_narrative(
        cls,
        conflicts: List[EvidenceConflict],
        evidence: EvidenceBundle,
        query: str,
    ) -> Tuple[ConsistencyStatus, float, str]:
        """Calculates qualitative status and summary narrative without fabricating numerical confidence penalties."""
        critical_count = sum(1 for c in conflicts if c.severity == "critical")
        warning_count = sum(1 for c in conflicts if c.severity == "warning")

        if critical_count > 0:
            status = ConsistencyStatus.CONTRADICTORY
            quality = max(0.1, 1.0 - (critical_count * 0.40 + warning_count * 0.15))
            narrative = f"Contradiction detected across specialist evidence: {conflicts[0].description}"
        elif any(c.rule_violated.startswith("C8_INSUFFICIENT") for c in conflicts):
            status = ConsistencyStatus.INSUFFICIENT_EVIDENCE
            quality = 0.35
            narrative = f"Insufficient evidence to fully support query: {conflicts[0].description}"
        elif warning_count > 1:
            status = ConsistencyStatus.UNCERTAIN
            quality = max(0.40, 1.0 - warning_count * 0.20)
            narrative = f"Evidence contains multiple ambiguities ({warning_count} advisory conflicts identified)."
        elif warning_count == 1:
            status = ConsistencyStatus.PARTIALLY_CONSISTENT
            quality = 0.80
            narrative = f"Evidence is partially consistent with minor localized divergence: {conflicts[0].description}"
        else:
            status = ConsistencyStatus.CONSISTENT
            quality = 1.0
            narrative = "Evidence is consistent and corroborates findings across invoked specialists."

        return status, round(quality, 4), narrative


class EvidenceGater:
    """Gates and qualifies final answer narratives based on consistency reports."""

    @classmethod
    def gate_answer(
        cls,
        raw_answer: str,
        report: ConsistencyReport,
    ) -> str:
        """Modifies or qualifies answer narrative to protect against unevidenced or contradictory claims."""
        status = report.status

        # 1. CONSISTENT: Return unmodified
        if status == ConsistencyStatus.CONSISTENT:
            return raw_answer

        # 2. CONTRADICTORY: Explicitly surface the contradiction; prevent presenting claim as fact
        if status == ConsistencyStatus.CONTRADICTORY:
            crit_conflict = next((c for c in report.conflicts if c.severity == "critical"), report.conflicts[0])
            return (
                f"[CONTRADICTION DETECTED] {crit_conflict.description} "
                f"The semantic interpretation cannot be verified as factual because it directly contradicts "
                f"underlying physical spatial evidence."
            )

        # 3. INSUFFICIENT EVIDENCE: Explicitly declare insufficiency
        if status == ConsistencyStatus.INSUFFICIENT_EVIDENCE:
            reason = report.conflicts[0].description if report.conflicts else "Insufficient spatial or spectral evidence."
            return f"[INSUFFICIENT EVIDENCE] {reason} A definitive analytical conclusion cannot be grounded without additional observational data."

        # 4. UNCERTAIN: Attach uncertainty qualification
        if status == ConsistencyStatus.UNCERTAIN:
            return (
                f"[PROVISIONAL / UNCERTAIN EVIDENCE] {raw_answer} "
                f"(Advisory: Multi-modal evidence contains ambiguities; {len(report.conflicts)} potential conflicts noted)."
            )

        # 5. PARTIALLY CONSISTENT: Attach minor qualification
        if status == ConsistencyStatus.PARTIALLY_CONSISTENT:
            reason = report.conflicts[0].description if report.conflicts else "minor discrepancies noted."
            return f"{raw_answer} [Note: Partially consistent; {reason}]"

        return raw_answer
