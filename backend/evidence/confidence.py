"""Defensible System Confidence Engine (Milestone M9).

Responsible for calculating the final system-level confidence for SatQuery AI.
Consumes:
1. Input / Geospatial Quality (valid CRS, resolution, overlap, temporal validity)
2. Specialist Signals (model confidences, cascade conditioning, degeneracy)
3. Evidence Quality (from M8 EvidenceNormalizer / ConsistencyReport)
4. Multi-Source Consistency (M8 ConsistencyStatus, conflicts, contradiction flags)
5. Uncertainty Signals (semantic uncertainty, ambiguous modality signatures)

Rules enforced:
- Contradictions dominate: a critical contradiction heavily caps final confidence (<= 0.35).
- Status-aware: maps CONSISTENT, PARTIALLY_CONSISTENT, UNCERTAIN, INSUFFICIENT_EVIDENCE, CONTRADICTORY.
- No naive averaging: models are conditioned on input quality, spatial cascade, and evidence.
- Explainable: provides confidence_level (HIGH, MEDIUM, LOW, UNSUPPORTED), factors, and warnings.
- Uncalibrated disclosure: explicitly marked as heuristic, not statistically calibrated probability.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.agent.schema import (
    ConfidenceBreakdown,
    ConfidenceLevel,
    ConsistencyReport,
    ConsistencyStatus,
    EvidenceBundle,
    SpecialistOutput,
    TaskType,
)

logger = logging.getLogger("satquery.confidence")


class ConfidenceEngine:
    """Multi-factor defensible system confidence engine for remote sensing analysis."""

    # Baseline weight allocation (Rule 11)
    W_INPUT: float = 0.15
    W_ALIGN: float = 0.15
    W_MODEL: float = 0.50
    W_EVIDENCE: float = 0.20

    # Status-aware hard ceilings (Dominance Rules)
    STATUS_CAPS: Dict[ConsistencyStatus, float] = {
        ConsistencyStatus.CONSISTENT: 0.99,
        ConsistencyStatus.PARTIALLY_CONSISTENT: 0.75,
        ConsistencyStatus.UNCERTAIN: 0.60,
        ConsistencyStatus.INSUFFICIENT_EVIDENCE: 0.40,
        ConsistencyStatus.CONTRADICTORY: 0.35,
    }

    # Status-aware consistency multiplier factors
    STATUS_FACTORS: Dict[ConsistencyStatus, float] = {
        ConsistencyStatus.CONSISTENT: 1.00,
        ConsistencyStatus.PARTIALLY_CONSISTENT: 0.85,
        ConsistencyStatus.UNCERTAIN: 0.70,
        ConsistencyStatus.INSUFFICIENT_EVIDENCE: 0.45,
        ConsistencyStatus.CONTRADICTORY: 0.30,
    }

    # Status-aware baseline consistency penalties
    STATUS_PENALTIES: Dict[ConsistencyStatus, float] = {
        ConsistencyStatus.CONSISTENT: 0.00,
        ConsistencyStatus.PARTIALLY_CONSISTENT: 0.12,
        ConsistencyStatus.UNCERTAIN: 0.20,
        ConsistencyStatus.INSUFFICIENT_EVIDENCE: 0.35,
        ConsistencyStatus.CONTRADICTORY: 0.50,
    }

    # Thresholds for human-interpretable confidence tiers
    THRESHOLDS = {
        ConfidenceLevel.HIGH: 0.78,
        ConfidenceLevel.MEDIUM: 0.55,
        ConfidenceLevel.LOW: 0.35,
    }

    @classmethod
    def determine_level(cls, score: float) -> ConfidenceLevel:
        """Determines human-interpretable confidence tier from continuous confidence score."""
        if score >= cls.THRESHOLDS[ConfidenceLevel.HIGH]:
            return ConfidenceLevel.HIGH
        elif score >= cls.THRESHOLDS[ConfidenceLevel.MEDIUM]:
            return ConfidenceLevel.MEDIUM
        elif score >= cls.THRESHOLDS[ConfidenceLevel.LOW]:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.UNSUPPORTED

    @classmethod
    def _determine_level(cls, score: float) -> ConfidenceLevel:
        return cls.determine_level(score)

    @classmethod
    def evaluate(
        cls,
        task: Union[TaskType, str],
        specialist_outputs: Optional[List[SpecialistOutput]] = None,
        consistency_report: Optional[ConsistencyReport] = None,
        input_quality: float = 1.0,
        spatial_alignment: float = 1.0,
        evidence_bundle: Optional[EvidenceBundle] = None,
        query: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        specialist_output: Optional[SpecialistOutput] = None,
        spatial_alignment_score: Optional[float] = None,
        geo_meta: Optional[Any] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, ConfidenceBreakdown, ConfidenceLevel, List[str], List[str]]:
        """Calculates final explainable system confidence and detailed breakdown.

        Returns:
            Tuple of:
            - overall_confidence: float in [0.05, 0.99]
            - confidence_breakdown: ConfidenceBreakdown
            - confidence_level: ConfidenceLevel (HIGH, MEDIUM, LOW, UNSUPPORTED)
            - factors: List[str]
            - warnings: List[str]
        """
        task_str = task.value if isinstance(task, TaskType) else str(task)
        if specialist_outputs is None and specialist_output is not None:
            outputs = [specialist_output]
        else:
            outputs = specialist_outputs or []
        meta = metadata or {}
        params = parameters or meta
        factors: List[str] = []
        conf_warnings: List[str] = []

        if spatial_alignment_score is not None:
            spatial_alignment = spatial_alignment_score

        # -------------------------------------------------------------------
        # 1. Evaluate Input & Geospatial Quality (Q_input)
        # -------------------------------------------------------------------
        q_input = max(0.20, min(1.0, float(input_quality)))
        if geo_meta is not None:
            crs = getattr(geo_meta, "crs", None)
            is_proj = getattr(geo_meta, "is_projected", False)
            if crs:
                factors.append("verified_spatial_crs")
                if is_proj:
                    factors.append("valid_geospatial_alignment")
            else:
                conf_warnings.append("Input lacks valid spatial CRS projection.")
                q_input = min(q_input, 0.70)
                spatial_alignment = min(spatial_alignment, 0.30)
        elif q_input >= 0.90:
            factors.append("valid_georeferenced_crs")
        elif q_input < 0.75:
            conf_warnings.append("Unprojected or local coordinates; physical ground measurements unverified.")
            factors.append("unprojected_local_coordinates")

        # -------------------------------------------------------------------
        # 2. Evaluate Spatial Alignment Quality (Q_align)
        # -------------------------------------------------------------------
        is_single = "single" in task_str or task_str in ["vqa", "grounding", "caption"]
        if is_single:
            q_align = 1.0
            factors.append("single_image_self_aligned")
        else:
            q_align = max(0.0, min(1.0, float(spatial_alignment)))
            if q_align >= 0.85:
                factors.append("strong_spatial_co_registration")
            elif q_align < 0.50:
                conf_warnings.append(f"Substantial spatial misalignment or low overlap ({q_align * 100:.1f}%).")
                factors.append("low_spatial_overlap")

        # -------------------------------------------------------------------
        # 3. Evaluate Specialist Signals (S_model) — Non-Naive Modeling
        # -------------------------------------------------------------------
        s_model, model_factors, model_warns = cls._compute_specialist_signal(
            task_str=task_str,
            outputs=outputs,
            consistency_report=consistency_report,
            evidence_bundle=evidence_bundle,
        )
        factors.extend(model_factors)
        conf_warnings.extend(model_warns)

        # -------------------------------------------------------------------
        # 4. Evaluate Evidence Quality (Q_evidence)
        # -------------------------------------------------------------------
        if consistency_report and consistency_report.evidence_quality_score is not None:
            q_evidence = max(0.0, min(1.0, consistency_report.evidence_quality_score))
        else:
            q_evidence = 1.0 if not is_single else 0.90

        if evidence_bundle is not None:
            has_spatial_evidence = bool(
                evidence_bundle.masks
                or evidence_bundle.boxes
                or evidence_bundle.regions
                or evidence_bundle.complementarity_report
            )
            if not has_spatial_evidence and not evidence_bundle.statistics:
                if params.get("changed_pixels") == 0 and params.get("change_ratio_pct") == 0.0:
                    factors.append("verified_zero_change_state")
                    q_evidence = max(q_evidence, 0.90)
                else:
                    q_evidence = min(q_evidence, 0.20)
                    conf_warnings.append("Sparse or missing spatial evidence items in bundle.")
                    factors.append("missing_evidence")
            elif params.get("changed_pixels") == 0 and params.get("change_ratio_pct") == 0.0:
                factors.append("verified_zero_change_state")
                q_evidence = max(q_evidence, 0.90)

        if q_evidence >= 0.85:
            factors.append("high_evidence_completeness")
        elif q_evidence < 0.60:
            conf_warnings.append(f"Incomplete spatial evidence bundle (quality: {q_evidence * 100:.0f}%).")
            factors.append("compromised_evidence_quality")

        # -------------------------------------------------------------------
        # 5. Base Weighted Score (S_base)
        # -------------------------------------------------------------------
        s_base = (
            cls.W_INPUT * q_input
            + cls.W_ALIGN * q_align
            + cls.W_MODEL * s_model
            + cls.W_EVIDENCE * q_evidence
        )

        # -------------------------------------------------------------------
        # 6. Apply Status-Aware Consistency Modulation & Penalties
        # -------------------------------------------------------------------
        status = consistency_report.status if consistency_report else ConsistencyStatus.CONSISTENT
        c_factor = cls.STATUS_FACTORS.get(status, 1.00)
        c_penalty = cls.STATUS_PENALTIES.get(status, 0.00)
        cap = cls.STATUS_CAPS.get(status, 0.99)

        # Adjust penalty for conflict volume if partially consistent
        if status == ConsistencyStatus.PARTIALLY_CONSISTENT and consistency_report:
            conflict_count = len(consistency_report.conflicts)
            c_penalty = min(0.30, c_penalty + 0.03 * max(0, conflict_count - 1))

        # Check for critical conflicts
        if consistency_report:
            has_critical = any(c.severity == "critical" for c in consistency_report.conflicts)
            if has_critical:
                cap = min(cap, 0.35)
                c_penalty = max(c_penalty, 0.45)
                conf_warnings.append("Critical contradiction between specialist outputs; confidence strictly capped.")
                factors.append("critical_contradiction_override")

        # Record qualitative status factor
        if status == ConsistencyStatus.CONSISTENT:
            factors.append("consistent_specialist_outputs")
        elif status == ConsistencyStatus.PARTIALLY_CONSISTENT:
            factors.append("partially_consistent_evidence")
            conf_warnings.append("Specialist outputs are partially consistent; minor discrepancies detected.")
        elif status == ConsistencyStatus.UNCERTAIN:
            factors.append("elevated_system_uncertainty")
            conf_warnings.append("System results reflect uncertain specialist findings or high classification entropy.")
        elif status == ConsistencyStatus.INSUFFICIENT_EVIDENCE:
            factors.append("insufficient_evidence_cap")
            conf_warnings.append("Insufficient visual/spatial evidence to confirm analytical claim.")
        elif status == ConsistencyStatus.CONTRADICTORY:
            factors.append("contradictory_specialist_claims")
            conf_warnings.append("Multi-source contradiction detected between specialists.")

        # -------------------------------------------------------------------
        # 7. Final System Confidence Calculation (Dominance Cap Applied)
        # -------------------------------------------------------------------
        raw_confidence = (s_base * c_factor) - c_penalty
        overall_confidence = max(0.05, min(cap, round(raw_confidence, 4)))

        # -------------------------------------------------------------------
        # 8. Categorize Confidence Level
        # -------------------------------------------------------------------
        confidence_level = cls.determine_level(overall_confidence)

        # Format details
        calc_details = {
            "task": task_str,
            "weights": {
                "w_input": cls.W_INPUT,
                "w_align": cls.W_ALIGN,
                "w_model": cls.W_MODEL,
                "w_evidence": cls.W_EVIDENCE,
            },
            "scores": {
                "input_quality": round(q_input, 4),
                "spatial_alignment": round(q_align, 4),
                "model_confidence": round(s_model, 4),
                "evidence_quality": round(q_evidence, 4),
                "base_score": round(s_base, 4),
            },
            "consistency": {
                "status": status.value,
                "factor": c_factor,
                "penalty": round(c_penalty, 4),
                "cap_applied": cap,
                "is_capped": raw_confidence > cap,
            },
            "capped_by_status": raw_confidence > cap or status in (ConsistencyStatus.CONTRADICTORY, ConsistencyStatus.INSUFFICIENT_EVIDENCE),
            "status_cap": cap,
            "is_calibrated_probability": False,
            "calibration_status": "system confidence heuristic — not statistically calibrated",
        }

        # Build breakdown object conforming strictly to schema
        breakdown = ConfidenceBreakdown(
            heuristic_name="evidence-weighted confidence heuristic",
            overall_confidence=overall_confidence,
            specialist_confidence=round(s_model, 4),
            input_quality_score=round(q_input, 4),
            spatial_alignment_score=round(q_align, 4),
            model_confidence_score=round(s_model, 4),
            consistency_penalty=round(min(0.50, c_penalty), 4),
            is_calibrated_probability=False,
            evidence_quality_score=round(q_evidence, 4),
            confidence_level=confidence_level,
            confidence_factors=factors,
            confidence_warnings=conf_warnings,
            calculation_details=calc_details,
        )

        return (
            overall_confidence,
            breakdown,
            confidence_level,
            factors,
            conf_warnings,
        )

    # -----------------------------------------------------------------------
    # Helper: Non-Naive Specialist Signal Computation
    # -----------------------------------------------------------------------
    @classmethod
    def _compute_specialist_signal(
        cls,
        task_str: str,
        outputs: List[SpecialistOutput],
        consistency_report: Optional[ConsistencyReport],
        evidence_bundle: Optional[EvidenceBundle],
    ) -> Tuple[float, List[str], List[str]]:
        """Computes specialist signal, accounting for cascaded multi-stage pipelines."""
        factors: List[str] = []
        warns: List[str] = []

        if not outputs:
            confs = consistency_report.specialist_confidences if consistency_report else {}
            if confs:
                mean_conf = sum(confs.values()) / len(confs)
                return mean_conf, ["registered_specialist_signals"], []
            return 0.70, ["default_baseline_specialist_score"], []

        # Case 1: Bi-temporal Multi-Stage Pipeline (CHANGE_DETECT + CHANGE_VQA)
        cd_out = next((o for o in outputs if o.specialist_id in ["CHANGE_DETECT", "TinyCD", "CVA"]), None)
        vqa_out = next((o for o in outputs if o.specialist_id == "CHANGE_VQA"), None)

        if cd_out and vqa_out:
            s_det = cd_out.confidence
            s_vqa = vqa_out.confidence
            sem_unc = 0.0
            if vqa_out.evidence and vqa_out.evidence.semantic_interpretation:
                sem_unc = vqa_out.evidence.semantic_interpretation.semantic_uncertainty

            # Non-naive cascade: semantic interpretation is physically bounded by detector accuracy
            s_combined = s_det * (0.35 + 0.65 * (1.0 - sem_unc * 0.5))
            if s_combined >= 0.82:
                factors.append("strong_bitemporal_cascade_support")
            else:
                warns.append("Weak or degraded detection stage bounds semantic interpretation.")
                factors.append("cascaded_uncertainty_propagation")
            return round(min(0.99, max(0.10, s_combined)), 4), factors, warns

        # Case 2: Optical-SAR Joint Analysis
        opt_sar_out = next((o for o in outputs if o.specialist_id in ["OPTICAL_SAR_FUSION", "OpticalSARSpecialist"]), None)
        if opt_sar_out:
            s_fused = opt_sar_out.confidence
            dist = opt_sar_out.parameters_used.get("class_distribution", {})
            unk_area = dist.get("unknown", 0)
            tot_px = opt_sar_out.parameters_used.get("total_pixels", 1)
            unk_pct = (unk_area / tot_px) * 100.0 if tot_px > 0 else 0.0

            if unk_pct > 25.0:
                s_fused = max(0.20, s_fused * 0.75)
                warns.append(f"High cross-modal ambiguity: {unk_pct:.1f}% unclassified terrain area.")
                factors.append("elevated_cross_modal_ambiguity")
            else:
                factors.append("dual_sensor_spectral_structural_concordance")
            return round(min(0.99, max(0.10, s_fused)), 4), factors, warns

        # Case 3: Single-Image Grounding
        ground_out = next(
            (o for o in outputs if o.specialist_id in ["RS_GROUND", "GroundingDINO", "rs_grounding_specialist"] or "ground" in task_str.lower()),
            None,
        )
        if ground_out:
            s_ground = ground_out.confidence
            boxes = (ground_out.evidence.boxes if ground_out.evidence else []) or (evidence_bundle.boxes if evidence_bundle else [])
            regions = (ground_out.evidence.regions if ground_out.evidence else []) or (evidence_bundle.regions if evidence_bundle else [])
            has_boxes = bool(boxes or regions)
            if not has_boxes:
                warns.append("Zero candidate regions satisfied referring expression.")
                factors.append("negative_detection_expression")
                return 0.25, factors, warns

            if any(b.is_degenerate or (b.coordinates_normalized and b.coordinates_normalized[2] - b.coordinates_normalized[0] > 0.95 and b.coordinates_normalized[3] - b.coordinates_normalized[1] > 0.95) for b in boxes):
                warns.append("Degenerate frame-filling bounding box detected.")
                factors.append("degenerate_bounding_box")
                return 0.30, factors, warns

            factors.append("grounded_bounding_box_available")
            factors.append("verified_spatial_region_grounding")
            return round(min(0.99, max(0.10, s_ground)), 4), factors, warns

        # Case 4: Single-Image VQA or general single specialist
        primary_out = outputs[0]
        s_single = primary_out.confidence
        if s_single >= 0.85:
            factors.append("high_specialist_confidence")
        elif s_single < 0.60:
            warns.append("Specialist model reported low internal confidence.")
            factors.append("low_specialist_confidence")

        return round(min(0.99, max(0.10, s_single)), 4), factors, warns
