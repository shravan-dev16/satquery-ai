"""Deterministic Operational Planner for Remote-Sensing Analysis (Milestone M7).

Adheres to AGENTS.md Rule 6 (Agentic Orchestration):
Constructs an explicit, observable execution plan from a RoutingDecision.
Does NOT implement hidden chain-of-thought; exposes strictly operational execution plans.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from backend.agent.schema import ExecutionPlan, PlanStep, RoutingDecision, TaskType

logger = logging.getLogger(__name__)


class AgentPlanner:
    """Constructs explicit execution sequences for single, bi-temporal, and multimodal tasks."""

    @classmethod
    def create_plan(
        cls,
        decision: RoutingDecision,
        primary_path: Path,
        secondary_path: Optional[Path] = None,
        query: str = "",
        parameters: Optional[Dict[str, Any]] = None,
    ) -> ExecutionPlan:
        """Constructs ordered PlanStep sequence based on resolved task and input configuration."""
        plan_id = f"plan_{uuid.uuid4().hex[:8]}"
        task = decision.task
        params = parameters or {}

        steps: List[PlanStep] = []
        step_idx = 1

        # -------------------------------------------------------------------
        # Plan 1: Single-Image VQA Plan
        # -------------------------------------------------------------------
        if task == TaskType.VQA:
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="InputValidation",
                    action_type="validate",
                    description="Validate single GeoTIFF raster readability, CRS, dimensions, and band count.",
                    inputs_required=["primary_image"],
                    outputs_produced=["validated_metadata"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="TaskRouting",
                    action_type="route",
                    description=f"Map query '{query[:40]}...' to single-image visual question answering.",
                    inputs_required=["query", "validated_metadata"],
                    outputs_produced=["task_assignment"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpecialistSelection",
                    action_type="select_specialist",
                    specialist_id="RS_VQA",
                    description="Resolve RemoteSensingVQASpecialist from ModelRegistry.",
                    inputs_required=["task_assignment"],
                    outputs_produced=["specialist_handle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ModelExecution",
                    action_type="specialist_predict",
                    specialist_id="RS_VQA",
                    description="Execute remote-sensing visual question answering inference.",
                    inputs_required=["primary_image", "query", "specialist_handle"],
                    outputs_produced=["vqa_answer", "model_confidence"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceNormalization",
                    action_type="normalize_evidence",
                    description="Normalize VQA output into standardized EvidenceItem with preserved source provenance.",
                    inputs_required=["vqa_answer", "model_confidence"],
                    outputs_produced=["normalized_evidence_items"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ConsistencyCheck",
                    action_type="consistency_check",
                    description="Evaluate evidence sufficiency and metadata provenance.",
                    inputs_required=["normalized_evidence_items"],
                    outputs_produced=["consistency_report"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceFusion",
                    action_type="fuse_evidence",
                    description="Synthesize normalized evidence units into consolidated EvidenceBundle.",
                    inputs_required=["normalized_evidence_items", "consistency_report"],
                    outputs_produced=["fused_evidence_bundle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceAssembly",
                    action_type="assemble_evidence",
                    description="Gate evidence claims and assemble StandardResultContract.",
                    inputs_required=["fused_evidence_bundle", "consistency_report"],
                    outputs_produced=["standard_result_contract"],
                )
            )

        # -------------------------------------------------------------------
        # Plan 2: Single-Image Grounding Plan
        # -------------------------------------------------------------------
        elif task == TaskType.GROUNDING:
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="InputValidation",
                    action_type="validate",
                    description="Validate single GeoTIFF raster readability and spatial projection.",
                    inputs_required=["primary_image"],
                    outputs_produced=["validated_metadata"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="TaskRouting",
                    action_type="route",
                    description=f"Route text-guided grounding query '{query[:40]}...' to RS_GROUND.",
                    inputs_required=["query"],
                    outputs_produced=["task_assignment"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpecialistSelection",
                    action_type="select_specialist",
                    specialist_id="RS_GROUND",
                    description="Resolve RemoteSensingGroundingSpecialist from ModelRegistry.",
                    inputs_required=["task_assignment"],
                    outputs_produced=["specialist_handle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ModelExecution",
                    action_type="specialist_predict",
                    specialist_id="RS_GROUND",
                    description="Execute open-vocabulary spatial grounding detector.",
                    inputs_required=["primary_image", "query"],
                    outputs_produced=["detected_boxes"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceNormalization",
                    action_type="normalize_evidence",
                    description="Normalize bounding boxes into standardized EvidenceItem records with spatial bounds.",
                    inputs_required=["detected_boxes"],
                    outputs_produced=["normalized_evidence_items"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ConsistencyCheck",
                    action_type="consistency_check",
                    description="Evaluate detection degeneracy, empty candidates, and CRS sufficiency (Rule C8).",
                    inputs_required=["normalized_evidence_items"],
                    outputs_produced=["consistency_report"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceFusion",
                    action_type="fuse_evidence",
                    description="Synthesize normalized spatial evidence units into consolidated EvidenceBundle.",
                    inputs_required=["normalized_evidence_items", "consistency_report"],
                    outputs_produced=["fused_evidence_bundle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceAssembly",
                    action_type="assemble_evidence",
                    description="Project pixel bounding boxes to GeoJSON and assemble StandardResultContract.",
                    inputs_required=["fused_evidence_bundle", "validated_metadata"],
                    outputs_produced=["standard_result_contract"],
                )
            )

        # -------------------------------------------------------------------
        # Plan 3: Single-Image Captioning Plan
        # -------------------------------------------------------------------
        elif task == TaskType.CAPTION:
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="InputValidation",
                    action_type="validate",
                    description="Validate single GeoTIFF raster.",
                    inputs_required=["primary_image"],
                    outputs_produced=["validated_metadata"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="TaskRouting",
                    action_type="route",
                    description="Route scene description query to RS_CAPTION.",
                    inputs_required=["query"],
                    outputs_produced=["task_assignment"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpecialistSelection",
                    action_type="select_specialist",
                    specialist_id="RS_CAPTION",
                    description="Resolve RemoteSensingCaptionSpecialist from ModelRegistry.",
                    inputs_required=["task_assignment"],
                    outputs_produced=["specialist_handle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ModelExecution",
                    action_type="specialist_predict",
                    specialist_id="RS_CAPTION",
                    description="Execute remote-sensing caption generation.",
                    inputs_required=["primary_image", "specialist_handle"],
                    outputs_produced=["caption_text"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceAssembly",
                    action_type="assemble_evidence",
                    description="Assemble StandardResultContract with generated description.",
                    inputs_required=["caption_text"],
                    outputs_produced=["standard_result_contract"],
                )
            )

        # -------------------------------------------------------------------
        # Plan 4: Multi-Stage Bi-Temporal Change Analysis (WHERE -> WHAT KIND)
        # -------------------------------------------------------------------
        elif task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="InputValidation",
                    action_type="validate",
                    description="Validate readability, band counts, and dimensions of both T1 and T2 rasters.",
                    inputs_required=["primary_image", "secondary_image"],
                    outputs_produced=["pair_metadata"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="TemporalValidation",
                    action_type="validate",
                    description="Verify temporal acquisition ordering (T1 earlier than T2).",
                    inputs_required=["pair_metadata"],
                    outputs_produced=["temporal_status"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpatialCompatibility",
                    action_type="validate",
                    description="Verify CRS compatibility and compute spatial overlap (>=20% policy).",
                    inputs_required=["pair_metadata"],
                    outputs_produced=["overlap_metrics"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="Alignment",
                    action_type="align",
                    description="Deterministically reproject and crop rasters to common spatial grid.",
                    inputs_required=["primary_image", "secondary_image", "overlap_metrics"],
                    outputs_produced=["aligned_t1", "aligned_t2"],
                )
            )
            step_idx += 1
            spec_list = "CHANGE_DETECT and CHANGE_VQA" if decision.is_multi_stage else "CHANGE_DETECT"
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpecialistSelection",
                    action_type="select_specialist",
                    specialist_id="CHANGE_DETECT",
                    description=f"Resolve {spec_list} from central ModelRegistry.",
                    inputs_required=["task"],
                    outputs_produced=["specialist_handles"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ModelExecution",
                    action_type="specialist_predict",
                    specialist_id="CHANGE_DETECT",
                    description="Execute binary change detection (TinyCD / CVA) to locate surface modifications.",
                    inputs_required=["aligned_t1", "aligned_t2"],
                    outputs_produced=["change_mask", "changed_pixels", "change_clusters"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ChangeMaskValidation",
                    action_type="validate",
                    description="Verify change mask topology, NaN absence, and extreme change ratio sanity.",
                    inputs_required=["change_mask"],
                    outputs_produced=["mask_diagnostics"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="Statistics",
                    action_type="calculate_statistics",
                    description="Compute physical changed area (m², hectares) and percentage coverage.",
                    inputs_required=["change_mask", "pair_metadata"],
                    outputs_produced=["zonal_statistics"],
                )
            )
            step_idx += 1

            if decision.is_multi_stage:
                steps.append(
                    PlanStep(
                        step_number=step_idx,
                        step_name="SemanticInterpretation",
                        action_type="specialist_predict",
                        specialist_id="CHANGE_VQA",
                        description=(
                            "Condition ChangeVQASpecialist on detected spatial clusters and statistics to generate "
                            "grounded land-cover transition interpretation (anti-hallucination gated)."
                        ),
                        inputs_required=["aligned_t1", "aligned_t2", "change_mask", "change_clusters", "zonal_statistics"],
                        outputs_produced=["semantic_interpretation", "vlm_answer"],
                    )
                )
                step_idx += 1

            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceNormalization",
                    action_type="normalize_evidence",
                    description="Normalize change detection and semantic VQA into standardized EvidenceItem records.",
                    inputs_required=["change_mask", "zonal_statistics"],
                    outputs_produced=["normalized_evidence_items"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ConsistencyCheck",
                    action_type="consistency_check",
                    description="Evaluate Rules C1-C5, C7, C8: zero-change, region support, area match, and direction.",
                    inputs_required=["normalized_evidence_items"],
                    outputs_produced=["consistency_report"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceFusion",
                    action_type="fuse_evidence",
                    description="Synthesize normalized multi-modal items into consolidated EvidenceBundle.",
                    inputs_required=["normalized_evidence_items", "consistency_report"],
                    outputs_produced=["fused_evidence_bundle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceAssembly",
                    action_type="assemble_evidence",
                    description="Synthesize visual overlays, gate answer claims, and assemble StandardResultContract.",
                    inputs_required=["fused_evidence_bundle", "consistency_report"],
                    outputs_produced=["standard_result_contract"],
                )
            )

        # -------------------------------------------------------------------
        # Plan 5: Optical + SAR Joint Analysis Plan
        # -------------------------------------------------------------------
        elif task == TaskType.OPTICAL_SAR_ANALYSIS:
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="InputValidation",
                    action_type="validate",
                    description="Validate Optical and SAR GeoTIFF headers, bands, and projections.",
                    inputs_required=["primary_image", "secondary_image"],
                    outputs_produced=["pair_metadata"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ModalityValidation",
                    action_type="validate",
                    description="Verify exactly one Optical/Multispectral raster and one SAR raster; normalize order if reversed.",
                    inputs_required=["pair_metadata"],
                    outputs_produced=["normalized_pair"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpatialCompatibility",
                    action_type="validate",
                    description="Verify CRS compatibility and spatial intersection (>=20% overlap policy).",
                    inputs_required=["normalized_pair"],
                    outputs_produced=["spatial_overlap"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="Alignment",
                    action_type="align",
                    description="Deterministically co-register SAR raster to Optical cropped intersection grid.",
                    inputs_required=["normalized_pair", "spatial_overlap"],
                    outputs_produced=["aligned_optical", "aligned_sar"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SpecialistSelection",
                    action_type="select_specialist",
                    specialist_id="OPTICAL_SAR_FUSION",
                    description="Resolve OpticalSARSpecialist (OPTICAL_SAR_FUSION) from central ModelRegistry.",
                    inputs_required=["task"],
                    outputs_produced=["specialist_handle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="OpticalFeatureExtraction",
                    action_type="feature_extract",
                    description="Extract normalized RGB, Excess Green (ExG) vegetative index, and spectral ratios.",
                    inputs_required=["aligned_optical"],
                    outputs_produced=["optical_features"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="SARFeatureExtraction",
                    action_type="feature_extract",
                    description="Extract calibrated backscatter intensity and 5x5 moving window texture roughness.",
                    inputs_required=["aligned_sar"],
                    outputs_produced=["sar_features"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="JointFusion",
                    action_type="specialist_predict",
                    specialist_id="OPTICAL_SAR_FUSION",
                    description="Execute raster-level joint feature fusion across 6 remote-sensing semantic classes.",
                    inputs_required=["optical_features", "sar_features"],
                    outputs_produced=["fused_class_map", "confidence_map"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="RegionExtraction",
                    action_type="extract_regions",
                    description="Extract discrete spatial clusters, bounding boxes, and cross-sensor complementarity telemetry.",
                    inputs_required=["fused_class_map"],
                    outputs_produced=["detected_regions", "bounding_boxes"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceNormalization",
                    action_type="normalize_evidence",
                    description="Normalize cross-modal cues into EvidenceItem records with optical/SAR provenance.",
                    inputs_required=["fused_class_map", "detected_regions"],
                    outputs_produced=["normalized_evidence_items"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="ConsistencyCheck",
                    action_type="consistency_check",
                    description="Evaluate Rule C6 (Optical/SAR agreement) and Rule C8 (cross-modal sufficiency).",
                    inputs_required=["normalized_evidence_items"],
                    outputs_produced=["consistency_report"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceFusion",
                    action_type="fuse_evidence",
                    description="Synthesize cross-modal evidence units into consolidated EvidenceBundle.",
                    inputs_required=["normalized_evidence_items", "consistency_report"],
                    outputs_produced=["fused_evidence_bundle"],
                )
            )
            step_idx += 1
            steps.append(
                PlanStep(
                    step_number=step_idx,
                    step_name="EvidenceAssembly",
                    action_type="assemble_evidence",
                    description="Assemble StandardResultContract with gated answer, composite, and ComplementarityReport.",
                    inputs_required=["fused_evidence_bundle", "consistency_report"],
                    outputs_produced=["standard_result_contract"],
                )
            )

        return ExecutionPlan(
            plan_id=plan_id,
            task=task,
            target_specialists=decision.target_specialist_ids,
            is_multi_stage=decision.is_multi_stage,
            steps=steps,
            metadata={
                "input_configuration": decision.input_configuration,
                "is_multi_stage": decision.is_multi_stage,
                "intent_category": decision.intent_category,
            },
        )
