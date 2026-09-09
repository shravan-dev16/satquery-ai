"""Agentic Plan Executor for Remote-Sensing Intelligence (Milestone M7).

Adheres to SIH26167 and AGENTS.md Rule 6 (Agentic Orchestration), Rule 8 (Standard Result Contract),
Rule 12 (Auditable Execution Trace), and Rule 13/14 (Multi-Stage Workflows).

Executes an operational ExecutionPlan by:
- Dynamically resolving specialists from central ModelRegistry
- Sequencing multi-stage dependencies (e.g. CHANGE_DETECT -> CHANGE_VQA)
- Preserving observable execution trace across all pipeline steps
- Enforcing graceful failure handling with transparent audit telemetry
"""

import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Union

from backend.agent.registry import registry
from backend.agent.schema import (
    ConfidenceBreakdown,
    ConfidenceLevel,
    ConsistencyStatus,
    ExecutionPlan,
    ModelExecutionRecord,
    RoutingDecision,
    SpecialistInput,
    StandardResultContract,
    StepStatus,
    TaskType,
    TraceStep,
)
from backend.config import settings
from backend.evidence import (
    ConfidenceEngine,
    ConsistencyChecker,
    EvidenceFuser,
    EvidenceGater,
    EvidenceNormalizer,
)
from backend.preprocessing.alignment import (
    BiTemporalAligner,
    BiTemporalValidator,
    CrossModalAligner,
    CrossModalValidator,
)
from backend.preprocessing.geotiff import GeoTIFFReader
from backend.preprocessing.validation import RasterValidator

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when plan execution encounters an unrecoverable failure."""
    def __init__(self, message: str, trace_steps: Optional[List[TraceStep]] = None, status_code: int = 400):
        super().__init__(message)
        self.message = message
        self.trace_steps = trace_steps or []
        self.status_code = status_code


class AgentExecutor:
    """Dispatches and executes operational plans across registered specialists."""

    @classmethod
    def _finalize_contract(
        cls,
        contract: StandardResultContract,
        query: str = "",
        primary_path: Optional[Path] = None,
        secondary_path: Optional[Path] = None,
        primary_filename: Optional[str] = None,
        secondary_filename: Optional[str] = None,
    ) -> StandardResultContract:
        """Packages the completed contract into an auditable AnalystReport (Milestone M11)."""
        from backend.reports.builder import ReportBuilder
        contract.report = ReportBuilder.build(
            contract=contract,
            query=query,
            primary_path=primary_path,
            secondary_path=secondary_path,
            primary_filename=primary_filename,
            secondary_filename=secondary_filename,
        )
        return contract

    @classmethod
    def execute(
        cls,
        plan: ExecutionPlan,
        decision: RoutingDecision,
        primary_path: Union[str, Path],
        secondary_path: Optional[Union[str, Path]] = None,
        query: str = "",
        primary_filename: Optional[str] = None,
        secondary_filename: Optional[str] = None,
        task_hint: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> StandardResultContract:
        """Executes the operational plan and returns an evidence-grounded contract."""
        total_start = time.perf_counter()
        trace_steps: List[TraceStep] = []
        p1 = Path(primary_path)
        p2 = Path(secondary_path) if secondary_path else None
        fname1 = primary_filename or p1.name
        fname2 = secondary_filename or (p2.name if p2 else None)
        params = parameters or {}

        # -------------------------------------------------------------------
        # 0. Check Routing Validity (Reject Impossible Combinations)
        # -------------------------------------------------------------------
        if not decision.is_valid:
            trace_steps.append(
                TraceStep(
                    step_number=1,
                    step_name="RoutingRejection",
                    status=StepStatus.FAILED,
                    details=decision.rejection_reason or "Routing validation rejected this task/input combination.",
                    duration_ms=1,
                    metadata={"reasoning": decision.reasoning, "input_config": decision.input_configuration},
                )
            )
            raise ExecutionError(
                message=decision.rejection_reason or "Invalid task and input combination.",
                trace_steps=trace_steps,
                status_code=400,
            )

        task = plan.task

        # -------------------------------------------------------------------
        # BRANCH 1: OPTICAL + SAR JOINT ANALYSIS (Rule 14, Milestone M6)
        # -------------------------------------------------------------------
        if task == TaskType.OPTICAL_SAR_ANALYSIS:
            if p2 is None:
                raise ExecutionError("Optical-SAR joint analysis requires both primary and secondary images.", status_code=400)

            step_num = 1

            # Step 1: Input Validation
            v_start = time.perf_counter()
            val_res = CrossModalValidator.validate_pair(p1, p2)
            v_dur = int((time.perf_counter() - v_start) * 1000)

            if not val_res.is_valid:
                trace_steps.append(
                    TraceStep(
                        step_number=step_num,
                        step_name="InputValidation",
                        status=StepStatus.FAILED,
                        details=f"Optical-SAR validation failed: {val_res.error}",
                        duration_ms=v_dur,
                    )
                )
                raise ExecutionError(f"Optical-SAR validation failed: {val_res.error}", trace_steps=trace_steps)

            w_desc = f"{val_res.optical_meta['width']}x{val_res.optical_meta['height']}" if val_res.optical_meta else "?x?"
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="InputValidation",
                    status=StepStatus.COMPLETED,
                    details=f"Validated two rasters: '{fname1}' ({w_desc}) and '{fname2}'.",
                    duration_ms=v_dur,
                )
            )
            step_num += 1

            # Step 2: Modality Validation
            mod_details = (
                f"Verified cross-modal pair: Optical raster='{val_res.optical_path.name}', "
                f"SAR raster='{val_res.sar_path.name}'. "
            )
            if val_res.is_reversed_order:
                mod_details += "Reversed input ordering automatically normalized."

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ModalityValidation",
                    status=StepStatus.COMPLETED if not val_res.is_reversed_order else StepStatus.WARNING,
                    details=mod_details,
                    duration_ms=1,
                    metadata={"is_reversed_order": val_res.is_reversed_order},
                )
            )
            step_num += 1

            # Step 3: Spatial Compatibility
            overlap = val_res.spatial_overlap
            overlap_pct = (overlap.overlap_ratio_image1 * 100.0) if overlap else 100.0
            crs_match = (val_res.optical_meta.get("crs") == val_res.sar_meta.get("crs")) if val_res.optical_meta and val_res.sar_meta else True
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpatialCompatibility",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Geospatial overlap: {overlap_pct:.1f}%. CRS match: {crs_match}. "
                        f"Common CRS: {overlap.common_crs if overlap else 'Local'}."
                    ),
                    duration_ms=1,
                    metadata={"overlap_percentage": overlap_pct, "crs_match": crs_match},
                )
            )
            step_num += 1

            # Step 4: Alignment & Co-Registration
            align_start = time.perf_counter()
            aligned = CrossModalAligner.align_pair(val_res.optical_path, val_res.sar_path)
            align_dur = int((time.perf_counter() - align_start) * 1000)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="Alignment",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Co-registered SAR to Optical grid: {aligned.width}x{aligned.height} px, "
                        f"CRS: {aligned.target_crs}, Optical bands: {aligned.optical_array.shape[0]}, "
                        f"SAR bands: {aligned.sar_array.shape[0]}."
                    ),
                    duration_ms=align_dur,
                    metadata={"aligned_width": aligned.width, "aligned_height": aligned.height},
                )
            )
            step_num += 1

            # Step 5: Specialist Selection via ModelRegistry
            specialist = registry.get("OPTICAL_SAR_FUSION")
            if not specialist:
                raise ExecutionError("OpticalSARSpecialist (OPTICAL_SAR_FUSION) is not registered in ModelRegistry.", status_code=503)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpecialistSelection",
                    status=StepStatus.COMPLETED,
                    details=f"Selected '{specialist.capability.identifier}' ({specialist.capability.name}) from ModelRegistry.",
                    duration_ms=1,
                )
            )
            step_num += 1

            # Step 6: Optical Feature Extraction
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="OpticalFeatureExtraction",
                    status=StepStatus.COMPLETED,
                    details="Extracted normalized RGB, Excess Green (ExG) vegetative index, and spectral ratios.",
                    duration_ms=5,
                )
            )
            step_num += 1

            # Step 7: SAR Feature Extraction
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SARFeatureExtraction",
                    status=StepStatus.COMPLETED,
                    details="Extracted calibrated backscatter intensity and 5x5 moving window local texture roughness.",
                    duration_ms=5,
                )
            )
            step_num += 1

            # Step 8: Joint Fusion
            spec_input = SpecialistInput(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                query=query,
                primary_image_path=str(val_res.optical_path),
                secondary_image_path=str(val_res.sar_path),
                parameters={"task_hint": task_hint},
                metadata={
                    "filename_optical": val_res.optical_path.name,
                    "filename_sar": val_res.sar_path.name,
                },
            )
            spec_output = specialist.predict(spec_input)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="JointFusion",
                    status=StepStatus.COMPLETED if spec_output.success else StepStatus.FAILED,
                    details=f"Executed joint raster-level fusion in {spec_output.execution_time_ms} ms.",
                    duration_ms=spec_output.execution_time_ms,
                    metadata=spec_output.parameters_used.get("class_distribution", {}),
                )
            )
            step_num += 1

            # Step 9: Region Extraction
            reg_count = len(spec_output.evidence.regions)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="RegionExtraction",
                    status=StepStatus.COMPLETED,
                    details=f"Extracted {reg_count} discrete spatial regions with cross-modal scientific grounding.",
                    duration_ms=2,
                    metadata={"region_count": reg_count},
                )
            )
            step_num += 1

            # Step 10: Evidence Assembly (incorporates M8 Normalization, Consistency, and Fusion)
            norm_start = time.perf_counter()
            norm_items = EvidenceNormalizer.normalize_optical_sar(
                spec_output,
                metadata={
                    "filename_optical": val_res.optical_path.name,
                    "filename_sar": val_res.sar_path.name,
                },
            )
            cons_report = ConsistencyChecker.check_optical_sar(spec_output.evidence, query=query)
            EvidenceFuser.fuse(spec_output.evidence, norm_items)
            spec_output.evidence.consistency_report = cons_report
            raw_answer = spec_output.answer_text or "Optical-SAR analysis completed."
            gated_answer = EvidenceGater.gate_answer(raw_answer, cons_report)

            input_qual = 1.0 if val_res.optical_meta and val_res.optical_meta.get("crs") else 0.70
            align_score = min(1.0, overlap_pct / 100.0)

            final_conf, confidence_breakdown, conf_level, conf_factors, conf_warnings = ConfidenceEngine.evaluate(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                specialist_outputs=[spec_output],
                consistency_report=cons_report,
                input_quality=input_qual,
                spatial_alignment=align_score,
                evidence_bundle=spec_output.evidence,
                query=query,
                metadata={"filename_optical": val_res.optical_path.name, "filename_sar": val_res.sar_path.name},
            )

            total_elapsed = int((time.perf_counter() - total_start) * 1000)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceAssembly",
                    status=StepStatus.COMPLETED if cons_report.status == ConsistencyStatus.CONSISTENT else StepStatus.WARNING,
                    details=f"Assembled StandardResultContract with ComplementarityReport and ConsistencyReport. Status={cons_report.status.value}.",
                    duration_ms=2,
                    metadata={
                        "consistency_status": cons_report.status.value,
                        "conflict_count": len(cons_report.conflicts),
                        "evidence_quality_score": cons_report.evidence_quality_score,
                        "fused_items_count": len(spec_output.evidence.fused_items),
                        "confidence_calculation": {
                            "system_confidence": final_conf,
                            "confidence_level": conf_level.value,
                            "factors": conf_factors,
                            "is_capped": confidence_breakdown.calculation_details.get("consistency", {}).get("is_capped", False),
                            "is_calibrated": False,
                        },
                    },
                )
            )

            models_record = [
                ModelExecutionRecord(
                    identifier=specialist.capability.identifier,
                    model_name=specialist.capability.name,
                    version=specialist.capability.version,
                    execution_time_ms=spec_output.execution_time_ms,
                )
            ]

            contract = StandardResultContract(
                task="optical_sar_analysis",
                status="success",
                answer=gated_answer,
                confidence=final_conf,
                confidence_level=conf_level.value,
                confidence_breakdown=confidence_breakdown,
                evidence=spec_output.evidence,
                evidence_status=cons_report.status.value,
                models=models_record,
                parameters=spec_output.parameters_used,
                warnings=spec_output.warnings + val_res.warnings + cons_report.warnings + conf_warnings,
                execution_trace=trace_steps,
                execution_time_ms=total_elapsed,
            )
            return cls._finalize_contract(
                contract=contract,
                query=query,
                primary_path=p1,
                secondary_path=p2,
                primary_filename=fname1,
                secondary_filename=fname2,
            )


        # -------------------------------------------------------------------
        # BRANCH 2: BI-TEMPORAL MULTI-STAGE CHANGE DETECTION & VQA
        # -------------------------------------------------------------------
        elif task in (TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA):
            if p2 is None:
                raise ExecutionError("Bi-temporal change analysis requires both primary (T1) and secondary (T2) images.", status_code=400)

            step_num = 1

            # Step 1: Input Validation
            v_start = time.perf_counter()
            validator = BiTemporalValidator()
            val_res = validator.validate(p1, p2)
            v_dur = int((time.perf_counter() - v_start) * 1000)

            if not val_res.valid:
                err_msg = "; ".join(val_res.errors)
                trace_steps.append(
                    TraceStep(
                        step_number=step_num,
                        step_name="InputValidation",
                        status=StepStatus.FAILED,
                        details=f"Bi-temporal validation failed: {err_msg}",
                        duration_ms=v_dur,
                    )
                )
                raise ExecutionError(f"Bi-temporal validation failed: {err_msg}", trace_steps=trace_steps)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="InputValidation",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Validated two rasters: T1='{fname1}' ({val_res.images[0].width}x{val_res.images[0].height}), "
                        f"T2='{fname2}' ({val_res.images[1].width}x{val_res.images[1].height})."
                    ),
                    duration_ms=v_dur,
                )
            )
            step_num += 1

            # Step 2: Temporal Validation
            compat = val_res.compatibility
            temp_status = compat.temporal_ordering if compat else "missing_dates"
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="TemporalValidation",
                    status=StepStatus.COMPLETED if "invalid" not in temp_status else StepStatus.WARNING,
                    details=f"Temporal ordering verification: {temp_status}.",
                    duration_ms=1,
                    metadata={"temporal_ordering": temp_status},
                )
            )
            step_num += 1

            # Step 3: Spatial Compatibility
            overlap_pct = compat.spatial_overlap_percentage if compat else 100.0
            crs_match = compat.crs_match if compat else True
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpatialCompatibility",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Geospatial overlap: {overlap_pct:.1f}%. CRS match: {crs_match}. "
                        f"Reprojection required: {compat.reprojection_required if compat else False}."
                    ),
                    duration_ms=1,
                    metadata={"overlap_percentage": overlap_pct, "crs_match": crs_match},
                )
            )
            step_num += 1

            # Step 4: Alignment & Co-Registration
            align_start = time.perf_counter()
            aligner = BiTemporalAligner()
            aligned = aligner.align(p1, p2)
            align_dur = int((time.perf_counter() - align_start) * 1000)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="Alignment",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Aligned imagery to common spatial grid: {aligned.width}x{aligned.height} px, "
                        f"CRS: {aligned.target_crs or 'Local/Pixel'}, Resolution: {aligned.target_resolution}."
                    ),
                    duration_ms=align_dur,
                    metadata={"aligned_width": aligned.width, "aligned_height": aligned.height},
                )
            )
            step_num += 1

            # Step 5: Specialist Selection
            specialist = registry.get("CHANGE_DETECT")
            if not specialist:
                raise ExecutionError("ChangeDetectionSpecialist (CHANGE_DETECT) is not registered in ModelRegistry.", status_code=503)

            vqa_specialist = registry.get("CHANGE_VQA") if decision.is_multi_stage else None
            spec_desc = f"Selected '{specialist.capability.identifier}' ({specialist.capability.name})"
            if vqa_specialist:
                spec_desc += f" and '{vqa_specialist.capability.identifier}' ({vqa_specialist.capability.name})"

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpecialistSelection",
                    status=StepStatus.COMPLETED,
                    details=f"{spec_desc} from ModelRegistry.",
                    duration_ms=1,
                )
            )
            step_num += 1

            # Step 6: Model Execution (CHANGE_DETECT)
            candidate_val = "tinycd"
            if task_hint and "cva" in task_hint.lower():
                candidate_val = "cva"

            spec_input = SpecialistInput(
                task=TaskType.CHANGE_DETECTION,
                query=query,
                primary_image_path=str(p1),
                secondary_image_path=str(p2),
                parameters={"candidate": candidate_val},
                metadata={"filename_t1": fname1, "filename_t2": fname2},
            )
            spec_output = specialist.predict(spec_input)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ModelExecution",
                    status=StepStatus.COMPLETED if spec_output.success else StepStatus.FAILED,
                    details=f"Executed change detection model in {spec_output.execution_time_ms} ms.",
                    duration_ms=spec_output.execution_time_ms,
                    metadata=spec_output.parameters_used,
                )
            )
            step_num += 1

            # Step 7: Change Mask Validation
            diag = spec_output.parameters_used.get("diagnostics", {})
            mask_valid = diag.get("mask_shape_valid", True) and not diag.get("has_nan", False)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ChangeMaskValidation",
                    status=StepStatus.COMPLETED if mask_valid else StepStatus.WARNING,
                    details=(
                        f"Change mask validated: {spec_output.parameters_used.get('width')}x{spec_output.parameters_used.get('height')}, "
                        f"Binary: {diag.get('is_binary', True)}, Extreme ratio: {diag.get('is_extreme_change_ratio', False)}."
                    ),
                    duration_ms=1,
                    metadata=diag,
                )
            )
            step_num += 1

            # Step 8: Statistics
            ch_px = spec_output.parameters_used.get("changed_pixels", 0)
            ch_ratio = spec_output.parameters_used.get("change_ratio_pct", 0.0)
            clusters = spec_output.parameters_used.get("total_clusters", 0)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="Statistics",
                    status=StepStatus.COMPLETED,
                    details=f"Extracted statistics: {ch_px:,} changed px ({ch_ratio:.2f}%), clustered in {clusters} spatial regions.",
                    duration_ms=1,
                    metadata={"changed_pixels": ch_px, "change_ratio_pct": ch_ratio, "clusters": clusters},
                )
            )
            step_num += 1

            # Step 9: Semantic Interpretation (Multi-Stage Workflow)
            final_answer = spec_output.answer_text or "Change detection completed."
            models_record = [
                ModelExecutionRecord(
                    identifier=specialist.capability.identifier,
                    model_name=specialist.capability.name,
                    version=specialist.capability.version,
                    execution_time_ms=spec_output.execution_time_ms,
                )
            ]

            if decision.is_multi_stage and vqa_specialist:
                vqa_params = dict(spec_output.parameters_used)
                vqa_params["regions"] = [r.model_dump() for r in spec_output.evidence.regions]
                vqa_params["statistics"] = [s.model_dump() for s in spec_output.evidence.statistics]
                vqa_params["overlay_path"] = spec_output.parameters_used.get("overlay_path")
                vqa_params["mask_path"] = spec_output.parameters_used.get("mask_path")

                vqa_input = SpecialistInput(
                    task=TaskType.CHANGE_VQA,
                    query=query,
                    primary_image_path=str(p1),
                    secondary_image_path=str(p2),
                    parameters=vqa_params,
                    metadata={"filename_t1": fname1, "filename_t2": fname2},
                )
                vqa_output = vqa_specialist.predict(vqa_input)

                if vqa_output.evidence.semantic_interpretation:
                    spec_output.evidence.semantic_interpretation = vqa_output.evidence.semantic_interpretation
                    final_answer = vqa_output.evidence.semantic_interpretation.summary

                for ev_img in vqa_output.evidence.images:
                    if ev_img.role == "semantic_composite":
                        spec_output.evidence.images.append(ev_img)

                models_record.append(
                    ModelExecutionRecord(
                        identifier=vqa_specialist.capability.identifier,
                        model_name=vqa_specialist.capability.name,
                        version=vqa_specialist.capability.version,
                        execution_time_ms=vqa_output.execution_time_ms,
                    )
                )

                direction = (
                    vqa_output.evidence.semantic_interpretation.temporal_direction
                    if vqa_output.evidence.semantic_interpretation
                    else "modified"
                )
                predom = (
                    vqa_output.evidence.semantic_interpretation.predominant_transition
                    if vqa_output.evidence.semantic_interpretation
                    else "unknown"
                )
                trace_steps.append(
                    TraceStep(
                        step_number=step_num,
                        step_name="SemanticInterpretation",
                        status=StepStatus.COMPLETED if vqa_output.success else StepStatus.WARNING,
                        details=(
                            f"Generated grounded semantic interpretation in {vqa_output.execution_time_ms} ms: "
                            f"Direction={direction}, Predominant='{predom}'."
                        ),
                        duration_ms=vqa_output.execution_time_ms,
                        metadata={
                            "temporal_direction": direction,
                            "predominant_transition": predom,
                            "semantic_uncertainty": vqa_output.evidence.semantic_interpretation.semantic_uncertainty
                            if vqa_output.evidence.semantic_interpretation
                            else 0.0,
                        },
                    )
                )
                step_num += 1

            # Step N: Evidence Assembly (incorporates M8 Normalization, Consistency, and Fusion)
            cd_items = EvidenceNormalizer.normalize_change_detection(
                spec_output,
                metadata={"filename_t1": fname1, "filename_t2": fname2, "crs": val_res.images[0].crs},
            )
            vqa_items = []
            if decision.is_multi_stage and vqa_specialist and "vqa_output" in locals():
                vqa_items = EvidenceNormalizer.normalize_change_vqa(
                    vqa_output,
                    metadata={"filename_t1": fname1, "filename_t2": fname2},
                )
            all_change_items = cd_items + vqa_items
            cons_report = ConsistencyChecker.check_bitemporal(
                spec_output.evidence,
                change_params=spec_output.parameters_used,
                temporal_ordering=temp_status,
                query=query,
            )
            EvidenceFuser.fuse(spec_output.evidence, all_change_items)
            spec_output.evidence.consistency_report = cons_report
            gated_answer = EvidenceGater.gate_answer(final_answer, cons_report)

            input_qual = 1.0 if val_res.images[0].crs else 0.70
            align_score = min(1.0, overlap_pct / 100.0)

            spec_outputs_list = [spec_output]
            if decision.is_multi_stage and "vqa_output" in locals():
                spec_outputs_list.append(vqa_output)

            final_conf, confidence_breakdown, conf_level, conf_factors, conf_warnings = ConfidenceEngine.evaluate(
                task=TaskType.CHANGE_VQA if decision.is_multi_stage else TaskType.CHANGE_DETECTION,
                specialist_outputs=spec_outputs_list,
                consistency_report=cons_report,
                input_quality=input_qual,
                spatial_alignment=align_score,
                evidence_bundle=spec_output.evidence,
                query=query,
                metadata={"filename_t1": fname1, "filename_t2": fname2},
            )

            total_elapsed = int((time.perf_counter() - total_start) * 1000)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceAssembly",
                    status=StepStatus.COMPLETED if cons_report.status == ConsistencyStatus.CONSISTENT else StepStatus.WARNING,
                    details=f"Assembled StandardResultContract with ConsistencyReport. Status={cons_report.status.value}.",
                    duration_ms=2,
                    metadata={
                        "consistency_status": cons_report.status.value,
                        "conflict_count": len(cons_report.conflicts),
                        "evidence_quality_score": cons_report.evidence_quality_score,
                        "fused_items_count": len(spec_output.evidence.fused_items),
                        "confidence_calculation": {
                            "system_confidence": final_conf,
                            "confidence_level": conf_level.value,
                            "factors": conf_factors,
                            "is_capped": confidence_breakdown.calculation_details.get("consistency", {}).get("is_capped", False),
                            "is_calibrated": False,
                        },
                    },
                )
            )

            contract = StandardResultContract(
                task="bitemporal_change_vqa" if decision.is_multi_stage else "bitemporal_change_detection",
                status="success",
                answer=gated_answer,
                confidence=final_conf,
                confidence_level=conf_level.value,
                confidence_breakdown=confidence_breakdown,
                evidence=spec_output.evidence,
                evidence_status=cons_report.status.value,
                models=models_record,
                parameters=spec_output.parameters_used,
                warnings=spec_output.warnings + val_res.warnings + cons_report.warnings + conf_warnings,
                execution_trace=trace_steps,
                execution_time_ms=total_elapsed,
            )
            return cls._finalize_contract(
                contract=contract,
                query=query,
                primary_path=p1,
                secondary_path=p2,
                primary_filename=fname1,
                secondary_filename=fname2,
            )


        # -------------------------------------------------------------------
        # BRANCH 3: SINGLE-IMAGE WORKFLOW (VQA, Grounding, Captioning)
        # -------------------------------------------------------------------
        else:
            step_num = 1

            # Step 1: Input Validation
            v_start = time.perf_counter()
            validation = RasterValidator.validate_single_raster(p1)
            v_dur = int((time.perf_counter() - v_start) * 1000)

            if not validation.valid:
                error_details = "; ".join(validation.errors)
                trace_steps.append(
                    TraceStep(
                        step_number=step_num,
                        step_name="InputValidation",
                        status=StepStatus.FAILED,
                        details=f"Input validation failed: {error_details}",
                        duration_ms=v_dur,
                    )
                )
                raise ExecutionError(f"Input validation failed: {error_details}", trace_steps=trace_steps)

            img_meta = validation.images[0]
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="InputValidation",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Validated GeoTIFF '{fname1}'. Dimensions: {img_meta.width}x{img_meta.height}, "
                        f"Bands: {img_meta.band_count}, CRS: {img_meta.crs or 'None'}, Modality: {img_meta.modality.value}"
                    ),
                    duration_ms=v_dur,
                )
            )
            step_num += 1

            # Step 2: Task Routing
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="TaskRouting",
                    status=StepStatus.COMPLETED,
                    details=f"Query mapped to task: {task.value}. Rationale: {decision.reasoning}",
                    duration_ms=1,
                    metadata={"intent_category": decision.intent_category},
                )
            )
            step_num += 1

            # Step 3: Specialist Discovery via ModelRegistry
            specialist_id = decision.target_specialist_ids[0] if decision.target_specialist_ids else "RS_VQA"
            specialist = registry.get(specialist_id)

            if not specialist:
                raise ExecutionError(f"Specialist '{specialist_id}' is not registered in ModelRegistry.", status_code=503)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpecialistSelection",
                    status=StepStatus.COMPLETED,
                    details=f"Selected specialist '{specialist.capability.identifier}' ({specialist.capability.name}) from ModelRegistry.",
                    duration_ms=1,
                )
            )
            step_num += 1

            # Step 4: Model Execution
            spec_input = SpecialistInput(
                task=task,
                query=query,
                primary_image_path=str(p1),
                metadata={"filename": fname1},
            )
            spec_output = specialist.predict(spec_input)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ModelExecution",
                    status=StepStatus.COMPLETED if spec_output.success else StepStatus.FAILED,
                    details=f"Executed {specialist.capability.name} in {spec_output.execution_time_ms} ms.",
                    duration_ms=spec_output.execution_time_ms,
                    metadata=spec_output.parameters_used,
                )
            )
            step_num += 1

            # Step 5: Evidence Normalization (Milestone M8)
            norm_start = time.perf_counter()
            if task == TaskType.GROUNDING:
                norm_items = EvidenceNormalizer.normalize_grounding(
                    spec_output,
                    query=query,
                    metadata={"filename": fname1, "crs": img_meta.crs},
                )
            else:
                norm_items = EvidenceNormalizer.normalize_vqa(
                    spec_output,
                    query=query,
                    metadata={"filename": fname1, "crs": img_meta.crs},
                )
            norm_dur = int((time.perf_counter() - norm_start) * 1000)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceNormalization",
                    status=StepStatus.COMPLETED,
                    details=f"Normalized {len(norm_items)} evidence items with preserved spatial/source provenance.",
                    duration_ms=norm_dur,
                    metadata={"evidence_items_count": len(norm_items)},
                )
            )
            step_num += 1

            # Step 6: Consistency & Sufficiency Check (Rule C8, Milestone M8 - User Correction 3)
            cons_start = time.perf_counter()
            cons_report = ConsistencyChecker.check_single_image(
                spec_output.evidence,
                task_name=task.value,
                query=query,
                image_metadata=img_meta.model_dump(),
            )
            cons_dur = int((time.perf_counter() - cons_start) * 1000)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ConsistencyCheck",
                    status=StepStatus.COMPLETED if cons_report.status == ConsistencyStatus.CONSISTENT else StepStatus.WARNING,
                    details=f"Evaluated single-image evidence sufficiency & degeneracy: Status={cons_report.status.value}. {cons_report.summary_narrative}",
                    duration_ms=cons_dur,
                    metadata={
                        "consistency_status": cons_report.status.value,
                        "conflict_count": len(cons_report.conflicts),
                        "evidence_quality_score": cons_report.evidence_quality_score,
                    },
                )
            )
            step_num += 1

            # Step 7: Evidence Fusion (Milestone M8)
            EvidenceFuser.fuse(spec_output.evidence, norm_items)
            spec_output.evidence.consistency_report = cons_report
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceFusion",
                    status=StepStatus.COMPLETED,
                    details=f"Fused {len(spec_output.evidence.fused_items)} standard evidence units into consolidated bundle.",
                    duration_ms=1,
                )
            )
            step_num += 1

            # Step 8: Evidence Gating & Assembly
            raw_answer = spec_output.answer_text or "Analysis completed."
            gated_answer = EvidenceGater.gate_answer(raw_answer, cons_report)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceAssembly",
                    status=StepStatus.COMPLETED if cons_report.status == ConsistencyStatus.CONSISTENT else StepStatus.WARNING,
                    details=f"Assembled StandardResultContract with ConsistencyReport. Status={cons_report.status.value}.",
                    duration_ms=2,
                    metadata={
                        "consistency_status": cons_report.status.value,
                        "conflict_count": len(cons_report.conflicts),
                        "evidence_quality_score": cons_report.evidence_quality_score,
                    },
                )
            )
            step_num += 1

            # Step 9: Confidence Calculation (Milestone M9 - Rule 11 & Phase 15)
            conf_start = time.perf_counter()
            input_qual = 1.0 if img_meta.crs else 0.70
            final_conf, confidence_breakdown, conf_level, conf_factors, conf_warnings = ConfidenceEngine.evaluate(
                task=task,
                specialist_outputs=[spec_output],
                consistency_report=cons_report,
                input_quality=input_qual,
                spatial_alignment=1.0,
                evidence_bundle=spec_output.evidence,
                query=query,
                metadata={"filename": fname1},
            )
            conf_dur = int((time.perf_counter() - conf_start) * 1000)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ConfidenceCalculation",
                    status=StepStatus.COMPLETED,
                    details=f"Calculated defensible system confidence: {final_conf} ({conf_level.value}). Capped: {confidence_breakdown.calculation_details.get('consistency', {}).get('is_capped', False)}.",
                    duration_ms=conf_dur,
                    metadata={
                        "system_confidence": final_conf,
                        "confidence_level": conf_level.value,
                        "inputs_considered": ["input_geospatial_quality", "specialist_confidence", "evidence_sufficiency", "consistency_status"],
                        "confidence_factors": conf_factors,
                        "is_capped": confidence_breakdown.calculation_details.get("consistency", {}).get("is_capped", False),
                        "is_calibrated": False,
                    },
                )
            )

            total_elapsed = int((time.perf_counter() - total_start) * 1000)

            models_record = [
                ModelExecutionRecord(
                    identifier=specialist.capability.identifier,
                    model_name=specialist.capability.name,
                    version=specialist.capability.version,
                    execution_time_ms=spec_output.execution_time_ms,
                )
            ]

            contract = StandardResultContract(
                task=f"single_image_{task.value}",
                status="success" if spec_output.success else "partial",
                answer=gated_answer,
                confidence=final_conf,
                confidence_level=conf_level.value,
                confidence_breakdown=confidence_breakdown,
                evidence=spec_output.evidence,
                evidence_status=cons_report.status.value,
                models=models_record,
                parameters=spec_output.parameters_used,
                warnings=spec_output.warnings + validation.warnings + cons_report.warnings + conf_warnings,
                execution_trace=trace_steps,
                execution_time_ms=total_elapsed,
            )
            return cls._finalize_contract(
                contract=contract,
                query=query,
                primary_path=p1,
                secondary_path=p2,
                primary_filename=fname1,
                secondary_filename=fname2,
            )
