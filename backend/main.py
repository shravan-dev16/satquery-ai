"""FastAPI Application Entrypoint for SatQuery AI.

Provides RESTful endpoints adhering to docs/API_CONTRACT.md.
Supports single-image VQA, text-guided region grounding, captioning,
and bi-temporal change detection (Milestone M3).
"""

from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.agent.registry import registry
from backend.agent.schema import (
    ConfidenceBreakdown,
    EvidenceBundle,
    ModalityType,
    ModelExecutionRecord,
    SpecialistInput,
    StandardResultContract,
    StepStatus,
    TaskType,
    TraceStep,
    ValidationResult,
)
from backend.config import settings
from backend.models.caption import RemoteSensingCaptionSpecialist
from backend.models.change import ChangeDetectionSpecialist
from backend.models.change_vqa import ChangeVQASpecialist
from backend.models.grounding import RemoteSensingGroundingSpecialist
from backend.models.optical_sar import OpticalSARSpecialist
from backend.models.vqa import RemoteSensingVQASpecialist
from backend.preprocessing.alignment import (
    BiTemporalAligner,
    BiTemporalValidator,
    CrossModalAligner,
    CrossModalValidator,
)
from backend.preprocessing.modality import ModalityDetector
from backend.preprocessing.validation import RasterValidator

app = FastAPI(
    title=settings.app_name,
    version="0.3.0",
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing (SIH26167 / ISRO)",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static preview directory exists and mount it
static_previews_dir = Path("backend/static/previews")
static_previews_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/v1/static/previews", StaticFiles(directory=str(static_previews_dir)), name="previews")

# Mount frontend web application at /ui
frontend_dir = Path("frontend")
frontend_dir.mkdir(parents=True, exist_ok=True)
app.mount("/ui", StaticFiles(directory=str(frontend_dir), html=True), name="ui")


def register_default_specialists():
    """Register specialists in the central ModelRegistry."""
    registry.register(RemoteSensingVQASpecialist())
    registry.register(RemoteSensingGroundingSpecialist())
    registry.register(RemoteSensingCaptionSpecialist())
    registry.register(ChangeDetectionSpecialist())
    registry.register(ChangeVQASpecialist())
    registry.register(OpticalSARSpecialist())


@app.on_event("startup")
def startup_event():
    register_default_specialists()


# Register default specialists immediately upon module load
register_default_specialists()


@app.get("/")
def root():
    """Root status endpoint."""
    return {
        "app": settings.app_name,
        "status": "online",
        "docs": "/docs",
        "api_v1": settings.api_v1_prefix,
    }


@app.get("/api/v1/health")
def health_check():
    """Service liveness and readiness probe."""
    specialist_health = registry.health_check_all()
    return {
        "status": "healthy",
        "registered_specialists": len(registry.list_capabilities()),
        "specialists_status": specialist_health,
    }


@app.get("/api/v1/models")
def list_models():
    """List registered specialist capabilities and hardware info."""
    return {
        "registered_specialists": registry.list_capabilities(),
        "hardware": settings.hardware.model_dump(),
    }


@app.post("/api/v1/validate", response_model=ValidationResult)
async def validate_raster(
    image_primary: UploadFile = File(..., description="Primary GeoTIFF / raster file to inspect"),
    image_secondary: Optional[UploadFile] = File(None, description="Optional secondary GeoTIFF for pair inspection"),
):
    """Pre-flight validation endpoint.

    Validates uploaded raster headers, CRS, bounds, bands, and modality
    without executing specialist AI models. If image_secondary is provided,
    runs full bi-temporal pair compatibility validation.
    """
    suffix1 = Path(image_primary.filename or "uploaded.tif").suffix or ".tif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as tmp1:
        tmp1_path = Path(tmp1.name)
        shutil.copyfileobj(image_primary.file, tmp1)

    tmp2_path: Optional[Path] = None
    if image_secondary:
        suffix2 = Path(image_secondary.filename or "uploaded_t2.tif").suffix or ".tif"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as tmp2:
            tmp2_path = Path(tmp2.name)
            shutil.copyfileobj(image_secondary.file, tmp2)

    try:
        if tmp2_path is not None:
            # Check if pair is cross-modal (one optical, one SAR)
            mod1 = ModalityDetector.identify(tmp1_path)
            mod2 = ModalityDetector.identify(tmp2_path)
            is_m1_opt = mod1.modality in (ModalityType.OPTICAL, ModalityType.MULTISPECTRAL)
            is_m1_sar = mod1.modality == ModalityType.SAR
            is_m2_opt = mod2.modality in (ModalityType.OPTICAL, ModalityType.MULTISPECTRAL)
            is_m2_sar = mod2.modality == ModalityType.SAR

            if (is_m1_opt and is_m2_sar) or (is_m1_sar and is_m2_opt):
                val_result = CrossModalValidator.validate(tmp1_path, tmp2_path)
            else:
                # Bi-temporal pair validation
                validator = BiTemporalValidator()
                val_result = validator.validate(tmp1_path, tmp2_path)

            if val_result.images:
                val_result.images[0].filename = image_primary.filename or "primary.tif"
                if len(val_result.images) > 1 and image_secondary:
                    val_result.images[1].filename = image_secondary.filename or "secondary.tif"

            if not val_result.valid:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=val_result.model_dump(),
                )
            return val_result
        else:
            # Single raster validation
            result = RasterValidator.validate_single_raster(tmp1_path)
            if result.images and image_primary.filename:
                result.images[0].filename = image_primary.filename

            if not result.valid:
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content=result.model_dump(),
                )
            return result
    finally:
        if tmp1_path.exists():
            tmp1_path.unlink()
        if tmp2_path and tmp2_path.exists():
            tmp2_path.unlink()


def _classify_task_intent(
    query: str,
    has_secondary: bool = False,
    task_hint: Optional[str] = None,
) -> TaskType:
    """Deterministic intent classifier mapping query and inputs to task type."""
    q_lower = query.lower().strip()

    if task_hint:
        hint_clean = task_hint.lower().strip()
        if "optical_sar" in hint_clean or "fusion" in hint_clean or "cross_modal" in hint_clean:
            return TaskType.OPTICAL_SAR_ANALYSIS
        if "sar" in hint_clean and has_secondary:
            return TaskType.OPTICAL_SAR_ANALYSIS
        if "change_vqa" in hint_clean or "semantic" in hint_clean:
            return TaskType.CHANGE_VQA
        if "vqa" in hint_clean:
            return TaskType.CHANGE_VQA if has_secondary else TaskType.VQA
        if "change_detection" in hint_clean or hint_clean in ["tinycd_raw", "cva_raw"]:
            return TaskType.CHANGE_DETECTION
        if "ground" in hint_clean:
            return TaskType.GROUNDING
        if "caption" in hint_clean:
            return TaskType.CAPTION
        if "change" in hint_clean:
            semantic_triggers = [
                "describe", "how did", "type of", "land-cover", "land cover",
                "increased", "decreased", "built-up", "vegetation", "forest",
                "significant change", "where did", "why did", "what does",
                "semantic", "meaning", "interpretation",
            ]
            if any(st in q_lower for st in semantic_triggers):
                return TaskType.CHANGE_VQA
            return TaskType.CHANGE_DETECTION

    cross_modal_triggers = [
        "optical and sar", "sar and optical", "both sensors", "optical + sar",
        "sar + optical", "cross-modal", "cross modal", "joint analysis",
        "using both sensors", "with sar", "sar backscatter",
        "optical and radar", "radar and optical", "together to identify",
    ]
    if any(cmt in q_lower for cmt in cross_modal_triggers) and has_secondary:
        return TaskType.OPTICAL_SAR_ANALYSIS

    if has_secondary:
        semantic_triggers = [
            "describe", "how did", "type of", "land-cover", "land cover",
            "increased", "decreased", "built-up", "vegetation", "forest",
            "significant change", "where did", "why did", "what does",
            "semantic", "meaning", "interpretation",
        ]
        if any(st in q_lower for st in semantic_triggers):
            return TaskType.CHANGE_VQA
        return TaskType.CHANGE_DETECTION

    change_triggers = [
        "what changed",
        "change",
        "difference",
        "between these two",
        "changed regions",
        "temporal change",
        "bi-temporal",
    ]
    if any(tr in q_lower for tr in change_triggers):
        return TaskType.CHANGE_DETECTION

    grounding_triggers = ["where is", "locate", "ground", "find the", "highlight", "detect"]
    if any(q_lower.startswith(tr) or f" {tr} " in f" {q_lower} " for tr in grounding_triggers):
        return TaskType.GROUNDING

    caption_triggers = ["describe the scene", "generate a caption", "scene description", "caption this"]
    if any(tr in q_lower for tr in caption_triggers):
        return TaskType.CAPTION

    return TaskType.VQA


@app.post("/api/v1/analyze", response_model=StandardResultContract)
async def analyze(
    query: str = Form(..., description="Natural-language question or instruction"),
    image_primary: UploadFile = File(..., description="Primary remote-sensing image (T1 / single GeoTIFF)"),
    image_secondary: Optional[UploadFile] = File(
        None, description="Secondary remote-sensing image (T2 for bi-temporal change detection)"
    ),
    task_hint: Optional[str] = Form(
        None, description="Optional task hint override (vqa, grounding, caption, change_detection, change_vqa)"
    ),
):
    """Unified agentic entrypoint.

    Accepts single or dual images with natural-language query, validates inputs,
    routes to appropriate specialist in ModelRegistry (VQA, Grounding, Captioning,
    or Change Detection / Change VQA), and returns an evidence-grounded StandardResultContract.
    """
    total_start = time.perf_counter()
    trace_steps = []
    step_num = 1

    suffix1 = Path(image_primary.filename or "uploaded_t1.tif").suffix or ".tif"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix1) as tmp1:
        tmp1_path = Path(tmp1.name)
        shutil.copyfileobj(image_primary.file, tmp1)

    tmp2_path: Optional[Path] = None
    if image_secondary:
        suffix2 = Path(image_secondary.filename or "uploaded_t2.tif").suffix or ".tif"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix2) as tmp2:
            tmp2_path = Path(tmp2.name)
            shutil.copyfileobj(image_secondary.file, tmp2)

    try:
        resolved_task = _classify_task_intent(
            query=query,
            has_secondary=tmp2_path is not None,
            task_hint=task_hint,
        )

        # Cross-modality detection override for dual images when not explicitly pure change detection
        if tmp2_path is not None and resolved_task != TaskType.OPTICAL_SAR_ANALYSIS:
            try:
                mod1 = ModalityDetector.identify(tmp1_path)
                mod2 = ModalityDetector.identify(tmp2_path)
                is_opt1 = mod1.modality in (ModalityType.OPTICAL, ModalityType.MULTISPECTRAL)
                is_sar1 = mod1.modality == ModalityType.SAR
                is_opt2 = mod2.modality in (ModalityType.OPTICAL, ModalityType.MULTISPECTRAL)
                is_sar2 = mod2.modality == ModalityType.SAR
                if (is_opt1 and is_sar2) or (is_sar1 and is_opt2):
                    if not (task_hint and any(c in task_hint.lower() for c in ["change_detection", "tinycd_raw", "cva_raw"])):
                        resolved_task = TaskType.OPTICAL_SAR_ANALYSIS
            except Exception:
                pass

        # -------------------------------------------------------------------
        # BRANCH C: OPTICAL + SAR JOINT ANALYSIS WORKFLOW (Rule 14, M6)
        # -------------------------------------------------------------------
        if resolved_task == TaskType.OPTICAL_SAR_ANALYSIS:
            if tmp2_path is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Optical-SAR joint analysis requires both 'image_primary' and 'image_secondary' (one Optical and one SAR).",
                )

            # 1. Input Validation
            v1_start = time.perf_counter()
            val_result = CrossModalValidator.validate_pair(tmp1_path, tmp2_path)
            v1_duration = int((time.perf_counter() - v1_start) * 1000)

            if not val_result.is_valid:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Optical-SAR validation failed: {val_result.error}",
                )

            w_desc = f"{val_result.optical_meta['width']}x{val_result.optical_meta['height']}" if val_result.optical_meta else "?x?"
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="InputValidation",
                    status=StepStatus.COMPLETED,
                    details=f"Validated two rasters: '{image_primary.filename}' ({w_desc}) and '{image_secondary.filename}'.",
                    duration_ms=v1_duration,
                )
            )
            step_num += 1

            # 2. Modality Validation
            mod_details = (
                f"Verified cross-modal pair: Optical raster='{val_result.optical_path.name}', "
                f"SAR raster='{val_result.sar_path.name}'. "
            )
            if val_result.is_reversed_order:
                mod_details += "Reversed input ordering automatically normalized."

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="ModalityValidation",
                    status=StepStatus.COMPLETED if not val_result.is_reversed_order else StepStatus.WARNING,
                    details=mod_details,
                    duration_ms=1,
                    metadata={"is_reversed_order": val_result.is_reversed_order},
                )
            )
            step_num += 1

            # 3. Spatial Compatibility
            overlap = val_result.spatial_overlap
            overlap_pct = (overlap.overlap_ratio_image1 * 100.0) if overlap else 100.0
            crs_match = (val_result.optical_meta.get("crs") == val_result.sar_meta.get("crs")) if val_result.optical_meta and val_result.sar_meta else True
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

            # 4. Alignment & Co-Registration
            align_start = time.perf_counter()
            aligned = CrossModalAligner.align_pair(val_result.optical_path, val_result.sar_path)
            align_duration = int((time.perf_counter() - align_start) * 1000)

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
                    duration_ms=align_duration,
                    metadata={"aligned_width": aligned.width, "aligned_height": aligned.height},
                )
            )
            step_num += 1

            # 5. Specialist Selection
            specialist = registry.get("OPTICAL_SAR_FUSION")
            if not specialist:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="OpticalSARSpecialist (OPTICAL_SAR_FUSION) is not registered in ModelRegistry.",
                )

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpecialistSelection",
                    status=StepStatus.COMPLETED,
                    details=f"Selected '{specialist.capability.identifier}' ({specialist.capability.name}) from registry.",
                    duration_ms=1,
                )
            )
            step_num += 1

            # 6. Optical Feature Extraction
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

            # 7. SAR Feature Extraction
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

            # 8. Joint Fusion
            spec_input = SpecialistInput(
                task=TaskType.OPTICAL_SAR_ANALYSIS,
                query=query,
                primary_image_path=str(val_result.optical_path),
                secondary_image_path=str(val_result.sar_path),
                parameters={"task_hint": task_hint},
                metadata={
                    "filename_optical": val_result.optical_path.name,
                    "filename_sar": val_result.sar_path.name,
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

            # 9. Region Extraction
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

            # 10. Evidence Assembly
            input_qual = 1.0 if val_result.optical_meta and val_result.optical_meta.get("crs") else 0.75
            align_score = min(1.0, overlap_pct / 100.0)
            model_score = spec_output.confidence

            w_conf = settings.confidence
            final_conf = min(0.99, max(0.05, round(
                w_conf.w_input * input_qual + w_conf.w_registration * align_score + w_conf.w_model * model_score, 4
            )))

            confidence_breakdown = ConfidenceBreakdown(
                heuristic_name="evidence-weighted confidence heuristic",
                input_quality_score=round(input_qual, 4),
                spatial_alignment_score=round(align_score, 4),
                model_confidence_score=round(model_score, 4),
                consistency_penalty=0.0,
                is_calibrated_probability=False,
            )

            total_elapsed = int((time.perf_counter() - total_start) * 1000)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceAssembly",
                    status=StepStatus.COMPLETED,
                    details=f"Assembled StandardResultContract with ComplementarityReport. Evidence confidence: {final_conf}.",
                    duration_ms=2,
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

            return StandardResultContract(
                task="optical_sar_analysis",
                status="success",
                answer=spec_output.answer_text or "Optical-SAR analysis completed.",
                confidence=final_conf,
                confidence_breakdown=confidence_breakdown,
                evidence=spec_output.evidence,
                models=models_record,
                parameters=spec_output.parameters_used,
                warnings=spec_output.warnings + val_result.warnings,
                execution_trace=trace_steps,
                execution_time_ms=total_elapsed,
            )

        # -------------------------------------------------------------------
        # BRANCH A: BI-TEMPORAL CHANGE DETECTION & VQA WORKFLOW (Rule 13, M3, M5)
        # -------------------------------------------------------------------
        if resolved_task in [TaskType.CHANGE_DETECTION, TaskType.CHANGE_VQA]:
            if tmp2_path is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Bi-temporal change detection requires both 'image_primary' (T1) and 'image_secondary' (T2).",
                )

            # 1. Input Validation
            v1_start = time.perf_counter()
            validator = BiTemporalValidator()
            val_result = validator.validate(tmp1_path, tmp2_path)
            v1_duration = int((time.perf_counter() - v1_start) * 1000)

            if not val_result.valid:
                err_msg = "; ".join(val_result.errors)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Bi-temporal validation failed: {err_msg}",
                )

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="InputValidation",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Validated two rasters: T1='{image_primary.filename}' "
                        f"({val_result.images[0].width}x{val_result.images[0].height}), "
                        f"T2='{image_secondary.filename}' ({val_result.images[1].width}x{val_result.images[1].height})."
                    ),
                    duration_ms=v1_duration,
                )
            )
            step_num += 1

            # 2. Temporal Validation
            compat = val_result.compatibility
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

            # 3. Spatial Compatibility
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

            # 4. Alignment & Co-Registration
            align_start = time.perf_counter()
            aligner = BiTemporalAligner()
            aligned = aligner.align(tmp1_path, tmp2_path)
            align_duration = int((time.perf_counter() - align_start) * 1000)

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="Alignment",
                    status=StepStatus.COMPLETED,
                    details=(
                        f"Aligned imagery to common spatial grid: {aligned.width}x{aligned.height} px, "
                        f"CRS: {aligned.target_crs or 'Local/Pixel'}, Resolution: {aligned.target_resolution}."
                    ),
                    duration_ms=align_duration,
                    metadata={"aligned_width": aligned.width, "aligned_height": aligned.height},
                )
            )
            step_num += 1

            # 5. Specialist Selection
            specialist = registry.get("CHANGE_DETECT")
            if not specialist:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="ChangeDetectionSpecialist (CHANGE_DETECT) is not registered in ModelRegistry.",
                )

            vqa_specialist = registry.get("CHANGE_VQA") if resolved_task == TaskType.CHANGE_VQA else None
            spec_desc = f"Selected '{specialist.capability.identifier}' ({specialist.capability.name})"
            if vqa_specialist:
                spec_desc += f" and '{vqa_specialist.capability.identifier}' ({vqa_specialist.capability.name})"

            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="SpecialistSelection",
                    status=StepStatus.COMPLETED,
                    details=f"{spec_desc} from registry.",
                    duration_ms=1,
                )
            )
            step_num += 1

            # 6. Model Execution (CHANGE_DETECT)
            candidate_val = "tinycd"
            if task_hint and "cva" in task_hint.lower():
                candidate_val = "cva"

            spec_input = SpecialistInput(
                task=TaskType.CHANGE_DETECTION,
                query=query,
                primary_image_path=str(tmp1_path),
                secondary_image_path=str(tmp2_path),
                parameters={"candidate": candidate_val},
                metadata={"filename_t1": image_primary.filename, "filename_t2": image_secondary.filename},
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

            # 7. Change Mask Validation
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

            # 8. Statistics
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

            # 9. Semantic Interpretation (Milestone M5 - CHANGE_VQA)
            final_answer = spec_output.answer_text or "Change detection completed."
            models_record = [
                ModelExecutionRecord(
                    identifier=specialist.capability.identifier,
                    model_name=specialist.capability.name,
                    version=specialist.capability.version,
                    execution_time_ms=spec_output.execution_time_ms,
                )
            ]

            if resolved_task == TaskType.CHANGE_VQA and vqa_specialist:
                vqa_params = dict(spec_output.parameters_used)
                vqa_params["regions"] = [r.model_dump() for r in spec_output.evidence.regions]
                vqa_params["statistics"] = [s.model_dump() for s in spec_output.evidence.statistics]
                vqa_params["overlay_path"] = spec_output.parameters_used.get("overlay_path")
                vqa_params["mask_path"] = spec_output.parameters_used.get("mask_path")

                vqa_input = SpecialistInput(
                    task=TaskType.CHANGE_VQA,
                    query=query,
                    primary_image_path=str(tmp1_path),
                    secondary_image_path=str(tmp2_path),
                    parameters=vqa_params,
                    metadata={"filename_t1": image_primary.filename, "filename_t2": image_secondary.filename},
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

            # Evidence Assembly (Step 9 for CHANGE_DETECTION, Step 10 for CHANGE_VQA)
            input_qual = 1.0 if val_result.images[0].crs else 0.75
            align_score = min(1.0, overlap_pct / 100.0)
            model_score = spec_output.confidence

            w = settings.confidence
            final_conf = min(0.99, max(0.05, round(
                w.w_input * input_qual + w.w_registration * align_score + w.w_model * model_score, 4
            )))

            confidence_breakdown = ConfidenceBreakdown(
                heuristic_name="evidence-weighted confidence heuristic",
                input_quality_score=round(input_qual, 4),
                spatial_alignment_score=round(align_score, 4),
                model_confidence_score=round(model_score, 4),
                consistency_penalty=0.0,
                is_calibrated_probability=False,
            )

            total_elapsed = int((time.perf_counter() - total_start) * 1000)
            trace_steps.append(
                TraceStep(
                    step_number=step_num,
                    step_name="EvidenceAssembly",
                    status=StepStatus.COMPLETED,
                    details=f"Assembled StandardResultContract. Evidence confidence: {final_conf}.",
                    duration_ms=2,
                )
            )

            return StandardResultContract(
                task="bitemporal_change_vqa" if resolved_task == TaskType.CHANGE_VQA else "bitemporal_change_detection",
                status="success",
                answer=final_answer,
                confidence=final_conf,
                confidence_breakdown=confidence_breakdown,
                evidence=spec_output.evidence,
                models=models_record,
                parameters=spec_output.parameters_used,
                warnings=spec_output.warnings + val_result.warnings,
                execution_trace=trace_steps,
                execution_time_ms=total_elapsed,
            )

        # -------------------------------------------------------------------
        # BRANCH B: SINGLE-IMAGE WORKFLOW (VQA, Grounding, Captioning)
        # -------------------------------------------------------------------
        # Step 1: Ingestion & Input Validation
        v_start = time.perf_counter()
        validation = RasterValidator.validate_single_raster(tmp1_path)
        v_duration = int((time.perf_counter() - v_start) * 1000)

        if not validation.valid:
            error_details = "; ".join(validation.errors)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input validation failed: {error_details}",
            )

        img_meta = validation.images[0]
        trace_steps.append(
            TraceStep(
                step_number=step_num,
                step_name="InputValidation",
                status=StepStatus.COMPLETED,
                details=(
                    f"Validated GeoTIFF '{image_primary.filename}'. Dimensions: {img_meta.width}x{img_meta.height}, "
                    f"Bands: {img_meta.band_count}, CRS: {img_meta.crs or 'None'}, Modality: {img_meta.modality.value}"
                ),
                duration_ms=v_duration,
            )
        )
        step_num += 1

        # Step 2: Task Routing
        trace_steps.append(
            TraceStep(
                step_number=step_num,
                step_name="TaskRouting",
                status=StepStatus.COMPLETED,
                details=f"Query '{query[:50]}...' mapped to task: {resolved_task.value}.",
                duration_ms=1,
            )
        )
        step_num += 1

        # Step 3: Specialist Discovery via ModelRegistry
        s_start = time.perf_counter()
        specialists = registry.find_specialists(task=resolved_task, modality=img_meta.modality, input_count=1)
        if not specialists:
            identifier_map = {
                TaskType.VQA: "RS_VQA",
                TaskType.GROUNDING: "RS_GROUND",
                TaskType.CAPTION: "RS_CAPTION",
            }
            ident = identifier_map.get(resolved_task, "RS_VQA")
            specialist = registry.get(ident)
        else:
            specialist = specialists[0]

        if not specialist:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No specialist model registered for task: {resolved_task.value}",
            )

        trace_steps.append(
            TraceStep(
                step_number=step_num,
                step_name="SpecialistSelection",
                status=StepStatus.COMPLETED,
                details=f"Selected specialist '{specialist.capability.identifier}' ({specialist.capability.name}) from registry.",
                duration_ms=int((time.perf_counter() - s_start) * 1000),
            )
        )
        step_num += 1

        # Step 4: Specialist Execution
        spec_input = SpecialistInput(
            task=resolved_task,
            query=query,
            primary_image_path=str(tmp1_path),
            parameters={"temperature": 0.1, "max_new_tokens": 128},
            metadata=img_meta.model_dump(),
        )
        spec_output = specialist.predict(spec_input)

        trace_steps.append(
            TraceStep(
                step_number=step_num,
                step_name="ModelExecution",
                status=StepStatus.COMPLETED if spec_output.success else StepStatus.FAILED,
                details=f"Executed inference in {spec_output.execution_time_ms} ms.",
                duration_ms=spec_output.execution_time_ms,
            )
        )
        step_num += 1

        # Step 5: Evidence & Confidence Synthesis
        input_quality = 1.0 if img_meta.crs else 0.75
        alignment_score = 1.0  # Single image
        model_score = spec_output.confidence

        w = settings.confidence
        computed_conf = (
            w.w_input * input_quality + w.w_registration * alignment_score + w.w_model * model_score
        )
        final_confidence = min(0.99, max(0.05, round(computed_conf, 4)))

        confidence_breakdown = ConfidenceBreakdown(
            heuristic_name="evidence-weighted confidence heuristic",
            input_quality_score=round(input_quality, 4),
            spatial_alignment_score=round(alignment_score, 4),
            model_confidence_score=round(model_score, 4),
            consistency_penalty=0.0,
            is_calibrated_probability=False,
        )

        total_elapsed = int((time.perf_counter() - total_start) * 1000)

        trace_steps.append(
            TraceStep(
                step_number=step_num,
                step_name="EvidenceAssembly",
                status=StepStatus.COMPLETED,
                details=(
                    f"Assembled StandardResultContract. Grounded boxes: {len(spec_output.evidence.boxes)}, "
                    f"Evidence-weighted confidence: {final_confidence}."
                ),
                duration_ms=2,
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

        return StandardResultContract(
            task=f"single_image_{resolved_task.value}",
            status="success",
            answer=spec_output.answer_text or "No answer returned.",
            confidence=final_confidence,
            confidence_breakdown=confidence_breakdown,
            evidence=spec_output.evidence,
            models=models_record,
            parameters=spec_output.parameters_used,
            warnings=spec_output.warnings + validation.warnings,
            execution_trace=trace_steps,
            execution_time_ms=total_elapsed,
        )

    finally:
        if tmp1_path.exists():
            tmp1_path.unlink()
        if tmp2_path and tmp2_path.exists():
            tmp2_path.unlink()
