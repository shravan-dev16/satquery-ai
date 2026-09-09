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

from backend.agent import AgentExecutor, AgentPlanner, AgentRouter, ExecutionError
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


@app.get("/health")
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
    """Deterministic intent classifier mapping query and inputs to task type (backwards compatibility)."""
    p1 = Path("primary.tif")
    p2 = Path("secondary.tif") if has_secondary else None
    decision = AgentRouter.route(
        query=query,
        primary_path=p1,
        secondary_path=p2,
        task_hint=task_hint,
    )
    return decision.task


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
    """Unified agentic entrypoint (Milestone M7 Dynamic Orchestration).

    Automatically selects, sequences, and executes specialist models from ModelRegistry
    based on query semantics and input configuration. Rejects impossible combinations
    and produces an observable execution trace.
    """
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
        # 1. Routing
        decision = AgentRouter.route(
            query=query,
            primary_path=tmp1_path,
            secondary_path=tmp2_path,
            task_hint=task_hint,
            primary_filename=image_primary.filename,
            secondary_filename=image_secondary.filename if image_secondary else None,
        )

        # 2. Plan Creation
        plan = AgentPlanner.create_plan(
            decision=decision,
            primary_path=tmp1_path,
            secondary_path=tmp2_path,
            query=query,
            parameters={"task_hint": task_hint},
        )

        # 3. Specialist Execution
        try:
            return AgentExecutor.execute(
                plan=plan,
                decision=decision,
                primary_path=tmp1_path,
                secondary_path=tmp2_path,
                query=query,
                primary_filename=image_primary.filename,
                secondary_filename=image_secondary.filename if image_secondary else None,
                task_hint=task_hint,
            )
        except ExecutionError as ee:
            raise HTTPException(status_code=ee.status_code, detail=ee.message)

    finally:
        if tmp1_path.exists():
            tmp1_path.unlink()
        if tmp2_path and tmp2_path.exists():
            tmp2_path.unlink()
