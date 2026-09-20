"""FastAPI Application Entrypoint for SatQuery AI.

Provides RESTful endpoints adhering to docs/API_CONTRACT.md.
Supports single-image VQA, text-guided region grounding, captioning,
and bi-temporal change detection (Milestone M3).
"""

import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
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

# Mount developer test UI at /test-ui
test_ui_dir = Path("frontend/test_ui")
test_ui_dir.mkdir(parents=True, exist_ok=True)
app.mount("/test-ui", StaticFiles(directory=str(test_ui_dir), html=True), name="test_ui")

# Mount demo samples directory for developer test console
demo_dir = Path("datasets/ui_demo")
demo_dir.mkdir(parents=True, exist_ok=True)
app.mount("/api/v1/demo/files", StaticFiles(directory=str(demo_dir)), name="demo_files")


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
async def root(request: Request):
    """Canonical entrypoint for SatQuery AI.

    Serves the polished M12 frontend application to web browsers,
    while preserving backward-compatible JSON status telemetry when requested by API clients.
    """
    accept = request.headers.get("accept", "")
    user_agent = request.headers.get("user-agent", "")
    is_browser = "text/html" in accept or ("Mozilla" in user_agent and "testclient" not in user_agent.lower())
    if is_browser:
        index_file = Path("frontend/index.html")
        if index_file.exists():
            return FileResponse(str(index_file), media_type="text/html")
    return {
        "app": settings.app_name,
        "status": "online",
        "docs": "/docs",
        "api_v1": settings.api_v1_prefix,
    }


@app.get("/index.css", include_in_schema=False)
def get_root_css():
    """Serves root stylesheet for direct root URL navigation."""
    css_file = Path("frontend/index.css")
    if css_file.exists():
        return FileResponse(str(css_file), media_type="text/css")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Stylesheet not found")


@app.get("/app.js", include_in_schema=False)
def get_root_js():
    """Serves root JavaScript application for direct root URL navigation."""
    js_file = Path("frontend/app.js")
    if js_file.exists():
        return FileResponse(str(js_file), media_type="application/javascript")
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Application script not found")


@app.get("/favicon.ico", include_in_schema=False)
def get_favicon():
    """Returns 204 No Content for browser favicon requests."""
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.get("/api", include_in_schema=False)
@app.get("/api/v1", include_in_schema=False)
def api_root_status():
    """Explicit API root telemetry endpoint."""
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


@app.get("/api/v1/demo/manifest")
def get_demo_manifest():
    """Returns manifest of UI demo samples for developer testing console."""
    manifest_path = Path("datasets/ui_demo/manifest.json")
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


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


def _sanitize_error_message(msg: str) -> str:
    """Sanitizes local server filesystem paths from error messages returned to clients."""
    import re
    sanitized = re.sub(r'[A-Za-z]:\\[^"\'\s\n\r]+\\([A-Za-z0-9_.-]+)', r'\1', msg)
    sanitized = re.sub(r'/(?:tmp|home|var)/[^"\'\s\n\r]+/([A-Za-z0-9_.-]+)', r'\1', sanitized)
    return sanitized


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
            clean_msg = _sanitize_error_message(ee.message)
            raise HTTPException(status_code=ee.status_code, detail=clean_msg)

    finally:
        if tmp1_path.exists():
            tmp1_path.unlink()
        if tmp2_path and tmp2_path.exists():
            tmp2_path.unlink()
