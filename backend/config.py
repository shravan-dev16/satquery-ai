"""System configuration for SatQuery AI.

Defines environment variables, runtime hardware limits, model registry defaults,
and storage paths.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"
REPORTS_DIR = BASE_DIR / "reports" / "generated"

# Automatically load .env if present (without external dependencies)
_env_file = BASE_DIR / ".env"
if _env_file.exists():
    with open(_env_file, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip("'\"")
                if _k and _k not in os.environ:
                    os.environ[_k] = _v


class ModelConfig(BaseModel):
    """Configuration for specialist models and adapter checkpoints."""

    use_adapted_vlm: bool = Field(
        default_factory=lambda: os.environ.get("SATQUERY_USE_ADAPTED_VLM", "1") == "1",
        description="Enable RS-adapted LoRA VLM adapter",
    )
    adapter_path: str = Field(
        default_factory=lambda: os.environ.get(
            "SATQUERY_ADAPTER_PATH",
            "models/adapters/experiments/qwen_rs_exp_a/best_checkpoint"
            if (BASE_DIR / "models/adapters/experiments/qwen_rs_exp_a/best_checkpoint").exists()
            else "models/adapters/qwen2_vl_rs_lora",
        ),
        description="Filesystem path to RS VLM LoRA checkpoint",
    )
    tinycd_checkpoint: str = Field(
        default_factory=lambda: os.environ.get(
            "SATQUERY_TINYCD_CHECKPOINT",
            "models/checkpoints/tinycd_finetuned.pth",
        ),
        description="Filesystem path to primary TinyCD checkpoint",
    )
    shared_vlm_runtime: bool = Field(
        default_factory=lambda: os.environ.get("SATQUERY_SHARED_VLM_RUNTIME", "1") != "0",
        description="Deduplicate VLM memory across RS_VQA and CHANGE_VQA",
    )


class HardwareConfig(BaseModel):
    """Runtime hardware budget and execution limits."""

    device: str = Field(default="cuda", description="Primary compute device (cuda or cpu)")
    max_vram_allocation_mb: int = Field(
        default=7500,
        description="Peak working VRAM ceiling (MB) to preserve OS/DWM headroom on RTX 4070 (12 GB)",
    )
    enable_mixed_precision: bool = Field(default=True, description="Use FP16/BF16 where applicable")
    batch_size: int = Field(default=1, description="Strictly 1 for real-time interactive inference")
    tile_size: int = Field(default=512, description="Sliding window tile dimension for large rasters")
    tile_overlap: int = Field(default=64, description="Overlap in pixels between adjacent tiles")


class ValidationThresholds(BaseModel):
    """Geospatial pre-flight validation rules."""

    min_spatial_overlap_ratio: float = Field(
        default=0.20, description="Minimum intersection over union (IoU) for image pairs (20%)"
    )
    max_registration_residual_px: float = Field(
        default=3.0, description="Maximum acceptable sub-pixel co-registration residual shift"
    )
    max_image_dimension_px: int = Field(
        default=8192, description="Maximum allowed width/height before requiring downsampling"
    )


class ConfidenceWeights(BaseModel):
    """Evidence-weighted confidence heuristic weights."""

    w_input: float = Field(default=0.20, description="Weight for input quality & nodata fraction")
    w_registration: float = Field(default=0.20, description="Weight for spatial co-registration score")
    w_model: float = Field(default=0.60, description="Weight for model prediction probability")
    max_conflict_penalty: float = Field(default=0.50, description="Maximum penalty for cross-specialist contradiction")


class SystemSettings(BaseModel):
    """Global application settings."""

    app_name: str = "SatQuery AI"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    validation: ValidationThresholds = Field(default_factory=ValidationThresholds)
    confidence: ConfidenceWeights = Field(default_factory=ConfidenceWeights)


settings = SystemSettings()

