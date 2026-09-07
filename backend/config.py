"""System configuration for SatQuery AI.

Defines environment variables, runtime hardware limits, model registry defaults,
and storage paths.
"""

from pathlib import Path
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"
FIXTURES_DIR = BASE_DIR / "tests" / "fixtures"
REPORTS_DIR = BASE_DIR / "reports" / "generated"


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
    validation: ValidationThresholds = Field(default_factory=ValidationThresholds)
    confidence: ConfidenceWeights = Field(default_factory=ConfidenceWeights)


settings = SystemSettings()
