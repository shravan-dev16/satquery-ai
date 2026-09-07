"""Dataset evaluation manifests and sample schemas for remote-sensing benchmarks.

Defines:
- SceneCategory enumeration covering 8 operational land-cover and nuisance categories
- EvaluationSample representation supporting both quantitative (with GT mask) and qualitative samples
- EvaluationManifest container with serialization, filtering, and backward compatibility
"""

from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field


class SceneCategory(str, Enum):
    """Standardized remote-sensing scene and land-cover categories."""

    URBAN = "urban"
    FOREST = "forest"
    AGRICULTURE = "agriculture"
    WATER = "water"
    COASTAL = "coastal"
    MINING = "mining"
    INFRASTRUCTURE = "infrastructure"
    NUISANCE_VARIATION = "nuisance_variation"


class EvaluationSample(BaseModel):
    """Represents a single bi-temporal observation pair in a benchmark evaluation split."""

    sample_id: str = Field(..., description="Unique deterministic identifier for the sample")
    filename: str = Field(..., description="Base filename of observation")
    dataset: str = Field(..., description="Source dataset name (e.g. LEVIR-CD, DiverseRS)")
    split: str = Field(default="val", description="Evaluation split (val, test)")
    license: str = Field(default="CC-BY-4.0", description="Data license")
    dimensions: List[int] = Field(..., description="[width, height] in pixels")
    total_pixels: int = Field(..., description="Total pixel count (width x height)")
    changed_pixels: Optional[int] = Field(default=None, description="Ground truth changed pixels (if quantitative)")
    change_ratio_pct: Optional[float] = Field(default=None, description="Ground truth change ratio percentage")
    t1_path: str = Field(..., description="Relative or absolute path to Time 1 (earlier) image")
    t2_path: str = Field(..., description="Relative or absolute path to Time 2 (later) image")
    gt_mask_path: Optional[str] = Field(default=None, description="Path to ground truth binary mask (None if qualitative)")
    scene_category: SceneCategory = Field(default=SceneCategory.URBAN, description="Operational land-cover category")
    is_quantitative: bool = Field(default=True, description="True if ground truth mask exists for metric calculation")
    nuisance_type: Optional[str] = Field(
        default=None,
        description="Nuisance type if applicable: illumination, seasonal_color, shadow, registration",
    )
    crs: Optional[str] = Field(default=None, description="Projected or geographic Coordinate Reference System")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional geospatial or sensor telemetry")


class EvaluationManifest(BaseModel):
    """Container for a versioned, reproducible benchmark evaluation manifest."""

    manifest_id: str = Field(..., description="Identifier for manifest (e.g. levir_cd_val_20, diverse_rs_v1)")
    dataset_name: str = Field(..., description="Primary dataset name")
    version: str = Field(default="1.0.0", description="Manifest format version")
    description: str = Field(..., description="Summary of dataset and evaluation role")
    license: str = Field(..., description="Dataset license")
    samples: List[EvaluationSample] = Field(..., description="Ordered list of evaluation samples")

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def quantitative_count(self) -> int:
        return sum(1 for s in self.samples if s.is_quantitative)

    @property
    def qualitative_count(self) -> int:
        return sum(1 for s in self.samples if not s.is_quantitative)

    def filter_quantitative(self) -> List[EvaluationSample]:
        """Returns only samples that possess verified ground truth masks."""
        return [s for s in self.samples if s.is_quantitative and s.gt_mask_path is not None]

    def filter_by_category(self, category: SceneCategory) -> List[EvaluationSample]:
        """Filters samples by operational scene category."""
        return [s for s in self.samples if s.scene_category == category]

    def save(self, path: Union[str, Path]) -> Path:
        """Serializes manifest to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return p

    @classmethod
    def load(cls, path: Union[str, Path]) -> "EvaluationManifest":
        """Loads manifest from file, with backward-compatibility for plain sample lists."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            # Backward-compatibility: load raw sample list (e.g. existing levir_cd_subset.json)
            samples = []
            for item in data:
                # Ensure default values if missing in older schema
                if "scene_category" not in item:
                    item["scene_category"] = SceneCategory.URBAN.value
                if "is_quantitative" not in item:
                    item["is_quantitative"] = (item.get("gt_mask_path") is not None)
                samples.append(EvaluationSample(**item))

            return cls(
                manifest_id=p.stem,
                dataset_name=samples[0].dataset if samples else "Unknown",
                version="1.0.0",
                description=f"Manifest loaded from {p.name}",
                license=samples[0].license if samples else "Open",
                samples=samples,
            )

        return cls.model_validate(data)
