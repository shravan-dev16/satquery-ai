"""Evidence processing, consistency checking, and confidence estimation package."""

from backend.evidence.confidence import ConfidenceEngine
from backend.evidence.consistency import ConsistencyChecker, EvidenceGater
from backend.evidence.fusion import EvidenceFuser, EvidenceNormalizer
from backend.evidence.spatial import (
    GeospatialRegion,
    PixelBoundingBox,
    SpatialTransformer,
)

__all__ = [
    "PixelBoundingBox",
    "SpatialTransformer",
    "GeospatialRegion",
    "EvidenceNormalizer",
    "EvidenceFuser",
    "ConsistencyChecker",
    "EvidenceGater",
    "ConfidenceEngine",
]
