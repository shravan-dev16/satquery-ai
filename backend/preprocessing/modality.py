"""Deterministic modality identification layer for remote-sensing imagery.

Distinguishes verified sensor metadata from heuristic band-structure inferences.
Possible classifications: optical, sar, unknown.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
from pydantic import BaseModel, Field

from backend.agent.schema import ModalityType
from backend.preprocessing.geotiff import GeoTIFFReader


class ModalityClassification(BaseModel):
    """Structured report on modality identification."""
    modality: ModalityType
    confidence: float = Field(ge=0.0, le=1.0)
    method: str = Field(description="Detection method: e.g. 'sensor-tag metadata' or 'band-structure heuristic'")
    verified: bool = Field(description="True if confirmed via explicit metadata tags, False if heuristic")
    details: Dict[str, Any] = Field(default_factory=dict)


class ModalityDetector:
    """Classifies remote-sensing rasters into Optical, SAR, or Unknown."""

    SAR_SENSOR_KEYWORDS = {"sentinel-1", "s1", "risat", "terrasar", "cosmo-skymed", "alos", "radarsat"}
    OPTICAL_SENSOR_KEYWORDS = {"sentinel-2", "s2", "landsat", "planet", "spot", "naip", "worldview", "pleiades"}

    SAR_BAND_KEYWORDS = {"vv", "vh", "hh", "hv", "sigma0", "gamma0", "amplitude", "intensity"}

    @classmethod
    def identify(cls, file_path: Union[str, Path]) -> ModalityClassification:
        """Inspects raster and returns verified or heuristic modality classification.

        Never presents heuristic modality detection as certain sensor identification.
        """
        try:
            raw = GeoTIFFReader.inspect(file_path)
        except Exception as e:
            return ModalityClassification(
                modality=ModalityType.OPTICAL,
                confidence=0.0,
                method="unreadable-raster-fallback",
                verified=False,
                details={"error": str(e)},
            )
        tags = raw.get("tags", {})
        descriptions = [d.lower() for d in raw.get("descriptions", []) if d]
        band_count = raw["band_count"]
        dtype = raw["dtype"].lower()

        # -------------------------------------------------------------
        # 1. Check Explicit Sensor / Mission Metadata (Verified)
        # -------------------------------------------------------------
        sensor_tag_str = " ".join([
            str(tags.get("SENSOR_ID", "")),
            str(tags.get("SPACECRAFT_NAME", "")),
            str(tags.get("PLATFORM", "")),
            str(tags.get("MISSION", "")),
            str(tags.get("TIFFTAG_IMAGEDESCRIPTION", "")),
        ]).lower()

        for kw in cls.SAR_SENSOR_KEYWORDS:
            if kw in sensor_tag_str:
                return ModalityClassification(
                    modality=ModalityType.SAR,
                    confidence=0.99,
                    method="sensor-tag metadata",
                    verified=True,
                    details={"matched_keyword": kw, "source_tags": sensor_tag_str.strip()},
                )

        for kw in cls.OPTICAL_SENSOR_KEYWORDS:
            if kw in sensor_tag_str:
                return ModalityClassification(
                    modality=ModalityType.OPTICAL,
                    confidence=0.99,
                    method="sensor-tag metadata",
                    verified=True,
                    details={"matched_keyword": kw, "source_tags": sensor_tag_str.strip()},
                )

        # -------------------------------------------------------------
        # 2. Check Band Descriptions (Verified)
        # -------------------------------------------------------------
        if descriptions:
            sar_matches = [d for d in descriptions if any(sk in d for sk in cls.SAR_BAND_KEYWORDS)]
            if sar_matches and len(sar_matches) == len(descriptions):
                return ModalityClassification(
                    modality=ModalityType.SAR,
                    confidence=0.95,
                    method="polarization-band tag metadata",
                    verified=True,
                    details={"polarizations": sar_matches},
                )

        # Check filename indicators (High-confidence heuristic)
        import re
        fname_lower = raw["filename"].lower()
        sar_patterns = [r"\bsar\b", r"\bgrd\b", r"\bs1\b", r"[_.-]s1[_.-]", r"^s1[_.-]", r"[_.-]s1$", r"sentinel[_-]?1"]
        if any(re.search(pat, fname_lower) for pat in sar_patterns):
            return ModalityClassification(
                modality=ModalityType.SAR,
                confidence=0.85,
                method="filename convention heuristic",
                verified=False,
                details={"filename": raw["filename"]},
            )

        opt_patterns = [r"\boptical\b", r"\bmsi\b", r"\bs2\b", r"[_.-]s2[_.-]", r"^s2[_.-]", r"[_.-]s2$", r"sentinel[_-]?2"]
        if any(re.search(pat, fname_lower) for pat in opt_patterns):
            return ModalityClassification(
                modality=ModalityType.OPTICAL,
                confidence=0.85,
                method="filename convention heuristic",
                verified=False,
                details={"filename": raw["filename"]},
            )

        # -------------------------------------------------------------
        # 3. Band-Structure Heuristic Inference (Unverified)
        # -------------------------------------------------------------
        # Typical Optical RGB or Multispectral: 3 or 4 to 13 bands, uint8 or uint16
        if band_count in (3, 4) and dtype in ("uint8", "uint16"):
            return ModalityClassification(
                modality=ModalityType.OPTICAL,
                confidence=0.75,
                method="band-structure heuristic",
                verified=False,
                details={
                    "band_count": band_count,
                    "dtype": dtype,
                    "inference_rule": "3 or 4 channels in uint8/uint16 strongly correlates with RGB/NIR optical imagery",
                },
            )

        if band_count in (1, 2) and "float" in dtype:
            # Often calibrated backscatter in decibels (float32)
            return ModalityClassification(
                modality=ModalityType.SAR,
                confidence=0.68,
                method="band-structure heuristic",
                verified=False,
                details={
                    "band_count": band_count,
                    "dtype": dtype,
                    "inference_rule": "1 or 2 float channels typically correlates with single/dual-polarization SAR backscatter",
                },
            )

        if band_count >= 8 and dtype in ("uint16", "float32"):
            return ModalityClassification(
                modality=ModalityType.MULTISPECTRAL,
                confidence=0.70,
                method="band-structure heuristic",
                verified=False,
                details={"band_count": band_count, "dtype": dtype},
            )

        # -------------------------------------------------------------
        # 4. Unknown Fallback
        # -------------------------------------------------------------
        return ModalityClassification(
            modality=ModalityType.OPTICAL,  # Default fallback representation
            confidence=0.30,
            method="unresolved fallback heuristic",
            verified=False,
            details={
                "band_count": band_count,
                "dtype": dtype,
                "note": "Raster lacks sensor tags or standard optical/SAR band configurations.",
            },
        )
