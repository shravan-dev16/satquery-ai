"""Input validation logic for remote-sensing imagery.

Implements AGENTS.md Rule 7 (Input Validation) ensuring corrupted, unprojected,
or invalid files are caught with structured diagnostics.
"""

from pathlib import Path
from typing import List, Optional, Union

from backend.agent.schema import ImageMetadata, ModalityType, ValidationResult
from backend.preprocessing.geotiff import (
    CorruptedRasterError,
    GeoTIFFReader,
    GeospatialIngestionError,
    UnreadableRasterError,
    UnsupportedFormatError,
)
from backend.preprocessing.metadata import MetadataExtractor
from backend.preprocessing.modality import ModalityDetector


class RasterValidator:
    """Pre-flight validator for single and paired geospatial rasters."""

    @classmethod
    def validate_single_raster(cls, file_path: Union[str, Path]) -> ValidationResult:
        """Validates an uploaded raster file without executing specialist AI models."""
        path = Path(file_path)
        errors: List[str] = []
        warnings: List[str] = []
        images: List[ImageMetadata] = []

        try:
            # 1. Inspect raw raster
            raw_meta = GeoTIFFReader.inspect(path)

            # 2. Modality detection (distinguishing verified vs heuristic)
            modality_rep = ModalityDetector.identify(path)
            if not modality_rep.verified:
                warnings.append(
                    f"Modality '{modality_rep.modality.value}' determined via {modality_rep.method} "
                    f"(confidence: {modality_rep.confidence:.2f}); unverified by sensor tags."
                )

            # 3. Build structured ImageMetadata
            img_meta = MetadataExtractor.extract_image_metadata(
                path, inferred_modality=modality_rep.modality
            )
            images.append(img_meta)

            # 4. Check CRS
            if not img_meta.crs:
                warnings.append(
                    "Image lacks a Coordinate Reference System (CRS). Spatial coordinates are in pixel space only."
                )

            # 5. Check dimensions
            if img_meta.width < 16 or img_meta.height < 16:
                errors.append(f"Image dimensions ({img_meta.width}x{img_meta.height}) are too small for analysis.")

            # 6. Check band count
            if img_meta.band_count == 0:
                errors.append("Raster contains zero bands.")

        except CorruptedRasterError as e:
            errors.append(f"Corrupted or invalid file: {e}")
        except UnsupportedFormatError as e:
            errors.append(f"Unsupported raster format: {e}")
        except UnreadableRasterError as e:
            errors.append(f"Unreadable file: {e}")
        except GeospatialIngestionError as e:
            errors.append(f"Geospatial ingestion error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error during validation: {e}")

        is_valid = len(errors) == 0
        return ValidationResult(
            valid=is_valid,
            images=images,
            compatibility=None,
            errors=errors,
            warnings=warnings,
        )
