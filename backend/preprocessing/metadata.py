"""Geospatial metadata extraction and formatting.

Converts raw inspection dicts from GeoTIFFReader into typed ImageMetadata
compatible with backend.agent.schema.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from backend.agent.schema import ImageMetadata, ModalityType
from backend.preprocessing.geotiff import GeoTIFFReader


class MetadataExtractor:
    """Extracts and normalizes metadata from geospatial imagery."""

    @staticmethod
    def extract_image_metadata(
        file_path: Union[str, Path],
        inferred_modality: ModalityType = ModalityType.OPTICAL,
    ) -> ImageMetadata:
        """Inspects a raster file and returns a validated ImageMetadata instance."""
        raw = GeoTIFFReader.inspect(file_path)

        # Look for acquisition date in standard tags without inventing
        tags = raw.get("tags", {})
        acq_date = (
            tags.get("ACQUISITION_DATETIME")
            or tags.get("TIFFTAG_DATETIME")
            or tags.get("acquisition_date")
            or tags.get("DATETIME")
        )

        sensor = (
            tags.get("SENSOR_ID")
            or tags.get("SPACECRAFT_NAME")
            or tags.get("PLATFORM")
            or tags.get("MISSION")
        )

        return ImageMetadata(
            filename=raw["filename"],
            format=raw["driver"],
            width=raw["width"],
            height=raw["height"],
            band_count=raw["band_count"],
            crs=raw["crs"],
            resolution=raw["resolution"],
            bounds=raw["bounds"],
            nodata=raw["nodata"],
            dtype=raw["dtype"],
            modality=inferred_modality,
            acquisition_date=acq_date,
            sensor=sensor,
        )

    @staticmethod
    def format_spatial_summary(meta: ImageMetadata) -> Dict[str, Any]:
        """Formats spatial coverage and resolution for user-facing evidence."""
        return {
            "dimensions": f"{meta.width} x {meta.height}",
            "band_count": meta.band_count,
            "crs": meta.crs or "Unprojected (No CRS)",
            "ground_sample_distance_m": meta.resolution[0] if meta.resolution else None,
            "bounding_box": meta.bounds,
            "sensor": meta.sensor or "Unspecified",
            "acquisition_date": meta.acquisition_date or "Unspecified",
        }
