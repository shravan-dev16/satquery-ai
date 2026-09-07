"""GeoTIFF ingestion and geospatial metadata extraction.

Safely reads geospatial rasters using Rasterio, extracting coordinate reference systems,
affine transforms, geographic bounds, radiometric parameters, and coordinate units.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import rasterio
from rasterio.errors import RasterioIOError


class GeospatialIngestionError(Exception):
    """Base exception for geospatial raster ingestion failures."""
    pass


class CorruptedRasterError(GeospatialIngestionError):
    """Raised when the raster file has corrupted headers or unreadable blocks."""
    pass


class UnsupportedFormatError(GeospatialIngestionError):
    """Raised when the file format cannot be parsed by GDAL/Rasterio."""
    pass


class UnreadableRasterError(GeospatialIngestionError):
    """Raised when the file path is non-existent, unreadable, or locked."""
    pass


class GeoTIFFReader:
    """Safe reader for GeoTIFF and geospatial rasters."""

    SUPPORTED_DRIVERS = {"GTiff", "PNG", "JPEG", "JP2OpenJPEG", "NITF"}

    @staticmethod
    def inspect(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Safely inspects a raster file and returns raw geospatial metadata.

        Does not silently infer metadata that does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise UnreadableRasterError(f"Raster file not found: {path}")

        if not path.is_file():
            raise UnreadableRasterError(f"Path is not a regular file: {path}")

        if path.stat().st_size == 0:
            raise CorruptedRasterError(f"Raster file is empty (0 bytes): {path}")

        try:
            with rasterio.open(path) as src:
                driver = src.driver
                if driver not in GeoTIFFReader.SUPPORTED_DRIVERS:
                    raise UnsupportedFormatError(
                        f"Driver '{driver}' is not supported. Supported drivers: {GeoTIFFReader.SUPPORTED_DRIVERS}"
                    )

                # Extract CRS and determine coordinate units
                crs_obj = src.crs
                crs_str: Optional[str] = None
                coordinate_units: str = "unknown"

                if crs_obj is not None:
                    crs_str = crs_obj.to_string()
                    # Determine units from CRS
                    if crs_obj.is_projected:
                        coordinate_units = getattr(crs_obj, "linear_units", "metre") or "metre"
                    elif crs_obj.is_geographic:
                        coordinate_units = "degree"

                # Extract affine transform tuple (c, a, b, f, d, e)
                # affine matrix: | a  b  c |
                #                | d  e  f |
                transform = src.transform
                transform_tuple = tuple(transform)[:6]

                # Extract bounds: (left, bottom, right, top)
                bounds = src.bounds
                bounds_tuple = (bounds.left, bounds.bottom, bounds.right, bounds.top)

                # Extract resolution: (res_x, res_y)
                res = src.res
                res_tuple = (float(res[0]), float(res[1]))

                # Extract tags (TIFF tags, GDAL metadata)
                tags = dict(src.tags())
                descriptions = list(src.descriptions)

                # Check nodata without inventing
                nodata = src.nodata
                nodata_val = float(nodata) if nodata is not None else None

                return {
                    "filename": path.name,
                    "filepath": str(path.resolve()),
                    "driver": driver,
                    "width": int(src.width),
                    "height": int(src.height),
                    "band_count": int(src.count),
                    "dtype": str(src.dtypes[0]),
                    "crs": crs_str,
                    "coordinate_units": coordinate_units,
                    "transform": transform_tuple,
                    "bounds": bounds_tuple,
                    "resolution": res_tuple,
                    "nodata": nodata_val,
                    "tags": tags,
                    "descriptions": descriptions,
                    "colorinterp": [ci.name for ci in src.colorinterp],
                }

        except (UnsupportedFormatError, CorruptedRasterError, UnreadableRasterError):
            raise
        except RasterioIOError as e:
            err_msg = str(e).lower()
            if "not recognized as a supported file format" in err_msg or "cannot open" in err_msg:
                raise CorruptedRasterError(f"Corrupted or unrecognized raster file '{path.name}': {e}") from e
            raise UnreadableRasterError(f"Rasterio I/O error reading '{path.name}': {e}") from e
        except Exception as e:
            raise CorruptedRasterError(f"Failed to parse raster '{path.name}': {e}") from e

    @staticmethod
    def read_bands(
        file_path: Union[str, Path],
        band_indices: Optional[Tuple[int, ...]] = None,
        max_dimension: Optional[int] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Safely reads raster band data into a numpy array [bands, height, width]."""
        meta = GeoTIFFReader.inspect(file_path)
        path = Path(file_path)

        with rasterio.open(path) as src:
            if band_indices:
                data = src.read(indexes=band_indices)
            else:
                data = src.read()

        return data, meta

    @staticmethod
    def read_normalized_rgb(file_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reads raster and converts to a normalized RGB uint8 image [height, width, 3] for visualization/VLM.

        Performs 2nd-98th percentile contrast stretching for continuous reflectance bands.
        """
        data, meta = GeoTIFFReader.read_bands(file_path)
        count = data.shape[0]

        if count >= 3:
            rgb = data[:3].astype(np.float32)
        elif count == 1:
            rgb = np.repeat(data[:1].astype(np.float32), 3, axis=0)
        elif count == 2:
            # 2 bands (e.g. SAR VV/VH): add ratio as 3rd channel
            ch1 = data[0].astype(np.float32)
            ch2 = data[1].astype(np.float32)
            ratio = ch1 - ch2
            rgb = np.stack([ch1, ch2, ratio], axis=0)

        # Percentile contrast stretch per band
        stretched = np.zeros_like(rgb, dtype=np.uint8)
        for i in range(3):
            band = rgb[i]
            # Ignore nodata if present
            valid_mask = np.isfinite(band)
            if meta.get("nodata") is not None:
                valid_mask &= (band != meta["nodata"])

            if np.any(valid_mask):
                p2, p98 = np.percentile(band[valid_mask], (2, 98))
                if p98 > p2:
                    clipped = np.clip(band, p2, p98)
                    norm = ((clipped - p2) / (p98 - p2) * 255.0).astype(np.uint8)
                else:
                    norm = np.clip(band, 0, 255).astype(np.uint8)
            else:
                norm = np.zeros_like(band, dtype=np.uint8)

            stretched[i] = norm

        # Transpose from [3, H, W] to [H, W, 3]
        rgb_hwc = np.transpose(stretched, (1, 2, 0))
        return rgb_hwc, meta
