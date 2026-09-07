"""Environment verification test.

Verifies:
- Python 3.11
- PyTorch CUDA availability and RTX 4070 GPU detection
- Rasterio import and basic GeoTIFF read/write
"""

import sys
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin


def test_python_version():
    """Verify running on Python 3.11."""
    assert sys.version_info.major == 3
    assert sys.version_info.minor == 11, f"Expected Python 3.11, got {sys.version}"


def test_rasterio_read_write(tmp_path):
    """Verify rasterio can create and read back a valid GeoTIFF with CRS."""
    out_file = tmp_path / "test_write.tif"
    transform = from_origin(100.0, 200.0, 10.0, 10.0)
    data = np.arange(100, dtype=np.uint16).reshape((10, 10))

    with rasterio.open(
        out_file,
        "w",
        driver="GTiff",
        height=10,
        width=10,
        count=1,
        dtype=np.uint16,
        crs="EPSG:32643",
        transform=transform,
    ) as dst:
        dst.write(data, 1)

    assert out_file.exists()
    with rasterio.open(out_file) as src:
        assert src.crs.to_string() == "EPSG:32643"
        read_data = src.read(1)
        assert np.array_equal(read_data, data)


def test_pytorch_cuda_and_gpu():
    """Verify PyTorch detects CUDA and RTX 4070 GPU."""
    try:
        import torch
    except ImportError:
        pytest.fail("PyTorch is not installed in the environment")

    assert torch.__version__ is not None
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch!"
    device_name = torch.cuda.get_device_name(0)
    assert "RTX 4070" in device_name, f"Expected RTX 4070, detected: {device_name}"

    # Simple tensor operation on GPU
    x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    y = x * 2.0
    assert torch.equal(y, torch.tensor([2.0, 4.0, 6.0], device="cuda"))
