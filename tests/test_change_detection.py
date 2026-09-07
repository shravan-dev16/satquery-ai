"""Unit tests for change detection specialists (CVA baseline, TinyCD architecture, and facade)."""

import numpy as np
import pytest
import torch

from backend.agent.schema import TaskType
from backend.models.change import (
    ChangeDetectionSpecialist,
    DeterministicCVASpecialist,
    TinyCDSpecialist,
)
from backend.models.tinycd_arch import TinyCD


def test_cva_identical_images_zero_change():
    """Verify deterministic CVA on identical images yields zero change."""
    img = np.random.randint(50, 200, size=(3, 128, 128), dtype=np.uint8)
    cva = DeterministicCVASpecialist(min_threshold=0.15)
    binary_mask, prob_map, diag = cva.predict(img, img)

    assert np.count_nonzero(binary_mask) == 0
    assert diag["cleaned_changed_pixels"] == 0
    assert diag["method"] == "Change Vector Analysis (CVA)"


def test_cva_synthetic_square_change():
    """Verify CVA detects an injected high-contrast block."""
    img1 = np.full((3, 100, 100), 50, dtype=np.uint8)
    img2 = img1.copy()
    img2[:, 30:70, 30:70] = 230  # 40x40 = 1600 px change

    cva = DeterministicCVASpecialist(min_threshold=0.15)
    binary_mask, prob_map, diag = cva.predict(img1, img2)

    changed_px = np.count_nonzero(binary_mask)
    assert changed_px >= 1500, f"Expected ~1600 changed pixels, got {changed_px}"


def test_tinycd_neural_architecture_forward():
    """Verify TinyCD PyTorch architecture initializes and runs forward pass correctly."""
    model = TinyCD(pretrained_backbone=False)
    model.eval()

    t1 = torch.randn(1, 3, 256, 256)
    t2 = torch.randn(1, 3, 256, 256)

    with torch.no_grad():
        out = model(t1, t2)

    assert out.shape == (1, 1, 256, 256), f"Unexpected output shape: {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values produced by TinyCD forward pass"


def test_change_specialist_capability():
    """Verify ChangeDetectionSpecialist declares compliant ModelCapability."""
    spec = ChangeDetectionSpecialist(candidate="cva")
    cap = spec.capability

    assert cap.identifier == "CHANGE_DETECT"
    assert cap.task == TaskType.CHANGE_DETECTION
    assert cap.supported_input_count == [2]
    assert cap.confidence_available is True
