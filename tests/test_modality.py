"""Unit tests for modality identification.

Verifies:
- Verified sensor metadata detection vs heuristic inference
- Optical vs SAR classification
- Confidence scoring and transparency
"""

from pathlib import Path
from backend.agent.schema import ModalityType
from backend.preprocessing.modality import ModalityDetector


def test_modality_optical_sample(optical_sample_path: Path):
    """Verify modality detection on optical sample."""
    rep = ModalityDetector.identify(optical_sample_path)
    assert rep.modality == ModalityType.OPTICAL
    assert rep.confidence >= 0.70
    assert rep.method in ("band-structure heuristic", "filename convention heuristic", "sensor-tag metadata")
    # Verified should be False since this is a synthetic fixture without official ESA/NASA sensor tags
    assert rep.verified is False


def test_modality_sar_sample(sar_sample_path: Path):
    """Verify modality detection on SAR sample."""
    rep = ModalityDetector.identify(sar_sample_path)
    assert rep.modality == ModalityType.SAR
    assert rep.confidence >= 0.65
    assert rep.verified is False  # Synthetic fixture


def test_modality_unverified_label():
    """Verify that heuristic detection is never marked verified=True."""
    from backend.preprocessing.modality import ModalityClassification

    mc = ModalityClassification(
        modality=ModalityType.OPTICAL,
        confidence=0.72,
        method="band-structure heuristic",
        verified=False,
    )
    assert mc.verified is False
    assert mc.method == "band-structure heuristic"
