"""Scene Captioning Specialist interface (M2-B Preparation).

Provides the BaseSpecialist interface and ModelRegistry registration for RS_CAPTION,
ready for full captioning implementation in subsequent milestones.
"""

import logging
from typing import Optional

from backend.agent.schema import (
    EvidenceBundle,
    ModalityType,
    ModelCapability,
    SpecialistInput,
    SpecialistOutput,
    TaskType,
)
from backend.models.base import BaseSpecialist

logger = logging.getLogger(__name__)


class RemoteSensingCaptionSpecialist(BaseSpecialist):
    """Specialist interface for remote-sensing scene description and captioning."""

    def __init__(self, model_id: str = "Qwen/Qwen2-VL-2B-Instruct", device: str = "cpu") -> None:
        super().__init__(device=device)
        self.model_id = model_id

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            identifier="RS_CAPTION",
            name=f"Remote-Sensing Scene Captioning ({self.model_id})",
            version="1.0.0",
            task=TaskType.CAPTION,
            supported_modalities=[ModalityType.OPTICAL, ModalityType.MULTISPECTRAL],
            supported_input_count=[1],
            requires_gpu=True,
            vram_budget_mb=4500,
            confidence_available=True,
        )

    def load(self) -> None:
        self._is_loaded = True

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        """Executes captioning. Minimal placeholder returning un-executed notice."""
        return SpecialistOutput(
            specialist_id=self.capability.identifier,
            success=True,
            answer_text="Scene captioning interface initialized. Full captioning evaluation scheduled.",
            confidence=0.50,
            evidence=EvidenceBundle(),
            parameters_used={"model_id": self.model_id},
            warnings=["Scene captioning is in architecture preparation stage (M2-B)."],
        )

    def health_check(self) -> bool:
        return True
