"""Unit tests for ModelRegistry and BaseSpecialist interface.

Verifies registration, unregistration, capability queries, and health checks
without requiring heavyweight ML dependencies.
"""

import pytest

from backend.agent.registry import ModelRegistry
from backend.agent.schema import (
    EvidenceBundle,
    ModalityType,
    ModelCapability,
    SpecialistInput,
    SpecialistOutput,
    TaskType,
)
from backend.models.base import BaseSpecialist


class MockSpecialist(BaseSpecialist):
    """Mock specialist for testing the registry interface."""

    def __init__(self, identifier: str = "MOCK_VQA", task: TaskType = TaskType.VQA) -> None:
        super().__init__(device="cpu")
        self._identifier = identifier
        self._task = task
        self._healthy = True

    @property
    def capability(self) -> ModelCapability:
        return ModelCapability(
            identifier=self._identifier,
            name="Mock Specialist",
            version="1.0.0",
            task=self._task,
            supported_modalities=[ModalityType.OPTICAL],
            supported_input_count=[1],
            requires_gpu=False,
            vram_budget_mb=0,
            confidence_available=True,
        )

    def load(self) -> None:
        self._is_loaded = True

    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        return SpecialistOutput(
            specialist_id=self._identifier,
            success=True,
            answer_text="Mock answer",
            confidence=0.95,
        )

    def health_check(self) -> bool:
        return self._healthy


def test_registry_registration_and_retrieval():
    """Verify specialists can be registered, retrieved, and queried."""
    reg = ModelRegistry()
    spec1 = MockSpecialist("RS_VQA", TaskType.VQA)
    spec2 = MockSpecialist("RS_GROUND", TaskType.GROUNDING)

    reg.register(spec1)
    reg.register(spec2)

    assert reg.get("RS_VQA") is spec1
    assert reg.get("RS_GROUND") is spec2
    assert reg.get("NONEXISTENT") is None

    # Capability listing
    caps = reg.list_capabilities()
    assert len(caps) == 2
    ids = [c.identifier for c in caps]
    assert "RS_VQA" in ids
    assert "RS_GROUND" in ids


def test_registry_find_specialists():
    """Verify finding specialists by task, modality, and input count."""
    reg = ModelRegistry()
    vqa = MockSpecialist("RS_VQA", TaskType.VQA)
    ground = MockSpecialist("RS_GROUND", TaskType.GROUNDING)

    reg.register(vqa)
    reg.register(ground)

    matches = reg.find_specialists(TaskType.VQA, ModalityType.OPTICAL, input_count=1)
    assert len(matches) == 1
    assert matches[0].capability.identifier == "RS_VQA"

    # No match for SAR
    no_matches = reg.find_specialists(TaskType.VQA, ModalityType.SAR)
    assert len(no_matches) == 0


def test_registry_health_checks():
    """Verify health check aggregation across specialists."""
    reg = ModelRegistry()
    healthy = MockSpecialist("HEALTHY")
    unhealthy = MockSpecialist("UNHEALTHY")
    unhealthy._healthy = False

    reg.register(healthy)
    reg.register(unhealthy)

    status = reg.health_check_all()
    assert status["HEALTHY"] is True
    assert status["UNHEALTHY"] is False


def test_registry_unregister_and_clear():
    """Verify clean unregistration and clearing."""
    reg = ModelRegistry()
    spec = MockSpecialist("SPEC")
    reg.register(spec)
    assert reg.get("SPEC") is not None

    removed = reg.unregister("SPEC")
    assert removed is spec
    assert reg.get("SPEC") is None

    reg.register(spec)
    reg.clear()
    assert reg.list_capabilities() == []
