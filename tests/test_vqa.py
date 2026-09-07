"""Unit tests for the RemoteSensingVQASpecialist interface and lifecycle."""

import pytest
from backend.agent.registry import ModelRegistry
from backend.agent.schema import ModalityType, TaskType
from backend.models.base import BaseSpecialist
from backend.models.vqa import RemoteSensingVQASpecialist


def test_vqa_subclasses_base_specialist():
    """Verify VQA specialist implements the BaseSpecialist interface."""
    specialist = RemoteSensingVQASpecialist()
    assert isinstance(specialist, BaseSpecialist)
    assert specialist.capability.identifier == "RS_VQA"
    assert specialist.capability.task == TaskType.VQA
    assert ModalityType.OPTICAL in specialist.capability.supported_modalities
    assert 1 in specialist.capability.supported_input_count


def test_vqa_registration_in_registry():
    """Verify VQA specialist can be registered and retrieved from ModelRegistry."""
    reg = ModelRegistry()
    specialist = RemoteSensingVQASpecialist()
    reg.register(specialist)

    retrieved = reg.get("RS_VQA")
    assert retrieved is specialist

    matches = reg.find_specialists(task=TaskType.VQA, modality=ModalityType.OPTICAL, input_count=1)
    assert len(matches) == 1
    assert matches[0] is specialist


def test_vqa_health_check_and_cleanup():
    """Verify health check and cleanup lifecycle hooks."""
    specialist = RemoteSensingVQASpecialist()
    assert specialist.health_check() is True
    # Cleanup should run safely even if not loaded
    specialist.cleanup()
    assert specialist.is_loaded is False
