"""Abstract base interface for all specialist models.

Adheres to AGENTS.md Rule 5 (Specialist Model Architecture) ensuring
uniform inputs, outputs, error handling, and memory cleanup.
"""

from abc import ABC, abstractmethod
from typing import Optional

from backend.agent.schema import ModelCapability, SpecialistInput, SpecialistOutput


class BaseSpecialist(ABC):
    """Abstract Base Specialist defining the uniform specialist model contract."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._is_loaded = False

    @property
    @abstractmethod
    def capability(self) -> ModelCapability:
        """Returns the declared capabilities and metadata of this specialist."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Loads weights, adapters, or preprocessors into memory/device."""
        pass

    @abstractmethod
    def predict(self, inputs: SpecialistInput) -> SpecialistOutput:
        """Executes specialist inference and returns structured output with evidence."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Verifies if the specialist is operational and ready for inference."""
        pass

    def cleanup(self) -> None:
        """Optional hook to free GPU/RAM buffers after execution."""
        pass

    @property
    def is_loaded(self) -> bool:
        """Indicates whether weights are currently resident in memory."""
        return self._is_loaded
